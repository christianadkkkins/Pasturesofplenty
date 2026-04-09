#include "ltst_v1_live_pipeline.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "esp_adc/adc_oneshot.h"

#include "ltst_v1_feature_packet.h"
#include "ltst_v1_replay_samples.h"

#define LTST_V1_SAMPLE_RATE_HZ 250.0f
#define LTST_V1_LAG 4u
#define LTST_V1_DIM 9u
#define LTST_V1_SAMPLE_BUFFER_LEN 512u
#define LTST_V1_MWI_LEN 20u
#define LTST_V1_RR_HISTORY_LEN 8u
#define LTST_V1_QRS_REFRACTORY_SAMPLES 50u
#define LTST_V1_MIN_RR_SAMPLES 50u
#define LTST_V1_PEAK_REFINE_HALF_WINDOW 10u
#define LTST_V1_SEARCHBACK_MULTIPLIER 1.66f
#define LTST_V1_BASELINE_ALPHA 0.02f
#define LTST_V1_DC_ALPHA 0.005f
#define LTST_V1_MIN_SCALE 0.05f
#define LTST_V1_ENERGY_EPS 1e-6f
#define LTST_V1_HMM_WARMUP_BEATS 16u
#define LTST_V1_ADC_UNIT ADC_UNIT_1
#define LTST_V1_ADC_CHANNEL ADC_CHANNEL_6
#define LTST_V1_ADC_ATTEN ADC_ATTEN_DB_12
#define LTST_V1_ADC_BITWIDTH ADC_BITWIDTH_12

typedef enum ltst_v1_hmm_state_t {
    LTST_V1_HMM_BASELINE = 0,
    LTST_V1_HMM_TRANSITION = 1,
    LTST_V1_HMM_ACTIVE = 2,
} ltst_v1_hmm_state_t;

typedef struct ltst_v1_pipeline_state_t {
    float sample_buffer[LTST_V1_SAMPLE_BUFFER_LEN];
    uint16_t sample_head;
    uint16_t sample_count;
    uint32_t next_sample_index;

    float sq_window[LTST_V1_MWI_LEN];
    uint8_t sq_head;
    uint8_t sq_count;
    float sq_sum;

    float prev_centered;
    bool have_prev_centered;
    float signal_peak;
    float noise_peak;
    float threshold;

    bool candidate_active;
    uint32_t candidate_start_sample;
    uint32_t candidate_peak_sample;
    float candidate_peak_env;
    float candidate_peak_abs_signal;

    bool searchback_valid;
    uint32_t searchback_peak_sample;
    float searchback_peak_env;
    float searchback_peak_abs_signal;

    bool have_last_accepted;
    uint32_t last_accepted_sample;
    uint32_t beat_index;
    uint32_t rr_history[LTST_V1_RR_HISTORY_LEN];
    uint8_t rr_head;
    uint8_t rr_count;

    float dc_estimate;
    bool dc_initialized;

    ltst_v1_replay_trace_t replay_trace;
    uint32_t replay_position;
    bool replay_complete;

    adc_oneshot_unit_handle_t adc_handle;
    bool adc_ready;

    bool hmm_enabled;
    bool hmm_initialized;
    float hmm_dp[3];
    ltst_v1_hmm_state_t hmm_state;
    uint32_t hmm_episode_id;
    uint32_t hmm_episode_start_beat;
    uint32_t hmm_active_duration_beats;
    float hmm_score;
    float baseline_center[4];
    float baseline_scale[4];
    float prev_rel[4];
    float prev_vel[4];
    uint32_t valid_feature_beats;

    ltst_v1_live_status_t stats;
} ltst_v1_pipeline_state_t;

static ltst_v1_pipeline_state_t g_state;

static const char *const LTST_V1_HMM_STATE_NAMES[] = {
    "baseline",
    "transition",
    "active",
};

static const float LTST_V1_HMM_START_LOGP[3] = {
    -0.00501254f,
    -5.29831737f,
    -13.81551075f,
};

static const float LTST_V1_HMM_TRANS_LOGP[3][3] = {
    {-0.06187540f, -2.81341072f, -13.81551075f},
    {-2.52572864f, -0.24846136f, -1.96610569f},
    {-13.81551075f, -2.12026354f, -0.12783337f},
};

static int wrap_sample_index(int index) {
    while (index < 0) {
        index += (int)LTST_V1_SAMPLE_BUFFER_LEN;
    }
    return index % (int)LTST_V1_SAMPLE_BUFFER_LEN;
}

static void zero_output(ltst_v1_live_output_t *out_output) {
    if (out_output != NULL) {
        memset(out_output, 0, sizeof(*out_output));
    }
}

static const char *hmm_state_name(ltst_v1_hmm_state_t state) {
    return LTST_V1_HMM_STATE_NAMES[(int)state];
}

static void refresh_status_selected_record(void) {
    g_state.stats.selected_record = g_state.replay_trace.record_name;
    g_state.stats.replay_sample_index = g_state.replay_position;
    g_state.stats.replay_sample_count = g_state.replay_trace.sample_count;
    g_state.stats.replay_complete = g_state.replay_complete;
    g_state.stats.hmm_enabled = g_state.hmm_enabled;
    g_state.stats.hmm_state_name = hmm_state_name(g_state.hmm_state);
    g_state.stats.hmm_score = g_state.hmm_score;
    g_state.stats.hmm_episode_id = g_state.hmm_episode_id;
    g_state.stats.hmm_episode_start_beat = g_state.hmm_episode_start_beat;
    g_state.stats.hmm_active_duration_beats = g_state.hmm_active_duration_beats;
}

static void reset_processing_state(void) {
    const bool hmm_enabled = g_state.hmm_enabled;
    const bool adc_ready = g_state.adc_ready;
    const adc_oneshot_unit_handle_t adc_handle = g_state.adc_handle;
    const ltst_v1_replay_trace_t replay_trace = g_state.replay_trace;

    memset(&g_state, 0, sizeof(g_state));

    g_state.hmm_enabled = hmm_enabled;
    g_state.adc_ready = adc_ready;
    g_state.adc_handle = adc_handle;
    g_state.replay_trace = replay_trace;
    g_state.signal_peak = 1e-3f;
    g_state.noise_peak = 1e-5f;
    g_state.threshold = 2.5e-4f;
    g_state.hmm_state = LTST_V1_HMM_BASELINE;
    refresh_status_selected_record();
}

static bool sample_buffer_get(uint32_t absolute_index, float *out_value) {
    if (out_value == NULL || g_state.sample_count == 0u || g_state.next_sample_index == 0u) {
        return false;
    }
    const uint32_t newest = g_state.next_sample_index - 1u;
    const uint32_t oldest = newest + 1u - (uint32_t)g_state.sample_count;
    if (absolute_index < oldest || absolute_index > newest) {
        return false;
    }
    const uint32_t delta = newest - absolute_index;
    const int pos = wrap_sample_index((int)g_state.sample_head - 1 - (int)delta);
    *out_value = g_state.sample_buffer[pos];
    return true;
}

static void push_centered_sample(float centered_signal) {
    g_state.sample_buffer[g_state.sample_head] = centered_signal;
    g_state.sample_head = (uint16_t)((g_state.sample_head + 1u) % LTST_V1_SAMPLE_BUFFER_LEN);
    if (g_state.sample_count < LTST_V1_SAMPLE_BUFFER_LEN) {
        ++g_state.sample_count;
    }
}

static float current_rr_median(void) {
    if (g_state.rr_count == 0u) {
        return LTST_V1_SAMPLE_RATE_HZ * 0.80f;
    }
    uint32_t tmp[LTST_V1_RR_HISTORY_LEN];
    for (uint8_t i = 0u; i < g_state.rr_count; ++i) {
        tmp[i] = g_state.rr_history[i];
    }
    for (uint8_t i = 1u; i < g_state.rr_count; ++i) {
        uint32_t key = tmp[i];
        int j = (int)i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            --j;
        }
        tmp[j + 1] = key;
    }
    if ((g_state.rr_count & 1u) != 0u) {
        return (float)tmp[g_state.rr_count / 2u];
    }
    const uint32_t a = tmp[g_state.rr_count / 2u - 1u];
    const uint32_t b = tmp[g_state.rr_count / 2u];
    return 0.5f * (float)(a + b);
}

static void record_rr(uint32_t rr_samples) {
    if (rr_samples == 0u) {
        return;
    }
    g_state.rr_history[g_state.rr_head] = rr_samples;
    g_state.rr_head = (uint8_t)((g_state.rr_head + 1u) % LTST_V1_RR_HISTORY_LEN);
    if (g_state.rr_count < LTST_V1_RR_HISTORY_LEN) {
        ++g_state.rr_count;
    }
}

static uint32_t refine_peak_sample(uint32_t approx_sample) {
    uint32_t best_sample = approx_sample;
    float best_abs = 0.0f;
    const uint32_t lo = (approx_sample > LTST_V1_PEAK_REFINE_HALF_WINDOW) ? (approx_sample - LTST_V1_PEAK_REFINE_HALF_WINDOW) : 0u;
    const uint32_t hi = approx_sample + LTST_V1_PEAK_REFINE_HALF_WINDOW;
    for (uint32_t sample = lo; sample <= hi; ++sample) {
        float value = 0.0f;
        if (!sample_buffer_get(sample, &value)) {
            continue;
        }
        const float abs_value = fabsf(value);
        if (abs_value >= best_abs) {
            best_abs = abs_value;
            best_sample = sample;
        }
    }
    return best_sample;
}

static bool compute_primitives(uint32_t beat_sample, uint32_t rr_samples, ltst_v1_live_output_t *out_output) {
    uint16_t flags = 0u;
    float poincare_b = 0.0f;
    float drift_norm = 0.0f;
    float energy_asym = 0.0f;

    if (rr_samples == 0u) {
        flags |= LTST_V1_WARMUP;
    }

    if (flags == 0u) {
        const uint32_t min_history = rr_samples + (LTST_V1_DIM - 1u) * LTST_V1_LAG;
        if (beat_sample < min_history) {
            flags |= LTST_V1_WARMUP;
        }
    }

    float sx = 0.0f;
    float sy = 0.0f;
    float sxx = 0.0f;
    float syy = 0.0f;
    float sxy = 0.0f;
    if (flags == 0u) {
        for (uint32_t i = 0u; i < LTST_V1_DIM; ++i) {
            float curr = 0.0f;
            float past = 0.0f;
            if (!sample_buffer_get(beat_sample - i * LTST_V1_LAG, &curr) ||
                !sample_buffer_get(beat_sample - rr_samples - i * LTST_V1_LAG, &past)) {
                flags |= LTST_V1_WARMUP;
                break;
            }
            sx += curr;
            sy += past;
            sxx += curr * curr;
            syy += past * past;
            sxy += curr * past;
        }
    }

    bool feature_valid = false;
    if ((flags & LTST_V1_WARMUP) == 0u) {
        const float inv_d = 1.0f / (float)LTST_V1_DIM;
        const float curr_energy = sxx - (sx * sx) * inv_d;
        const float past_energy = syy - (sy * sy) * inv_d;
        const float dot_centered = sxy - (sx * sy) * inv_d;

        if (curr_energy < LTST_V1_ENERGY_EPS) {
            flags |= LTST_V1_LOW_CURR_ENERGY;
        }
        if (past_energy < LTST_V1_ENERGY_EPS) {
            flags |= LTST_V1_LOW_PAST_ENERGY;
        }

        const float total_energy = curr_energy + past_energy;
        if (total_energy <= LTST_V1_ENERGY_EPS || curr_energy <= LTST_V1_ENERGY_EPS) {
            flags |= LTST_V1_DIV_GUARD;
        }

        if ((flags & (LTST_V1_LOW_CURR_ENERGY | LTST_V1_LOW_PAST_ENERGY | LTST_V1_DIV_GUARD)) == 0u) {
            const float drift_centered_sq = fmaxf(curr_energy + past_energy - 2.0f * dot_centered, 0.0f);
            energy_asym = (curr_energy - past_energy) / total_energy;
            drift_norm = drift_centered_sq / total_energy;
            poincare_b = dot_centered / curr_energy;
            feature_valid = true;
            flags |= LTST_V1_FEATURE_VALID;
        }
    }

    out_output->feature_valid = feature_valid;
    out_output->quality_flags = flags;
    out_output->poincare_b = poincare_b;
    out_output->drift_norm = drift_norm;
    out_output->energy_asym = energy_asym;
    return feature_valid;
}

static void update_hmm(ltst_v1_live_output_t *out_output) {
    out_output->hmm_log_ready = false;
    if (!g_state.hmm_enabled || !out_output->feature_valid) {
        refresh_status_selected_record();
        return;
    }

    const float values[4] = {
        out_output->poincare_b,
        out_output->drift_norm,
        out_output->energy_asym,
        out_output->poincare_b / (out_output->drift_norm + 1e-6f),
    };
    float rel[4] = {0};
    float vel[4] = {0};
    float acc[4] = {0};

    for (int i = 0; i < 4; ++i) {
        if (g_state.valid_feature_beats == 0u) {
            g_state.baseline_center[i] = values[i];
            g_state.baseline_scale[i] = 1.0f;
        }
        const float prev_center = g_state.baseline_center[i];
        const float prev_scale = fmaxf(g_state.baseline_scale[i], LTST_V1_MIN_SCALE);
        rel[i] = (values[i] - prev_center) / prev_scale;
        vel[i] = rel[i] - g_state.prev_rel[i];
        acc[i] = vel[i] - g_state.prev_vel[i];
    }

    const float level_norm = sqrtf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2] + rel[3] * rel[3]);
    const float velocity_norm = sqrtf(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2] + vel[3] * vel[3]);
    const float curvature_norm = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2] + acc[3] * acc[3]);
    const float transition_score = sqrtf(
        0.5f * level_norm * level_norm +
        1.0f * velocity_norm * velocity_norm +
        1.5f * curvature_norm * curvature_norm +
        0.75f * acc[3] * acc[3]
    );

    const float emission[3] = {
        -1.15f * transition_score - 0.60f * level_norm - 0.35f * curvature_norm,
         0.90f * transition_score + 0.95f * curvature_norm + 0.45f * velocity_norm - 0.25f * level_norm,
         0.80f * transition_score + 0.90f * level_norm + 0.20f * velocity_norm - 0.20f * curvature_norm,
    };

    if (!g_state.hmm_initialized || g_state.valid_feature_beats < LTST_V1_HMM_WARMUP_BEATS) {
        for (int i = 0; i < 3; ++i) {
            g_state.hmm_dp[i] = LTST_V1_HMM_START_LOGP[i] + emission[i];
        }
        g_state.hmm_state = LTST_V1_HMM_BASELINE;
        g_state.hmm_initialized = true;
    } else {
        float next_dp[3] = {0};
        for (int dst = 0; dst < 3; ++dst) {
            float best = g_state.hmm_dp[0] + LTST_V1_HMM_TRANS_LOGP[0][dst];
            for (int src = 1; src < 3; ++src) {
                const float candidate = g_state.hmm_dp[src] + LTST_V1_HMM_TRANS_LOGP[src][dst];
                if (candidate > best) {
                    best = candidate;
                }
            }
            next_dp[dst] = emission[dst] + best;
        }
        for (int i = 0; i < 3; ++i) {
            g_state.hmm_dp[i] = next_dp[i];
        }
        int best_state = 0;
        if (g_state.hmm_dp[1] > g_state.hmm_dp[best_state]) {
            best_state = 1;
        }
        if (g_state.hmm_dp[2] > g_state.hmm_dp[best_state]) {
            best_state = 2;
        }
        const ltst_v1_hmm_state_t previous_state = g_state.hmm_state;
        g_state.hmm_state = (ltst_v1_hmm_state_t)best_state;
        if (previous_state == LTST_V1_HMM_BASELINE && g_state.hmm_state != LTST_V1_HMM_BASELINE) {
            ++g_state.hmm_episode_id;
            g_state.hmm_episode_start_beat = out_output->beat_index;
        }
        if (g_state.hmm_state == LTST_V1_HMM_ACTIVE && g_state.hmm_episode_start_beat > 0u) {
            g_state.hmm_active_duration_beats = out_output->beat_index - g_state.hmm_episode_start_beat + 1u;
        } else if (g_state.hmm_state == LTST_V1_HMM_BASELINE) {
            g_state.hmm_active_duration_beats = 0u;
        }
    }

    const bool update_baseline = (g_state.hmm_state != LTST_V1_HMM_ACTIVE);
    if (update_baseline || g_state.valid_feature_beats < LTST_V1_HMM_WARMUP_BEATS) {
        for (int i = 0; i < 4; ++i) {
            const float center = g_state.baseline_center[i];
            g_state.baseline_center[i] = center + LTST_V1_BASELINE_ALPHA * (values[i] - center);
            const float deviation = fabsf(values[i] - g_state.baseline_center[i]);
            const float scale = g_state.baseline_scale[i];
            g_state.baseline_scale[i] = fmaxf(LTST_V1_MIN_SCALE, scale + LTST_V1_BASELINE_ALPHA * (deviation - scale));
        }
    }

    for (int i = 0; i < 4; ++i) {
        g_state.prev_rel[i] = rel[i];
        g_state.prev_vel[i] = vel[i];
    }

    ++g_state.valid_feature_beats;
    g_state.hmm_score = transition_score;
    out_output->hmm_log_ready = true;
    out_output->hmm_state_name = hmm_state_name(g_state.hmm_state);
    out_output->hmm_score = transition_score;
    out_output->hmm_episode_id = g_state.hmm_episode_id;
    out_output->hmm_episode_start_beat = g_state.hmm_episode_start_beat;
    out_output->hmm_active_duration_beats = g_state.hmm_active_duration_beats;
    refresh_status_selected_record();
}

static bool finalize_candidate(uint32_t approx_sample, float peak_env, bool searchback_hit, ltst_v1_live_output_t *out_output) {
    uint32_t refined_sample = refine_peak_sample(approx_sample);
    if (g_state.have_last_accepted && refined_sample <= g_state.last_accepted_sample) {
        return false;
    }
    if (g_state.have_last_accepted && (refined_sample - g_state.last_accepted_sample) < LTST_V1_QRS_REFRACTORY_SAMPLES) {
        return false;
    }

    const float threshold_snapshot = fmaxf(g_state.threshold, 1e-6f);
    const float confidence = peak_env / threshold_snapshot;

    g_state.signal_peak = 0.125f * peak_env + 0.875f * g_state.signal_peak;
    g_state.threshold = g_state.noise_peak + 0.25f * (g_state.signal_peak - g_state.noise_peak);

    ++g_state.stats.candidate_beats;
    ++g_state.stats.accepted_beats;
    if (searchback_hit) {
        ++g_state.stats.searchback_hits;
    }

    const uint32_t rr_samples = g_state.have_last_accepted ? (refined_sample - g_state.last_accepted_sample) : 0u;
    g_state.last_accepted_sample = refined_sample;
    g_state.have_last_accepted = true;
    record_rr(rr_samples);

    ++g_state.beat_index;
    out_output->packet_ready = true;
    out_output->beat_log_ready = true;
    out_output->beat_index = g_state.beat_index;
    out_output->sample_index = refined_sample;
    out_output->rr_samples = rr_samples;
    out_output->detector_threshold = threshold_snapshot;
    out_output->detector_confidence = confidence;

    g_state.stats.last_beat_index = g_state.beat_index;
    g_state.stats.last_beat_sample = refined_sample;
    g_state.stats.last_rr_samples = rr_samples;
    g_state.stats.last_detector_threshold = threshold_snapshot;
    g_state.stats.last_detector_confidence = confidence;

    compute_primitives(refined_sample, rr_samples, out_output);
    if (out_output->packet_ready) {
        ++g_state.stats.emitted_packets;
    }
    update_hmm(out_output);
    refresh_status_selected_record();
    return true;
}

static void process_detector_sample(float centered_signal, uint32_t sample_index, ltst_v1_live_output_t *out_output) {
    float diff = 0.0f;
    if (g_state.have_prev_centered) {
        diff = centered_signal - g_state.prev_centered;
    }
    g_state.prev_centered = centered_signal;
    g_state.have_prev_centered = true;

    const float squared = diff * diff;
    if (g_state.sq_count < LTST_V1_MWI_LEN) {
        ++g_state.sq_count;
    } else {
        g_state.sq_sum -= g_state.sq_window[g_state.sq_head];
    }
    g_state.sq_window[g_state.sq_head] = squared;
    g_state.sq_sum += squared;
    g_state.sq_head = (uint8_t)((g_state.sq_head + 1u) % LTST_V1_MWI_LEN);
    const float envelope = g_state.sq_sum / fmaxf((float)g_state.sq_count, 1.0f);

    if (!g_state.candidate_active && envelope < g_state.threshold) {
        g_state.noise_peak = 0.125f * envelope + 0.875f * g_state.noise_peak;
        g_state.threshold = g_state.noise_peak + 0.25f * (g_state.signal_peak - g_state.noise_peak);
    }

    if (g_state.have_last_accepted && (!g_state.searchback_valid || envelope >= g_state.searchback_peak_env)) {
        g_state.searchback_valid = true;
        g_state.searchback_peak_env = envelope;
        g_state.searchback_peak_sample = sample_index;
        g_state.searchback_peak_abs_signal = fabsf(centered_signal);
    }

    if (!g_state.candidate_active) {
        const bool refractory = g_state.have_last_accepted &&
            ((sample_index - g_state.last_accepted_sample) < LTST_V1_QRS_REFRACTORY_SAMPLES);
        if (!refractory && envelope >= g_state.threshold) {
            g_state.candidate_active = true;
            g_state.candidate_start_sample = sample_index;
            g_state.candidate_peak_sample = sample_index;
            g_state.candidate_peak_env = envelope;
            g_state.candidate_peak_abs_signal = fabsf(centered_signal);
        }
    } else {
        const float abs_signal = fabsf(centered_signal);
        if (envelope >= g_state.candidate_peak_env || abs_signal >= g_state.candidate_peak_abs_signal) {
            g_state.candidate_peak_env = envelope;
            g_state.candidate_peak_sample = sample_index;
            g_state.candidate_peak_abs_signal = abs_signal;
        }
        if (envelope < (0.5f * g_state.threshold)) {
            (void)finalize_candidate(g_state.candidate_peak_sample, g_state.candidate_peak_env, false, out_output);
            g_state.candidate_active = false;
            g_state.searchback_valid = false;
            g_state.searchback_peak_env = 0.0f;
        }
    }

    if (!out_output->packet_ready && g_state.have_last_accepted && !g_state.candidate_active && g_state.searchback_valid) {
        const float rr_median = current_rr_median();
        const float limit = LTST_V1_SEARCHBACK_MULTIPLIER * rr_median;
        if ((float)(sample_index - g_state.last_accepted_sample) > limit && g_state.searchback_peak_env >= (0.5f * g_state.threshold)) {
            if (finalize_candidate(g_state.searchback_peak_sample, g_state.searchback_peak_env, true, out_output)) {
                g_state.searchback_valid = false;
                g_state.searchback_peak_env = 0.0f;
            }
        }
    }
}

static esp_err_t ensure_adc_initialized(void) {
    if (g_state.adc_ready) {
        return ESP_OK;
    }

    adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = LTST_V1_ADC_UNIT,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    if (adc_oneshot_new_unit(&unit_cfg, &g_state.adc_handle) != ESP_OK) {
        g_state.adc_handle = NULL;
        g_state.adc_ready = false;
        return ESP_FAIL;
    }

    adc_oneshot_chan_cfg_t chan_cfg = {
        .bitwidth = LTST_V1_ADC_BITWIDTH,
        .atten = LTST_V1_ADC_ATTEN,
    };
    if (adc_oneshot_config_channel(g_state.adc_handle, LTST_V1_ADC_CHANNEL, &chan_cfg) != ESP_OK) {
        g_state.adc_ready = false;
        return ESP_FAIL;
    }

    g_state.adc_ready = true;
    return ESP_OK;
}

static bool read_source_sample(ltst_v1_mode_t mode, float *out_centered_signal, uint32_t *out_raw_adc) {
    if (out_centered_signal == NULL || out_raw_adc == NULL) {
        return false;
    }
    *out_centered_signal = 0.0f;
    *out_raw_adc = 0u;

    if (mode == LTST_V1_MODE_REPLAY_SAMPLES) {
        if (g_state.replay_trace.record_name == NULL || g_state.replay_complete) {
            return false;
        }
        if (g_state.replay_position >= g_state.replay_trace.sample_count) {
            g_state.replay_complete = true;
            refresh_status_selected_record();
            return false;
        }
        *out_centered_signal = ltst_v1_replay_trace_sample(&g_state.replay_trace, g_state.replay_position);
        ++g_state.replay_position;
        if (g_state.replay_position >= g_state.replay_trace.sample_count) {
            g_state.replay_complete = true;
        }
        refresh_status_selected_record();
        return true;
    }

    if (mode == LTST_V1_MODE_ADC_LIVE) {
        if (ensure_adc_initialized() != ESP_OK || !g_state.adc_ready) {
            ++g_state.stats.adc_failures;
            return false;
        }

        int raw = 0;
        if (adc_oneshot_read(g_state.adc_handle, LTST_V1_ADC_CHANNEL, &raw) != ESP_OK) {
            ++g_state.stats.adc_failures;
            return false;
        }
        *out_raw_adc = (uint32_t)raw;
        const float normalized = ((float)raw - 2048.0f) / 2048.0f;
        if (!g_state.dc_initialized) {
            g_state.dc_estimate = normalized;
            g_state.dc_initialized = true;
        } else {
            g_state.dc_estimate += LTST_V1_DC_ALPHA * (normalized - g_state.dc_estimate);
        }
        *out_centered_signal = normalized - g_state.dc_estimate;
        return true;
    }

    return false;
}

esp_err_t ltst_v1_live_pipeline_init(void) {
    memset(&g_state, 0, sizeof(g_state));
    g_state.hmm_enabled = true;
    g_state.signal_peak = 1e-3f;
    g_state.noise_peak = 1e-5f;
    g_state.threshold = 2.5e-4f;
    g_state.hmm_state = LTST_V1_HMM_BASELINE;

    ltst_v1_replay_trace_t trace;
    if (ltst_v1_replay_trace_by_index(0u, &trace)) {
        g_state.replay_trace = trace;
    }
    (void)ensure_adc_initialized();
    refresh_status_selected_record();
    return ESP_OK;
}

void ltst_v1_live_pipeline_reset_for_mode(ltst_v1_mode_t mode) {
    (void)mode;
    reset_processing_state();
}

void ltst_v1_live_pipeline_note_timer_ticks(uint32_t tick_count) {
    if (tick_count > 1u) {
        const uint32_t missed = tick_count - 1u;
        g_state.stats.timer_misses += missed;
        g_state.stats.dropped_samples += missed;
    }
}

esp_err_t ltst_v1_live_pipeline_tick(ltst_v1_mode_t mode, ltst_v1_live_output_t *out_output) {
    zero_output(out_output);
    float centered_signal = 0.0f;
    uint32_t raw_adc = 0u;
    if (!read_source_sample(mode, &centered_signal, &raw_adc)) {
        return ESP_OK;
    }

    (void)raw_adc;
    const uint32_t sample_index = g_state.next_sample_index;
    ++g_state.next_sample_index;
    ++g_state.stats.samples_acquired;
    push_centered_sample(centered_signal);
    process_detector_sample(centered_signal, sample_index, out_output);
    return ESP_OK;
}

void ltst_v1_live_pipeline_get_status(ltst_v1_live_status_t *out_status) {
    if (out_status == NULL) {
        return;
    }
    refresh_status_selected_record();
    *out_status = g_state.stats;
}

bool ltst_v1_live_pipeline_set_replay_record(const char *record_name) {
    ltst_v1_replay_trace_t trace;
    if (!ltst_v1_replay_find_trace(record_name, &trace)) {
        return false;
    }
    g_state.replay_trace = trace;
    reset_processing_state();
    refresh_status_selected_record();
    return true;
}

const char *ltst_v1_live_pipeline_selected_record(void) {
    return g_state.replay_trace.record_name;
}

bool ltst_v1_live_pipeline_set_hmm_enabled(bool enabled) {
    g_state.hmm_enabled = enabled;
    g_state.hmm_initialized = false;
    g_state.hmm_state = LTST_V1_HMM_BASELINE;
    g_state.hmm_episode_id = 0u;
    g_state.hmm_episode_start_beat = 0u;
    g_state.hmm_active_duration_beats = 0u;
    g_state.hmm_score = 0.0f;
    memset(g_state.hmm_dp, 0, sizeof(g_state.hmm_dp));
    memset(g_state.baseline_center, 0, sizeof(g_state.baseline_center));
    for (int i = 0; i < 4; ++i) {
        g_state.baseline_scale[i] = 1.0f;
        g_state.prev_rel[i] = 0.0f;
        g_state.prev_vel[i] = 0.0f;
    }
    g_state.valid_feature_beats = 0u;
    refresh_status_selected_record();
    return g_state.hmm_enabled;
}

bool ltst_v1_live_pipeline_hmm_enabled(void) {
    return g_state.hmm_enabled;
}
