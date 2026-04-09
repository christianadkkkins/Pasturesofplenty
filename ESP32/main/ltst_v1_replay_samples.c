#include "ltst_v1_replay_samples.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "../generated/ltst_v1_replay_samples_generated.h"

#define LTST_V1_REPLAY_SAMPLE_RATE_HZ 250.0f
#define LTST_V1_FALLBACK_TRACE_COUNT 4u
#define LTST_V1_TWO_PI 6.28318530717958647692f

typedef struct ltst_v1_fallback_trace_t {
    const char *record_name;
    uint32_t sample_count;
    float bpm;
    float amplitude;
    float t_wave;
    float p_wave;
    float baseline_phase;
} ltst_v1_fallback_trace_t;

static const ltst_v1_fallback_trace_t LTST_V1_FALLBACK_TRACES[LTST_V1_FALLBACK_TRACE_COUNT] = {
    {"s20021", 3000u, 74.0f, 1.00f, 0.22f, 0.12f, 0.00f},
    {"s20041", 3000u, 72.0f, 0.92f, 0.18f, 0.10f, 0.35f},
    {"s20151", 3000u, 69.0f, 0.88f, 0.15f, 0.08f, 0.70f},
    {"s30742", 3000u, 67.0f, 0.84f, 0.12f, 0.07f, 1.05f},
};

static float pulse(float phase, float center, float width) {
    const float delta = phase - center;
    return expf(-(delta * delta) / (2.0f * width * width));
}

static float fallback_sample(const ltst_v1_fallback_trace_t *trace, uint32_t sample_index) {
    const float cycles = ((float)sample_index / LTST_V1_REPLAY_SAMPLE_RATE_HZ) * (trace->bpm / 60.0f);
    const float phase = cycles - floorf(cycles);

    const float baseline = 0.02f * sinf(LTST_V1_TWO_PI * (0.3f * cycles + trace->baseline_phase));
    const float p = trace->p_wave * pulse(phase, 0.18f, 0.035f);
    const float q = -0.18f * pulse(phase, 0.36f, 0.010f);
    const float r = 1.00f * pulse(phase, 0.40f, 0.008f);
    const float s = -0.22f * pulse(phase, 0.44f, 0.012f);
    const float t = trace->t_wave * pulse(phase, 0.68f, 0.070f);

    return trace->amplitude * (baseline + p + q + r + s + t);
}

size_t ltst_v1_replay_trace_count(void) {
    if (LTST_V1_GENERATED_TRACE_COUNT > 0u) {
        return (size_t)LTST_V1_GENERATED_TRACE_COUNT;
    }
    return (size_t)LTST_V1_FALLBACK_TRACE_COUNT;
}

bool ltst_v1_replay_trace_by_index(size_t index, ltst_v1_replay_trace_t *out_trace) {
    if (out_trace == NULL) {
        return false;
    }
    memset(out_trace, 0, sizeof(*out_trace));
    if (LTST_V1_GENERATED_TRACE_COUNT > 0u) {
        if (index >= (size_t)LTST_V1_GENERATED_TRACE_COUNT) {
            return false;
        }
        out_trace->record_name = LTST_V1_GENERATED_TRACES[index].record_name;
        out_trace->sample_count = LTST_V1_GENERATED_TRACES[index].sample_count;
        out_trace->trace_index = (uint32_t)index;
        out_trace->generated = true;
        return true;
    }
    if (index >= (size_t)LTST_V1_FALLBACK_TRACE_COUNT) {
        return false;
    }
    out_trace->record_name = LTST_V1_FALLBACK_TRACES[index].record_name;
    out_trace->sample_count = LTST_V1_FALLBACK_TRACES[index].sample_count;
    out_trace->trace_index = (uint32_t)index;
    out_trace->generated = false;
    return true;
}

bool ltst_v1_replay_find_trace(const char *record_name, ltst_v1_replay_trace_t *out_trace) {
    if (record_name == NULL || out_trace == NULL) {
        return false;
    }
    for (size_t idx = 0; idx < ltst_v1_replay_trace_count(); ++idx) {
        ltst_v1_replay_trace_t trace;
        if (!ltst_v1_replay_trace_by_index(idx, &trace)) {
            continue;
        }
        if (trace.record_name != NULL && strcmp(trace.record_name, record_name) == 0) {
            *out_trace = trace;
            return true;
        }
    }
    return false;
}

float ltst_v1_replay_trace_sample(const ltst_v1_replay_trace_t *trace, uint32_t sample_index) {
    if (trace == NULL || trace->sample_count == 0u) {
        return 0.0f;
    }
    const uint32_t clamped = (sample_index < trace->sample_count) ? sample_index : (trace->sample_count - 1u);
    if (trace->generated) {
        const ltst_v1_generated_trace_t *generated = &LTST_V1_GENERATED_TRACES[trace->trace_index];
        if (generated->samples_q15 == NULL || generated->sample_scale == 0.0f) {
            return 0.0f;
        }
        return (float)generated->samples_q15[clamped] / generated->sample_scale;
    }
    return fallback_sample(&LTST_V1_FALLBACK_TRACES[trace->trace_index], clamped);
}
