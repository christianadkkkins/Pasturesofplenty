#ifndef LTST_V1_LIVE_PIPELINE_H
#define LTST_V1_LIVE_PIPELINE_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

#include "ltst_v1_commands.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltst_v1_live_output_t {
    bool packet_ready;
    bool feature_valid;
    bool hmm_log_ready;
    bool beat_log_ready;
    uint32_t beat_index;
    uint32_t sample_index;
    uint32_t rr_samples;
    float poincare_b;
    float drift_norm;
    float energy_asym;
    uint16_t quality_flags;
    float detector_threshold;
    float detector_confidence;
    const char *hmm_state_name;
    float hmm_score;
    uint32_t hmm_episode_id;
    uint32_t hmm_episode_start_beat;
    uint32_t hmm_active_duration_beats;
} ltst_v1_live_output_t;

typedef struct ltst_v1_live_status_t {
    uint64_t samples_acquired;
    uint64_t timer_misses;
    uint64_t adc_failures;
    uint64_t dropped_samples;
    uint32_t candidate_beats;
    uint32_t accepted_beats;
    uint32_t searchback_hits;
    uint32_t emitted_packets;
    uint32_t last_beat_index;
    uint32_t last_beat_sample;
    uint32_t last_rr_samples;
    float last_detector_threshold;
    float last_detector_confidence;
    const char *selected_record;
    uint32_t replay_sample_index;
    uint32_t replay_sample_count;
    bool replay_complete;
    bool hmm_enabled;
    const char *hmm_state_name;
    float hmm_score;
    uint32_t hmm_episode_id;
    uint32_t hmm_episode_start_beat;
    uint32_t hmm_active_duration_beats;
} ltst_v1_live_status_t;

esp_err_t ltst_v1_live_pipeline_init(void);
void ltst_v1_live_pipeline_reset_for_mode(ltst_v1_mode_t mode);
void ltst_v1_live_pipeline_note_timer_ticks(uint32_t tick_count);
esp_err_t ltst_v1_live_pipeline_tick(ltst_v1_mode_t mode, ltst_v1_live_output_t *out_output);
void ltst_v1_live_pipeline_get_status(ltst_v1_live_status_t *out_status);
bool ltst_v1_live_pipeline_set_replay_record(const char *record_name);
const char *ltst_v1_live_pipeline_selected_record(void);
bool ltst_v1_live_pipeline_set_hmm_enabled(bool enabled);
bool ltst_v1_live_pipeline_hmm_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_LIVE_PIPELINE_H */
