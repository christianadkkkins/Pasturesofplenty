#ifndef LTST_V1_REPLAY_SAMPLES_GENERATED_H
#define LTST_V1_REPLAY_SAMPLES_GENERATED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltst_v1_generated_trace_t {
    const char *record_name;
    const int16_t *samples_q15;
    uint32_t sample_count;
    float sample_scale;
} ltst_v1_generated_trace_t;

#define LTST_V1_GENERATED_TRACE_COUNT 0u
static const ltst_v1_generated_trace_t LTST_V1_GENERATED_TRACES[1] = {
    {0, 0, 0u, 1.0f},
};

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_REPLAY_SAMPLES_GENERATED_H */
