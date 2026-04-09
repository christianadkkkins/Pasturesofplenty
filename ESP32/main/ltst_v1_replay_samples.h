#ifndef LTST_V1_REPLAY_SAMPLES_H
#define LTST_V1_REPLAY_SAMPLES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ltst_v1_replay_trace_t {
    const char *record_name;
    uint32_t sample_count;
    uint32_t trace_index;
    bool generated;
} ltst_v1_replay_trace_t;

size_t ltst_v1_replay_trace_count(void);
bool ltst_v1_replay_trace_by_index(size_t index, ltst_v1_replay_trace_t *out_trace);
bool ltst_v1_replay_find_trace(const char *record_name, ltst_v1_replay_trace_t *out_trace);
float ltst_v1_replay_trace_sample(const ltst_v1_replay_trace_t *trace, uint32_t sample_index);

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_REPLAY_SAMPLES_H */
