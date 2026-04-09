#ifndef LTST_V1_COMMANDS_H
#define LTST_V1_COMMANDS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ltst_v1_command_type_t {
    LTST_V1_CMD_NONE = 0,
    LTST_V1_CMD_HELP,
    LTST_V1_CMD_STATUS,
    LTST_V1_CMD_TEST,
    LTST_V1_CMD_MODE,
    LTST_V1_CMD_PACKET,
    LTST_V1_CMD_HMM,
    LTST_V1_CMD_RECORD,
    LTST_V1_CMD_INVALID,
} ltst_v1_command_type_t;

typedef enum ltst_v1_mode_t {
    LTST_V1_MODE_SELFTEST = 0,
    LTST_V1_MODE_REPLAY_PACKETS,
    LTST_V1_MODE_REPLAY_SAMPLES,
    LTST_V1_MODE_ADC_LIVE,
} ltst_v1_mode_t;

typedef struct ltst_v1_packet_command_t {
    uint32_t beat_index;
    float poincare_b;
    float drift_norm;
    float energy_asym;
    uint16_t flags;
} ltst_v1_packet_command_t;

typedef struct ltst_v1_command_t {
    ltst_v1_command_type_t type;
    ltst_v1_mode_t mode;
    ltst_v1_packet_command_t packet;
    bool hmm_enable;
    char record_name[24];
    char error[96];
} ltst_v1_command_t;

bool ltst_v1_parse_command(const char *line, ltst_v1_command_t *out_cmd);
const char *ltst_v1_mode_name(ltst_v1_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_COMMANDS_H */
