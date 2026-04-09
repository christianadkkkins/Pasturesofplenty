#include "ltst_v1_commands.h"

#include <stdio.h>
#include <string.h>

static void copy_error(char *dst, size_t dst_len, const char *message) {
    if (dst == NULL || dst_len == 0u) {
        return;
    }
    if (message == NULL) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_len, "%s", message);
}

const char *ltst_v1_mode_name(ltst_v1_mode_t mode) {
    switch (mode) {
        case LTST_V1_MODE_SELFTEST: return "selftest";
        case LTST_V1_MODE_REPLAY_PACKETS: return "replay_packets";
        case LTST_V1_MODE_REPLAY_SAMPLES: return "replay_samples";
        case LTST_V1_MODE_ADC_LIVE: return "adc_live";
        default: return "unknown";
    }
}

bool ltst_v1_parse_command(const char *line, ltst_v1_command_t *out_cmd) {
    if (line == NULL || out_cmd == NULL) {
        return false;
    }

    memset(out_cmd, 0, sizeof(*out_cmd));
    out_cmd->type = LTST_V1_CMD_INVALID;
    out_cmd->mode = LTST_V1_MODE_SELFTEST;

    char keyword[24] = {0};
    if (sscanf(line, "%23s", keyword) != 1) {
        out_cmd->type = LTST_V1_CMD_NONE;
        return true;
    }

    if (strcmp(keyword, "help") == 0) {
        out_cmd->type = LTST_V1_CMD_HELP;
        return true;
    }
    if (strcmp(keyword, "status") == 0) {
        out_cmd->type = LTST_V1_CMD_STATUS;
        return true;
    }
    if (strcmp(keyword, "test") == 0) {
        out_cmd->type = LTST_V1_CMD_TEST;
        return true;
    }

    if (strcmp(keyword, "mode") == 0) {
        char mode_name[24] = {0};
        if (sscanf(line, "mode %23s", mode_name) != 1) {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "usage: mode selftest|replay|replay_packets|replay_samples|adc|adc_live");
            return false;
        }
        if (strcmp(mode_name, "selftest") == 0) {
            out_cmd->mode = LTST_V1_MODE_SELFTEST;
        } else if (strcmp(mode_name, "replay") == 0 || strcmp(mode_name, "replay_packets") == 0) {
            out_cmd->mode = LTST_V1_MODE_REPLAY_PACKETS;
        } else if (strcmp(mode_name, "replay_samples") == 0) {
            out_cmd->mode = LTST_V1_MODE_REPLAY_SAMPLES;
        } else if (strcmp(mode_name, "adc") == 0 || strcmp(mode_name, "adc_live") == 0) {
            out_cmd->mode = LTST_V1_MODE_ADC_LIVE;
        } else {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "unknown mode");
            return false;
        }
        out_cmd->type = LTST_V1_CMD_MODE;
        return true;
    }

    if (strcmp(keyword, "hmm") == 0) {
        char onoff[16] = {0};
        if (sscanf(line, "hmm %15s", onoff) != 1) {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "usage: hmm on|off");
            return false;
        }
        if (strcmp(onoff, "on") == 0) {
            out_cmd->hmm_enable = true;
        } else if (strcmp(onoff, "off") == 0) {
            out_cmd->hmm_enable = false;
        } else {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "usage: hmm on|off");
            return false;
        }
        out_cmd->type = LTST_V1_CMD_HMM;
        return true;
    }

    if (strcmp(keyword, "record") == 0) {
        char record_name[24] = {0};
        if (sscanf(line, "record %23s", record_name) != 1) {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "usage: record <record_name>");
            return false;
        }
        snprintf(out_cmd->record_name, sizeof(out_cmd->record_name), "%s", record_name);
        out_cmd->type = LTST_V1_CMD_RECORD;
        return true;
    }

    if (strcmp(keyword, "packet") == 0) {
        unsigned int beat_index = 0u;
        float poincare_b = 0.0f;
        float drift_norm = 0.0f;
        float energy_asym = 0.0f;
        unsigned int flags = 0u;
        if (sscanf(line, "packet %u %f %f %f %x", &beat_index, &poincare_b, &drift_norm, &energy_asym, &flags) != 5) {
            copy_error(out_cmd->error, sizeof(out_cmd->error), "usage: packet <beat_index> <poincare_b> <drift_norm> <energy_asym> <flags_hex>");
            return false;
        }
        out_cmd->type = LTST_V1_CMD_PACKET;
        out_cmd->packet.beat_index = (uint32_t)beat_index;
        out_cmd->packet.poincare_b = poincare_b;
        out_cmd->packet.drift_norm = drift_norm;
        out_cmd->packet.energy_asym = energy_asym;
        out_cmd->packet.flags = (uint16_t)(flags & 0xFFFFu);
        return true;
    }

    copy_error(out_cmd->error, sizeof(out_cmd->error), "unknown command");
    return false;
}
