#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/uart.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "ltst_v1_commands.h"
#include "ltst_v1_feature_packet.h"
#include "ltst_v1_live_pipeline.h"
#include "ltst_v1_transport.h"

#define LTST_V1_DATA_UART_PORT UART_NUM_1
#define LTST_V1_DATA_UART_TX_PIN 17
#define LTST_V1_DATA_UART_RX_PIN 16
#define LTST_V1_DATA_UART_BAUD 115200
#define LTST_V1_CONSOLE_UART_BAUD 115200
#define LTST_V1_DATA_UART_TX_BUFFER 1024
#define LTST_V1_DATA_UART_RX_BUFFER 1024
#define LTST_V1_SAMPLE_PERIOD_US 4000

static const char *TAG = "ltst_v1_bringup";

_Static_assert(sizeof(ltst_v1_feature_packet_t) == LTST_V1_PACKET_SIZE_BYTES, "LTST V1 packet must stay 12 bytes");
_Static_assert(LTST_V1_PACKET_VERSION == 0x0001u, "Unexpected LTST V1 packet version");

typedef struct ltst_v1_test_vector_t {
    uint32_t beat_index;
    float poincare_b;
    float drift_norm;
    float energy_asym;
    uint16_t quality_flags;
    const char *label;
} ltst_v1_test_vector_t;

typedef struct app_state_t {
    ltst_v1_mode_t mode;
    ltst_v1_feature_packet_t last_packet;
    ltst_v1_feature_float_view_t last_decoded;
    ltst_v1_frame_info_t last_frame;
    uint32_t last_beat_index;
    bool has_last_packet;
} app_state_t;

static app_state_t g_app = {
    .mode = LTST_V1_MODE_SELFTEST,
};

static TaskHandle_t g_acquisition_task = NULL;
static esp_timer_handle_t g_sample_timer = NULL;

static const ltst_v1_test_vector_t SELFTEST_VECTORS[] = {
    {1u, 0.8599f, 0.1229f,  0.0031f, LTST_V1_FEATURE_VALID, "loose_like"},
    {2u, 0.9742f, 0.0175f, -0.0007f, LTST_V1_FEATURE_VALID, "constrained_like"},
    {3u, 0.9955f, 0.0110f, -0.0047f, LTST_V1_FEATURE_VALID, "rigid_like"},
    {4u, 0.9984f, 0.0090f, -0.0002f, LTST_V1_FEATURE_VALID, "rigid_high"},
    {5u, 4.2f,    16.5f,    2.1f,    LTST_V1_FEATURE_VALID, "saturation"},
    {6u, 0.90f,   0.0f,     0.0f,    (uint16_t)(LTST_V1_WARMUP | LTST_V1_DIV_GUARD), "guarded_invalid"},
};

static void print_help(void) {
    printf(
        "commands:\n"
        "  help\n"
        "  status\n"
        "  test\n"
        "  mode selftest\n"
        "  mode replay\n"
        "  mode replay_packets\n"
        "  mode replay_samples\n"
        "  mode adc\n"
        "  mode adc_live\n"
        "  hmm on\n"
        "  hmm off\n"
        "  record <record_name>\n"
        "  packet <beat_index> <poincare_b> <drift_norm> <energy_asym> <flags_hex>\n"
    );
}

static void print_packet_debug(
    const char *source,
    const ltst_v1_feature_packet_t *packet,
    const ltst_v1_feature_float_view_t *decoded,
    const ltst_v1_frame_info_t *frame_info,
    float src_pb,
    float src_drift,
    float src_asym
) {
    printf(
        "[%s] beat=%" PRIu32
        " src(pb=%.6f drift=%.6f asym=%.6f)"
        " raw(pb=%" PRIu16 " drift=%" PRIu16 " asym=%" PRId16 ")"
        " dec(pb=%.6f drift=%.6f asym=%.6f)"
        " flags=0x%04X crc=0x%04X len=%u\n",
        source,
        packet->beat_index,
        src_pb,
        src_drift,
        src_asym,
        packet->poincare_b_uq2_14,
        packet->drift_norm_uq4_12,
        packet->energy_asym_sq1_14,
        decoded->poincare_b,
        decoded->drift_norm,
        decoded->energy_asym,
        packet->quality_flags,
        frame_info->crc16,
        frame_info->payload_len
    );
}

static const char *selected_record_safe(void) {
    const char *record = ltst_v1_live_pipeline_selected_record();
    return (record != NULL) ? record : "(none)";
}

static void emit_packet(uint32_t beat_index, float poincare_b, float drift_norm, float energy_asym, uint16_t flags, const char *source) {
    ltst_v1_feature_packet_t packet;
    ltst_v1_frame_info_t frame_info;

    ltst_v1_init_packet(&packet, beat_index, poincare_b, drift_norm, energy_asym, flags);
    const ltst_v1_feature_float_view_t decoded = ltst_v1_decode_packet(&packet);
    const int err = ltst_v1_send_packet(&packet, &frame_info);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to send packet: %d", err);
        return;
    }

    g_app.last_packet = packet;
    g_app.last_decoded = decoded;
    g_app.last_frame = frame_info;
    g_app.last_beat_index = beat_index;
    g_app.has_last_packet = true;

    print_packet_debug(source, &packet, &decoded, &frame_info, poincare_b, drift_norm, energy_asym);
}

static void run_selftest_burst(void) {
    printf("running self-test burst (%u vectors)\n", (unsigned int)(sizeof(SELFTEST_VECTORS) / sizeof(SELFTEST_VECTORS[0])));
    for (size_t i = 0; i < sizeof(SELFTEST_VECTORS) / sizeof(SELFTEST_VECTORS[0]); ++i) {
        const ltst_v1_test_vector_t *vec = &SELFTEST_VECTORS[i];
        emit_packet(vec->beat_index, vec->poincare_b, vec->drift_norm, vec->energy_asym, vec->quality_flags, vec->label);
    }
}

static void print_status(void) {
    const ltst_v1_transport_stats_t *transport_stats = ltst_v1_transport_stats();
    ltst_v1_live_status_t live = {0};
    ltst_v1_live_pipeline_get_status(&live);

    printf(
        "status: mode=%s packet_version=0x%04X console_baud=%d data_uart=%d tx=%d rx=%d data_baud=%d packets_sent=%" PRIu32 " bytes_sent=%" PRIu32 " frame_errors=%" PRIu32 " crc_errors=%" PRIu32 "\n",
        ltst_v1_mode_name(g_app.mode),
        LTST_V1_PACKET_VERSION,
        LTST_V1_CONSOLE_UART_BAUD,
        LTST_V1_DATA_UART_PORT,
        LTST_V1_DATA_UART_TX_PIN,
        LTST_V1_DATA_UART_RX_PIN,
        LTST_V1_DATA_UART_BAUD,
        transport_stats->packets_sent,
        transport_stats->bytes_sent,
        transport_stats->frame_errors,
        transport_stats->crc_errors
    );
    printf(
        "live: record=%s samples=%" PRIu64 " timer_misses=%" PRIu64 " adc_failures=%" PRIu64 " dropped=%" PRIu64 " candidates=%" PRIu32 " accepted=%" PRIu32 " searchback=%" PRIu32 " last_sample=%" PRIu32 " last_rr=%" PRIu32 " thr=%.6f conf=%.6f replay=%" PRIu32 "/%" PRIu32 " replay_complete=%u\n",
        (live.selected_record != NULL) ? live.selected_record : "(none)",
        live.samples_acquired,
        live.timer_misses,
        live.adc_failures,
        live.dropped_samples,
        live.candidate_beats,
        live.accepted_beats,
        live.searchback_hits,
        live.last_beat_sample,
        live.last_rr_samples,
        live.last_detector_threshold,
        live.last_detector_confidence,
        live.replay_sample_index,
        live.replay_sample_count,
        live.replay_complete ? 1u : 0u
    );
    printf(
        "hmm: enabled=%u state=%s score=%.6f episode=%" PRIu32 " start_beat=%" PRIu32 " active_duration=%" PRIu32 "\n",
        live.hmm_enabled ? 1u : 0u,
        (live.hmm_state_name != NULL) ? live.hmm_state_name : "baseline",
        live.hmm_score,
        live.hmm_episode_id,
        live.hmm_episode_start_beat,
        live.hmm_active_duration_beats
    );
    if (g_app.has_last_packet) {
        printf(
            "last_packet: beat=%" PRIu32 " pb=%.6f drift=%.6f asym=%.6f flags=0x%04X crc=0x%04X\n",
            g_app.last_beat_index,
            g_app.last_decoded.poincare_b,
            g_app.last_decoded.drift_norm,
            g_app.last_decoded.energy_asym,
            g_app.last_packet.quality_flags,
            g_app.last_frame.crc16
        );
    } else {
        printf("last_packet: none\n");
    }
}

static void sample_timer_callback(void *arg) {
    (void)arg;
    if (g_acquisition_task != NULL) {
        xTaskNotifyGive(g_acquisition_task);
    }
}

static void acquisition_task(void *arg) {
    (void)arg;
    ltst_v1_live_output_t output;
    while (true) {
        const uint32_t tick_count = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (tick_count == 0u) {
            continue;
        }
        ltst_v1_live_pipeline_note_timer_ticks(tick_count);
        if (g_app.mode != LTST_V1_MODE_REPLAY_SAMPLES && g_app.mode != LTST_V1_MODE_ADC_LIVE) {
            continue;
        }
        if (ltst_v1_live_pipeline_tick(g_app.mode, &output) != ESP_OK) {
            continue;
        }
        if (!output.packet_ready) {
            continue;
        }

        const char *source = (g_app.mode == LTST_V1_MODE_REPLAY_SAMPLES) ? "replay_samples" : "adc_live";
        emit_packet(output.beat_index, output.poincare_b, output.drift_norm, output.energy_asym, output.quality_flags, source);

        if (output.beat_log_ready) {
            printf(
                "[beat] record=%s beat=%" PRIu32 " sample=%" PRIu32 " rr=%" PRIu32 " feature_valid=%u pb=%.6f drift=%.6f asym=%.6f flags=0x%04X conf=%.6f thr=%.6f\n",
                selected_record_safe(),
                output.beat_index,
                output.sample_index,
                output.rr_samples,
                output.feature_valid ? 1u : 0u,
                output.poincare_b,
                output.drift_norm,
                output.energy_asym,
                output.quality_flags,
                output.detector_confidence,
                output.detector_threshold
            );
        }

        if (output.hmm_log_ready) {
            printf(
                "[hmm] record=%s beat=%" PRIu32 " state=%s score=%.6f episode=%" PRIu32 " start=%" PRIu32 " active_duration=%" PRIu32 "\n",
                selected_record_safe(),
                output.beat_index,
                output.hmm_state_name,
                output.hmm_score,
                output.hmm_episode_id,
                output.hmm_episode_start_beat,
                output.hmm_active_duration_beats
            );
        }
    }
}

static void command_task(void *arg) {
    (void)arg;
    char line[160];
    print_help();
    while (true) {
        printf("> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        ltst_v1_command_t cmd;
        if (!ltst_v1_parse_command(line, &cmd)) {
            printf("error: %s\n", cmd.error[0] != '\0' ? cmd.error : "parse failed");
            continue;
        }

        switch (cmd.type) {
            case LTST_V1_CMD_NONE:
                break;
            case LTST_V1_CMD_HELP:
                print_help();
                break;
            case LTST_V1_CMD_STATUS:
                print_status();
                break;
            case LTST_V1_CMD_TEST:
                run_selftest_burst();
                break;
            case LTST_V1_CMD_MODE:
                g_app.mode = cmd.mode;
                ltst_v1_live_pipeline_reset_for_mode(g_app.mode);
                if (g_app.mode == LTST_V1_MODE_REPLAY_SAMPLES || g_app.mode == LTST_V1_MODE_ADC_LIVE) {
                    ltst_v1_live_pipeline_set_hmm_enabled(true);
                }
                printf("mode set to %s\n", ltst_v1_mode_name(g_app.mode));
                break;
            case LTST_V1_CMD_PACKET:
                emit_packet(
                    cmd.packet.beat_index,
                    cmd.packet.poincare_b,
                    cmd.packet.drift_norm,
                    cmd.packet.energy_asym,
                    cmd.packet.flags,
                    "replay_packets"
                );
                break;
            case LTST_V1_CMD_HMM:
                if (g_app.mode == LTST_V1_MODE_REPLAY_PACKETS || g_app.mode == LTST_V1_MODE_SELFTEST) {
                    printf("hmm ignored in %s mode\n", ltst_v1_mode_name(g_app.mode));
                    break;
                }
                ltst_v1_live_pipeline_set_hmm_enabled(cmd.hmm_enable);
                printf("hmm %s\n", cmd.hmm_enable ? "on" : "off");
                break;
            case LTST_V1_CMD_RECORD:
                if (!ltst_v1_live_pipeline_set_replay_record(cmd.record_name)) {
                    printf("error: unknown record '%s'\n", cmd.record_name);
                } else {
                    printf("record set to %s\n", cmd.record_name);
                }
                break;
            case LTST_V1_CMD_INVALID:
            default:
                printf("error: %s\n", cmd.error[0] != '\0' ? cmd.error : "invalid command");
                break;
        }
    }
}

void app_main(void) {
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);

    const ltst_v1_transport_config_t transport_cfg = {
        .uart_port = LTST_V1_DATA_UART_PORT,
        .tx_pin = LTST_V1_DATA_UART_TX_PIN,
        .rx_pin = LTST_V1_DATA_UART_RX_PIN,
        .baud_rate = LTST_V1_DATA_UART_BAUD,
        .tx_buffer_size = LTST_V1_DATA_UART_TX_BUFFER,
        .rx_buffer_size = LTST_V1_DATA_UART_RX_BUFFER,
    };

    const int transport_err = ltst_v1_transport_init(&transport_cfg);
    if (transport_err != ESP_OK) {
        ESP_LOGE(TAG, "transport init failed: %d", transport_err);
        return;
    }
    if (ltst_v1_live_pipeline_init() != ESP_OK) {
        ESP_LOGE(TAG, "live pipeline init failed");
        return;
    }

    printf(
        "LTST V1 bring-up\n"
        "contract_version=0x%04X packet_size=%u frame_bytes=%u console_uart=0 data_uart=%d data_tx=%d data_rx=%d baud=%d\n",
        LTST_V1_PACKET_VERSION,
        (unsigned int)sizeof(ltst_v1_feature_packet_t),
        (unsigned int)LTST_V1_FRAME_SIZE_BYTES,
        LTST_V1_DATA_UART_PORT,
        LTST_V1_DATA_UART_TX_PIN,
        LTST_V1_DATA_UART_RX_PIN,
        LTST_V1_DATA_UART_BAUD
    );

    run_selftest_burst();

    xTaskCreatePinnedToCore(command_task, "ltst_cmd", 4096, NULL, 5, NULL, tskNO_AFFINITY);
    xTaskCreatePinnedToCore(acquisition_task, "ltst_acq", 6144, NULL, 6, &g_acquisition_task, tskNO_AFFINITY);

    const esp_timer_create_args_t timer_args = {
        .callback = &sample_timer_callback,
        .arg = NULL,
        .name = "ltst_sampler",
    };
    if (esp_timer_create(&timer_args, &g_sample_timer) != ESP_OK) {
        ESP_LOGE(TAG, "failed to create sample timer");
        return;
    }
    if (esp_timer_start_periodic(g_sample_timer, LTST_V1_SAMPLE_PERIOD_US) != ESP_OK) {
        ESP_LOGE(TAG, "failed to start sample timer");
        return;
    }
}
