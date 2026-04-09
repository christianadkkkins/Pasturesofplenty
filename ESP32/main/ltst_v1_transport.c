#include "ltst_v1_transport.h"

#include <string.h>

#include "esp_err.h"

static ltst_v1_transport_config_t g_cfg;
static ltst_v1_transport_stats_t g_stats;

int ltst_v1_transport_init(const ltst_v1_transport_config_t *cfg) {
    if (cfg == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    g_cfg = *cfg;
    memset(&g_stats, 0, sizeof(g_stats));

    const uart_config_t uart_cfg = {
        .baud_rate = cfg->baud_rate,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t err = uart_driver_install(cfg->uart_port, cfg->rx_buffer_size, cfg->tx_buffer_size, 0, NULL, 0);
    if (err != ESP_OK) {
        return err;
    }
    err = uart_param_config(cfg->uart_port, &uart_cfg);
    if (err != ESP_OK) {
        return err;
    }
    err = uart_set_pin(cfg->uart_port, cfg->tx_pin, cfg->rx_pin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    if (err != ESP_OK) {
        return err;
    }
    return ESP_OK;
}

uint16_t ltst_v1_crc16_ccitt_false(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (int bit = 0; bit < 8; ++bit) {
            if ((crc & 0x8000u) != 0u) {
                crc = (uint16_t)((crc << 1) ^ 0x1021u);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

size_t ltst_v1_build_frame(uint8_t *out_frame, size_t out_capacity, const ltst_v1_feature_packet_t *packet, ltst_v1_frame_info_t *info) {
    if (out_frame == NULL || packet == NULL || out_capacity < LTST_V1_FRAME_SIZE_BYTES) {
        return 0u;
    }

    out_frame[0] = LTST_V1_FRAME_SYNC0;
    out_frame[1] = LTST_V1_FRAME_SYNC1;
    out_frame[2] = LTST_V1_FRAME_VERSION;
    out_frame[3] = LTST_V1_MSG_TYPE_FEATURE_PACKET;
    out_frame[4] = (uint8_t)(LTST_V1_PACKET_SIZE_BYTES & 0xFFu);
    out_frame[5] = (uint8_t)((LTST_V1_PACKET_SIZE_BYTES >> 8) & 0xFFu);
    memcpy(&out_frame[LTST_V1_FRAME_HEADER_SIZE], packet, LTST_V1_PACKET_SIZE_BYTES);

    const uint16_t crc = ltst_v1_crc16_ccitt_false(&out_frame[2], 1u + 1u + 2u + LTST_V1_PACKET_SIZE_BYTES);
    out_frame[LTST_V1_FRAME_HEADER_SIZE + LTST_V1_PACKET_SIZE_BYTES + 0u] = (uint8_t)(crc & 0xFFu);
    out_frame[LTST_V1_FRAME_HEADER_SIZE + LTST_V1_PACKET_SIZE_BYTES + 1u] = (uint8_t)((crc >> 8) & 0xFFu);

    if (info != NULL) {
        info->version = LTST_V1_FRAME_VERSION;
        info->msg_type = LTST_V1_MSG_TYPE_FEATURE_PACKET;
        info->payload_len = LTST_V1_PACKET_SIZE_BYTES;
        info->crc16 = crc;
    }
    return LTST_V1_FRAME_SIZE_BYTES;
}

int ltst_v1_send_packet(const ltst_v1_feature_packet_t *packet, ltst_v1_frame_info_t *info) {
    uint8_t frame[LTST_V1_FRAME_SIZE_BYTES];
    const size_t frame_len = ltst_v1_build_frame(frame, sizeof(frame), packet, info);
    if (frame_len == 0u) {
        ++g_stats.frame_errors;
        return ESP_ERR_INVALID_ARG;
    }

    const int written = uart_write_bytes(g_cfg.uart_port, frame, frame_len);
    if (written < 0) {
        ++g_stats.frame_errors;
        return written;
    }

    ++g_stats.packets_sent;
    g_stats.bytes_sent += (uint32_t)written;
    return ESP_OK;
}

const ltst_v1_transport_stats_t *ltst_v1_transport_stats(void) {
    return &g_stats;
}
