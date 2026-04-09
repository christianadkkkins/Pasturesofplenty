#ifndef LTST_V1_TRANSPORT_H
#define LTST_V1_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#include "driver/uart.h"

#include "ltst_v1_feature_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LTST_V1_FRAME_SYNC0 0xA5u
#define LTST_V1_FRAME_SYNC1 0x5Au
#define LTST_V1_FRAME_VERSION 0x01u
#define LTST_V1_MSG_TYPE_FEATURE_PACKET 0x01u
#define LTST_V1_FRAME_HEADER_SIZE 6u
#define LTST_V1_FRAME_TRAILER_SIZE 2u
#define LTST_V1_FRAME_SIZE_BYTES (LTST_V1_FRAME_HEADER_SIZE + LTST_V1_PACKET_SIZE_BYTES + LTST_V1_FRAME_TRAILER_SIZE)

typedef struct ltst_v1_transport_config_t {
    uart_port_t uart_port;
    int tx_pin;
    int rx_pin;
    int baud_rate;
    int tx_buffer_size;
    int rx_buffer_size;
} ltst_v1_transport_config_t;

typedef struct ltst_v1_transport_stats_t {
    uint32_t packets_sent;
    uint32_t bytes_sent;
    uint32_t frame_errors;
    uint32_t crc_errors;
} ltst_v1_transport_stats_t;

typedef struct ltst_v1_frame_info_t {
    uint8_t version;
    uint8_t msg_type;
    uint16_t payload_len;
    uint16_t crc16;
} ltst_v1_frame_info_t;

int ltst_v1_transport_init(const ltst_v1_transport_config_t *cfg);
uint16_t ltst_v1_crc16_ccitt_false(const uint8_t *data, size_t len);
size_t ltst_v1_build_frame(uint8_t *out_frame, size_t out_capacity, const ltst_v1_feature_packet_t *packet, ltst_v1_frame_info_t *info);
int ltst_v1_send_packet(const ltst_v1_feature_packet_t *packet, ltst_v1_frame_info_t *info);
const ltst_v1_transport_stats_t *ltst_v1_transport_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_TRANSPORT_H */
