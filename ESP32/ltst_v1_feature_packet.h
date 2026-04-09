#ifndef LTST_V1_FEATURE_PACKET_H
#define LTST_V1_FEATURE_PACKET_H

#include <stdbool.h>
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LTST V1 cardiac feature packet
 *
 * Frozen fields:
 *   - beat_index
 *   - poincare_b  (UQ2.14)
 *   - drift_norm  (UQ4.12)
 *   - energy_asym (SQ1.14)
 *   - quality_flags
 *
 * Packet size: 12 bytes
 * Cadence: emit once per accepted beat after warmup
 */

#define LTST_V1_PACKET_VERSION         0x0001u
#define LTST_V1_PACKET_SIZE_BYTES      12u

#define LTST_V1_POINCARE_FRAC_BITS     14u
#define LTST_V1_DRIFT_FRAC_BITS        12u
#define LTST_V1_ENERGY_ASYM_FRAC_BITS  14u

#define LTST_V1_POINCARE_MAX_FLOAT     3.99993896484375f
#define LTST_V1_DRIFT_MAX_FLOAT        15.999755859375f
#define LTST_V1_ENERGY_ASYM_MIN_FLOAT -2.0f
#define LTST_V1_ENERGY_ASYM_MAX_FLOAT  1.99993896484375f

enum ltst_v1_quality_flag_bits {
    LTST_V1_FEATURE_VALID  = (1u << 0),
    LTST_V1_WARMUP         = (1u << 1),
    LTST_V1_LOW_CURR_ENERGY = (1u << 2),
    LTST_V1_LOW_PAST_ENERGY = (1u << 3),
    LTST_V1_SATURATED      = (1u << 4),
    LTST_V1_DIV_GUARD      = (1u << 5),
    LTST_V1_REJECTED_BEAT  = (1u << 6),
};

#if defined(__GNUC__) || defined(__clang__)
#define LTST_V1_PACKED __attribute__((packed))
#else
#define LTST_V1_PACKED
#endif

typedef struct LTST_V1_PACKED ltst_v1_feature_packet_t {
    uint32_t beat_index;
    uint16_t poincare_b_uq2_14;
    uint16_t drift_norm_uq4_12;
    int16_t energy_asym_sq1_14;
    uint16_t quality_flags;
} ltst_v1_feature_packet_t;

typedef struct ltst_v1_feature_float_view_t {
    float poincare_b;
    float drift_norm;
    float energy_asym;
} ltst_v1_feature_float_view_t;

static inline uint16_t ltst_v1_clip_uq2_14(float value, bool *saturated) {
    const float scaled = value * (float)(1u << LTST_V1_POINCARE_FRAC_BITS);
    long raw = lroundf(scaled);
    if (raw < 0L) {
        raw = 0L;
        if (saturated) { *saturated = true; }
    } else if (raw > 65535L) {
        raw = 65535L;
        if (saturated) { *saturated = true; }
    }
    return (uint16_t)raw;
}

static inline uint16_t ltst_v1_clip_uq4_12(float value, bool *saturated) {
    const float scaled = value * (float)(1u << LTST_V1_DRIFT_FRAC_BITS);
    long raw = lroundf(scaled);
    if (raw < 0L) {
        raw = 0L;
        if (saturated) { *saturated = true; }
    } else if (raw > 65535L) {
        raw = 65535L;
        if (saturated) { *saturated = true; }
    }
    return (uint16_t)raw;
}

static inline int16_t ltst_v1_clip_sq1_14(float value, bool *saturated) {
    const float scaled = value * (float)(1u << LTST_V1_ENERGY_ASYM_FRAC_BITS);
    long raw = lroundf(scaled);
    if (raw < -32768L) {
        raw = -32768L;
        if (saturated) { *saturated = true; }
    } else if (raw > 32767L) {
        raw = 32767L;
        if (saturated) { *saturated = true; }
    }
    return (int16_t)raw;
}

static inline float ltst_v1_decode_poincare_b(uint16_t raw) {
    return ((float)raw) / (float)(1u << LTST_V1_POINCARE_FRAC_BITS);
}

static inline float ltst_v1_decode_drift_norm(uint16_t raw) {
    return ((float)raw) / (float)(1u << LTST_V1_DRIFT_FRAC_BITS);
}

static inline float ltst_v1_decode_energy_asym(int16_t raw) {
    return ((float)raw) / (float)(1u << LTST_V1_ENERGY_ASYM_FRAC_BITS);
}

static inline void ltst_v1_init_packet(
    ltst_v1_feature_packet_t *packet,
    uint32_t beat_index,
    float poincare_b,
    float drift_norm,
    float energy_asym,
    uint16_t quality_flags
) {
    bool saturated = false;
    packet->beat_index = beat_index;
    packet->poincare_b_uq2_14 = ltst_v1_clip_uq2_14(poincare_b, &saturated);
    packet->drift_norm_uq4_12 = ltst_v1_clip_uq4_12(drift_norm, &saturated);
    packet->energy_asym_sq1_14 = ltst_v1_clip_sq1_14(energy_asym, &saturated);
    packet->quality_flags = quality_flags | (saturated ? LTST_V1_SATURATED : 0u);
}

static inline ltst_v1_feature_float_view_t ltst_v1_decode_packet(const ltst_v1_feature_packet_t *packet) {
    ltst_v1_feature_float_view_t out;
    out.poincare_b = ltst_v1_decode_poincare_b(packet->poincare_b_uq2_14);
    out.drift_norm = ltst_v1_decode_drift_norm(packet->drift_norm_uq4_12);
    out.energy_asym = ltst_v1_decode_energy_asym(packet->energy_asym_sq1_14);
    return out;
}

static inline bool ltst_v1_packet_is_valid(const ltst_v1_feature_packet_t *packet) {
    return (packet->quality_flags & LTST_V1_FEATURE_VALID) != 0u;
}

static inline bool ltst_v1_packet_is_saturated(const ltst_v1_feature_packet_t *packet) {
    return (packet->quality_flags & LTST_V1_SATURATED) != 0u;
}

static inline bool ltst_v1_packet_div_guarded(const ltst_v1_feature_packet_t *packet) {
    return (packet->quality_flags & LTST_V1_DIV_GUARD) != 0u;
}

#ifdef __cplusplus
}
#endif

#endif /* LTST_V1_FEATURE_PACKET_H */
