/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// ClusterANN batch quantized dot product — platform-independent with SIMD when available.
// Pure functions, no state, thread-safe.

#include <cstdint>
#include <cstring>
#include <algorithm>

// Scalar int4BitDotProduct: 4-bit query × 1-bit doc
static int64_t scalarInt4BitDotProduct(const uint8_t* q, const uint8_t* d, int32_t binaryCodeBytes) {
    int64_t result = 0;
    for (int32_t bitPlane = 0; bitPlane < 4; ++bitPlane) {
        int64_t subResult = 0;
        for (int32_t r = 0; r < binaryCodeBytes; ++r) {
            subResult += __builtin_popcount((q[bitPlane * binaryCodeBytes + r] & d[r]) & 0xFF);
        }
        result += subResult << bitPlane;
    }
    return result;
}

// 1-bit batch: each doc is bytesPerCode bytes (1 stripe)
void batchDotProduct1bit(const uint8_t* q, const uint8_t* d, float* r, int32_t bpc, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        r[i] = static_cast<float>(scalarInt4BitDotProduct(q, d + (int64_t)i * bpc, bpc));
    }
}

// 2-bit batch: each doc has 2 stripes of bpc/2 bytes
void batchDotProduct2bit(const uint8_t* q, const uint8_t* d, float* r, int32_t bpc, int32_t n) {
    int32_t ss = bpc / 2;
    for (int32_t i = 0; i < n; i++) {
        const uint8_t* doc = d + (int64_t)i * bpc;
        float s0 = static_cast<float>(scalarInt4BitDotProduct(q, doc, ss));
        float s1 = static_cast<float>(scalarInt4BitDotProduct(q, doc + ss, ss));
        r[i] = s0 + s1 * 2.0f;
    }
}

// 4-bit batch: each doc has 4 stripes of bpc/4 bytes
void batchDotProduct4bit(const uint8_t* q, const uint8_t* d, float* r, int32_t bpc, int32_t n) {
    int32_t ss = bpc / 4;
    for (int32_t i = 0; i < n; i++) {
        const uint8_t* doc = d + (int64_t)i * bpc;
        r[i] = 0;
        for (int32_t s = 0; s < 4; s++) {
            float stripe = static_cast<float>(scalarInt4BitDotProduct(q, doc + s * ss, ss));
            r[i] += stripe * static_cast<float>(1 << s);
        }
    }
}
