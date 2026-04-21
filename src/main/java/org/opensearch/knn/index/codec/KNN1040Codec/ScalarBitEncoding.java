/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

/**
 * Encapsulates quantization bit-width configuration including packing, query packing,
 * and packed size computation for each supported bit-width.
 *
 * <p>Supported encodings:
 * <ul>
 *   <li>{@link #ONE_BIT} — 32x compression, 4-bit query (asymmetric)</li>
 *   <li>{@link #TWO_BIT} — 16x compression, 4-bit query (asymmetric)</li>
 *   <li>{@link #FOUR_BIT} — 8x compression, 4-bit query (symmetric)</li>
 * </ul>
 */
public enum ScalarBitEncoding {

    ONE_BIT((byte) 1, (byte) 4) {
        @Override
        public void packDoc(byte[] raw, byte[] packed, int dimension) {
            // MSB-first binary packing: 8 values per byte
            for (int i = 0; i < dimension;) {
                byte result = 0;
                for (int j = 7; j >= 0 && i < dimension; j--) {
                    result |= (byte) ((raw[i] & 1) << j);
                    ++i;
                }
                packed[((i + 7) / 8) - 1] = result;
            }
        }

        @Override
        public int docPackedBytes(int dimension) {
            return (dimension + 7) / 8;
        }
    },

    TWO_BIT((byte) 2, (byte) 4) {
        @Override
        public void packDoc(byte[] raw, byte[] packed, int dimension) {
            // Transpose into 2 stripes (lower bits, upper bits) MSB-first
            int stripeSize = packed.length / 2;
            int i = 0, index = 0;
            int limit = dimension - 7;
            for (; i < limit; i += 8, index++) {
                int lower = 0, upper = 0;
                for (int j = 7; j >= 0; j--) {
                    lower |= (raw[i + (7 - j)] & 1) << j;
                    upper |= ((raw[i + (7 - j)] >> 1) & 1) << j;
                }
                packed[index] = (byte) lower;
                packed[index + stripeSize] = (byte) upper;
            }
            if (i < dimension) {
                int lower = 0, upper = 0;
                for (int j = 7; i < dimension; j--, i++) {
                    lower |= (raw[i] & 1) << j;
                    upper |= ((raw[i] >> 1) & 1) << j;
                }
                packed[index] = (byte) lower;
                packed[index + stripeSize] = (byte) upper;
            }
        }

        @Override
        public int docPackedBytes(int dimension) {
            return ((dimension + 7) / 8) * 2;
        }
    },

    FOUR_BIT((byte) 4, (byte) 4) {
        @Override
        public void packDoc(byte[] raw, byte[] packed, int dimension) {
            // Transpose into 4 nibble stripes for SIMD-friendly dot product
            int stripeSize = (dimension + 7) / 8;
            for (int i = 0; i < dimension; i++) {
                int val = raw[i] & 0x0F;
                int byteIdx = i / 8;
                int bitIdx = 7 - (i % 8);
                if ((val & 1) != 0) packed[byteIdx] |= (byte) (1 << bitIdx);
                if ((val & 2) != 0) packed[stripeSize + byteIdx] |= (byte) (1 << bitIdx);
                if ((val & 4) != 0) packed[2 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
                if ((val & 8) != 0) packed[3 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
            }
        }

        @Override
        public int docPackedBytes(int dimension) {
            return ((dimension + 7) / 8) * 4;
        }
    };

    private final byte docBits;
    private final byte queryBits;

    ScalarBitEncoding(byte docBits, byte queryBits) {
        this.docBits = docBits;
        this.queryBits = queryBits;
    }

    /** Pack raw quantized values into the encoding-specific format. */
    public abstract void packDoc(byte[] raw, byte[] packed, int dimension);

    /** Number of bytes needed to store one packed document vector. */
    public abstract int docPackedBytes(int dimension);

    /** Pack query (always 4-bit → 4 nibble stripes). */
    public void packQuery(byte[] raw, byte[] packed, int dimension) {
        FOUR_BIT.packDoc(raw, packed, dimension);
    }

    /** Number of bytes for a packed query vector. */
    public int queryPackedBytes(int dimension) {
        return ((dimension + 7) / 8) * 4; // always 4 stripes
    }

    /** Document quantization bits. */
    public byte docBits() {
        return docBits;
    }

    /** Query quantization bits. */
    public byte queryBits() {
        return queryBits;
    }

    /** Per-vector record size in .claq: packed codes + 4 correction ints (16 bytes). */
    public int recordBytes(int dimension) {
        return docPackedBytes(dimension) + 16;
    }

    /** Scale factor for document bit-width: 1 / (2^bits - 1). */
    public float docBitScale() {
        return 1.0f / ((1 << docBits) - 1);
    }

    /** Lookup by doc bits value. */
    public static ScalarBitEncoding fromDocBits(int bits) {
        return switch (bits) {
            case 1 -> ONE_BIT;
            case 2 -> TWO_BIT;
            case 4 -> FOUR_BIT;
            default -> throw new IllegalArgumentException("Unsupported doc bits: " + bits + " (must be 1, 2, or 4)");
        };
    }

    /** Ordinal ID for serialization. */
    public int id() {
        return ordinal();
    }

    /** Lookup by serialized ID. */
    public static ScalarBitEncoding fromId(int id) {
        ScalarBitEncoding[] values = values();
        if (id < 0 || id >= values.length) {
            throw new IllegalArgumentException("Unknown ScalarBitEncoding id: " + id);
        }
        return values[id];
    }
}
