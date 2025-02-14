/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Note: This class does not support retrying due to the sequential nature of {@link KNNVectorValues}
 */
@Log4j2
public class VectorValuesInputStream extends InputStream {

    private final KNNVectorValues<?> knnVectorValues;
    // It is extremely difficult to avoid using a buffer in this class as we need to be able to convert from float[] to byte[]. this buffer
    // will be filled 1 vector at a time.
    private ByteBuffer currentBuffer;
    private final int bytesPerVector;
    private final long bytesRemaining;
    private final VectorDataType vectorDataType;

    /**
     * Used to represent a part of a {@link KNNVectorValues} as an {@link InputStream}. Expected to be used with
     * {@link org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer#asyncBlobUpload}
     *
     * @param knnVectorValues
     * @param vectorDataType
     * @param startPosition
     * @param size
     * @throws IOException
     */
    public VectorValuesInputStream(KNNVectorValues<?> knnVectorValues, VectorDataType vectorDataType, long startPosition, long size)
        throws IOException {
        this.bytesRemaining = size;
        this.knnVectorValues = knnVectorValues;
        this.vectorDataType = vectorDataType;
        initializeVectorValues(this.knnVectorValues);
        this.bytesPerVector = this.knnVectorValues.bytesPerVector();

        // TODO: For S3 the retryable input stream is backed by a buffer of part size, so all readLimit bytes are loaded onto heap at once
        // (16mb chunks by default). This means if we use a buffer here it is basically duplicate memory usage.
        this.currentBuffer = ByteBuffer.allocate(bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        setPosition(startPosition);
    }

    /**
     * Used to represent the entire {@link KNNVectorValues} as a single {@link InputStream}. Expected to be used with
     * {@link org.opensearch.common.blobstore.BlobContainer#writeBlob}
     *
     * @param knnVectorValues
     * @param vectorDataType
     * @throws IOException
     */
    public VectorValuesInputStream(KNNVectorValues<?> knnVectorValues, VectorDataType vectorDataType) throws IOException {
        this(knnVectorValues, vectorDataType, 0, Long.MAX_VALUE);
    }

    @Override
    public int read() throws IOException {
        if (bytesRemaining <= 0 || currentBuffer == null) {
            return -1;
        }

        if (!currentBuffer.hasRemaining()) {
            reloadBuffer();
            if (currentBuffer == null) {
                return -1;
            }
        }

        return currentBuffer.get() & 0xFF;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        if (bytesRemaining <= 0 || currentBuffer == null) {
            return -1;
        }

        int available = currentBuffer.remaining();
        if (available <= 0) {
            reloadBuffer();
            if (currentBuffer == null) {
                return -1;
            }
            available = currentBuffer.remaining();
        }

        int bytesToRead = Math.min(available, len);
        currentBuffer.get(b, off, bytesToRead);
        return bytesToRead;
    }

    /**
     * This class does not support skipping. Instead, use {@link VectorValuesInputStream#setPosition}
     *
     * @param n   the number of bytes to be skipped.
     * @return
     * @throws IOException
     */
    @Override
    public long skip(long n) throws IOException {
        throw new UnsupportedOperationException("VectorValuesInputStream does not support skip()");
    }

    /**
     * Advances n bytes forward in the knnVectorValues.
     * Note: {@link KNNVectorValues#advance} is not supported when we are merging segments, so we do not use it here.
     * Note: {@link KNNVectorValues#nextDoc} is relatively efficient, but {@link KNNVectorValues#getVector} may perform a disk read, so we avoid using {@link VectorValuesInputStream#reloadBuffer()} here.
     *
     * @param n
     * @return
     * @throws IOException
     */
    private void setPosition(long n) throws IOException {
        if (currentBuffer.position() != 0) {
            throw new UnsupportedOperationException("setPosition is only supported from the start of a vector");
        }

        long bytesSkipped = 0;
        int vectorsToSkip = (int) (n / bytesPerVector);
        log.debug("Skipping {} bytes, {} vectors", n, vectorsToSkip);
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS && vectorsToSkip > 0) {
            docId = knnVectorValues.nextDoc();
            bytesSkipped += bytesPerVector;
            vectorsToSkip--;
        }

        // After skipping the correct number of vectors, fill the buffer with the current vector
        float[] vector = ((KNNFloatVectorValues) knnVectorValues).getVector();
        currentBuffer.asFloatBuffer().put(vector);

        // Advance to the correct position within the current vector
        long remainingBytes = n - bytesSkipped;
        if (remainingBytes > 0) {
            currentBuffer.position((int) remainingBytes);
        }
    }

    /**
     * Advances to the next doc, and then refills the buffer with the new doc.
     * @throws IOException
     */
    private void reloadBuffer() throws IOException {
        currentBuffer.clear();

        int docId = knnVectorValues.nextDoc();
        if (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            if (vectorDataType != FLOAT) {
                throw new UnsupportedOperationException("Unsupported vector data type: " + vectorDataType);
            } else {
                float[] vector = ((KNNFloatVectorValues) knnVectorValues).getVector();
                currentBuffer.asFloatBuffer().put(vector);
            }
        } else {
            // Reset buffer to null to indicate that there are no more docs to be read
            currentBuffer = null;
        }
    }
}
