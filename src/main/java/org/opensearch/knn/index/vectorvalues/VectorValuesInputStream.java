/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@Log4j2
public class VectorValuesInputStream extends InputStream {
    private final KNNFloatVectorValues floatVectorValues;
    private ByteBuffer currentBuffer;
    private int position = 0;

    public VectorValuesInputStream(final KNNFloatVectorValues floatVectorValues) {
        this.floatVectorValues = floatVectorValues;
        try {
            loadNextVector();
        } catch (IOException e) {
            log.error("Failed to load initial vector", e);
        }
    }

    /**
     * Reads the next byte of data from the input stream. The value byte is
     * returned as an {@code int} in the range {@code 0} to
     * {@code 255}. If no byte is available because the end of the stream
     * has been reached, the value {@code -1} is returned. This method
     * blocks until input data is available, the end of the stream is detected,
     * or an exception is thrown.
     *
     * @return     the next byte of data, or {@code -1} if the end of the
     *             stream is reached.
     * @throws     IOException  if an I/O error occurs.
     */
    @Override
    public int read() throws IOException {
        if (currentBuffer == null || position >= currentBuffer.capacity()) {
            loadNextVector();
            if (currentBuffer == null) {
                return -1;
            }
        }

        return currentBuffer.get(position++) & 0xFF;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        if (currentBuffer == null) {
            return -1;
        }

        int available = currentBuffer.capacity() - position;
        if (available <= 0) {
            loadNextVector();
            if (currentBuffer == null) {
                return -1;
            }
            available = currentBuffer.capacity() - position;
        }

        int bytesToRead = Math.min(available, len);
        currentBuffer.position(position);
        currentBuffer.get(b, off, bytesToRead);
        position += bytesToRead;

        return bytesToRead;
    }

    private void loadNextVector() throws IOException {
        int docId = floatVectorValues.nextDoc();
        if (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            float[] vector = floatVectorValues.getVector();
            // this is a costly operation. We should optimize this
            currentBuffer = ByteBuffer.allocate(vector.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            currentBuffer.asFloatBuffer().put(vector);
            position = 0;
        } else {
            currentBuffer = null;
        }
    }
}
