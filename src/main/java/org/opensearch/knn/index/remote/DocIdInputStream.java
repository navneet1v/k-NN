/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

// TODO: This class needs a revamp to make more efficient
@Log4j2
public class DocIdInputStream extends InputStream {
    private final KNNVectorValues<?> knnVectorValues;
    private ByteBuffer currentBuffer;
    private int position = 0;

    public DocIdInputStream(KNNVectorValues<?> knnVectorValues) {
        this.knnVectorValues = knnVectorValues;
        try {
            loadNextVector();
        } catch (IOException e) {
            log.error("Failed to load initial vector", e);
        }
    }

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
        int docId = knnVectorValues.nextDoc();
        if (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            float[] vector = new float[1];
            vector[0] = (float) docId;
            currentBuffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            currentBuffer.asFloatBuffer().put(vector);
            position = 0;
        } else {
            currentBuffer = null;
        }
    }
}
