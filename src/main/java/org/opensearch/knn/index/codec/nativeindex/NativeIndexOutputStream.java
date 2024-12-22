/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.Builder;
import lombok.Value;
import org.apache.commons.lang.NotImplementedException;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.io.OutputStream;

@Builder
@Value
public class NativeIndexOutputStream extends OutputStream {

    IndexOutput indexOutput;

    /**
     * Writes the specified byte to this output stream. The general
     * contract for {@code write} is that one byte is written
     * to the output stream. The byte to be written is the eight
     * low-order bits of the argument {@code b}. The 24
     * high-order bits of {@code b} are ignored.
     *
     * @param b the {@code byte}.
     * @throws IOException if an I/O error occurs. In particular,
     *                     an {@code IOException} may be thrown if the
     *                     output stream has been closed.
     */
    @Override
    public void write(int b) throws IOException {
        throw new NotImplementedException("This function is not implemented");
    }

    /**
     * Writes {@code b.length} bytes from the specified byte array
     * to this output stream. The general contract for {@code write(b)}
     * is that it should have exactly the same effect as the call
     * {@code write(b, 0, b.length)}.
     *
     * @param b the data.
     * @throws IOException if an I/O error occurs.
     * @see OutputStream#write(byte[], int, int)
     */
    @Override
    public void write(byte[] b) throws IOException {
        indexOutput.writeBytes(b, b.length);
    }
}
