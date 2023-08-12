/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.simpleapis.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import org.opensearch.common.io.stream.NamedWriteable;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

@Builder
@AllArgsConstructor
@Getter
@ToString
public class QueryRequest implements NamedWriteable {
    String indexName;
    float[] vector;
    int k;
    @Builder.Default
    int size = 10;
    String vectorFieldName;

    /**
     * Write this into the {@linkplain StreamOutput}.
     *
     * @param out
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(indexName);
        out.writeFloatArray(vector);
        out.writeVInt(k);
        out.writeVInt(size);
        out.writeString(vectorFieldName);
    }

    public QueryRequest(StreamInput in) throws IOException {
        indexName = in.readString();
        vector = in.readFloatArray();
        k = in.readVInt();
        size = in.readVInt();
        vectorFieldName = in.readString();
    }

    /**
     * Returns the name of the writeable object
     */
    @Override
    public String getWriteableName() {
        return "QueryRequest";
    }
}
