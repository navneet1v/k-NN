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
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentFragment;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@ToString
public class SimpleQueryResults implements Writeable {

    private List<SimpleQueryResult> queryResultList;

    public static SimpleQueryResults INSTANCE = new SimpleQueryResults();

    private SimpleQueryResults() {}

    public SimpleQueryResults(StreamInput in) throws IOException {
        queryResultList = in.readList(SimpleQueryResult::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeList(queryResultList);
    }

    public static SimpleQueryResults readEmptyResultFrom(StreamInput in) {
        return INSTANCE;
    }

    @Data
    @AllArgsConstructor
    public static class SimpleQueryResult implements Writeable, ToXContentFragment {
        private String id;
        private float score;

        public SimpleQueryResult(StreamInput in) throws IOException {
            id = in.readString();
            score = in.readFloat();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeString(id);
            out.writeFloat(score);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject();
            builder.field("_id", id);
            builder.field("_score", score);
            builder.endObject();
            return builder;
        }

        @Override
        public boolean isFragment() {
            return true;
        }
    }
}
