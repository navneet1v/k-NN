/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote.model;

import lombok.Builder;
import lombok.Value;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

@Value
@Builder
public class CreateIndexRequest implements ToXContentObject {
    String bucketName;
    String objectLocation;
    long numberOfVectors;
    int dimensions;
    String spaceType;

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        return builder.startObject()
            .field("bucket_name", bucketName)
            .field("object_location", objectLocation)
            .field("number_of_vectors", numberOfVectors)
            .field("dimensions", dimensions)
            .field("space_type", spaceType)
            .endObject();
    }
}
