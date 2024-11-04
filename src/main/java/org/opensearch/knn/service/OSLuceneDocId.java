/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.service;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import lombok.Value;

@Value
@Builder
@EqualsAndHashCode
@ToString
public class OSLuceneDocId {
    @Builder.Default
    String opensearchIndexName = "my-index";
    byte[] segmentId;
    int segmentDocId;
    @Builder.Default
    float score = 0;

    public OSLuceneDocId cloneWithScore(float score) {
        return OSLuceneDocId.builder()
            .score(score)
            .segmentDocId(segmentDocId)
            .segmentId(segmentId)
            .opensearchIndexName(opensearchIndexName)
            .build();
    }
}
