/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote.model;

import lombok.Builder;
import lombok.Value;

@Value
@Builder
public class GetJobRequest {
    String jobId;
}
