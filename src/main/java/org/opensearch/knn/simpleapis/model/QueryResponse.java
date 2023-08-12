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

import lombok.Builder;
import lombok.Value;

import java.util.List;

@Value
@Builder
public class QueryResponse {
    List<SimpleQueryResults> simpleQueryResults;
}
