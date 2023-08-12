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

package org.opensearch.knn.simpleapis.transport;

import org.opensearch.action.ActionType;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.knn.simpleapis.model.QueryActionResponse;

public class SimpleQueryAction extends ActionType<QueryActionResponse> {
    public static final String NAME = "cluster:admin/simple_query_transport_action";
    public static final SimpleQueryAction INSTANCE = new SimpleQueryAction(NAME, QueryActionResponse::new);

    /**
     * @param name                The name of the action, must be unique across actions.
     * @param queryResponseReader A reader for the response type
     */
    public SimpleQueryAction(String name, Writeable.Reader<QueryActionResponse> queryResponseReader) {
        super(name, queryResponseReader);
    }
}
