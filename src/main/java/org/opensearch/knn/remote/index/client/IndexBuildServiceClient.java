/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.client;

import org.apache.http.HttpEntity;
import org.apache.http.HttpHost;
import org.apache.http.HttpRequest;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.remote.index.model.CreateIndexRequest;
import org.opensearch.knn.remote.index.model.CreateIndexResponse;
import org.opensearch.knn.remote.index.s3.S3Client;

import java.io.IOException;

/**
 * Main class to class the IndexBuildServiceAPIs
 */
public class IndexBuildServiceClient {
    private static volatile IndexBuildServiceClient INSTANCE;
    private static final String CONTENT_TYPE = "Content-Type";
    private static final String APPLICATION_JSON = "application/json";
    private static final String ACCEPT = "Accept";
    private final HttpClient httpClient;
    private final HttpHost httpHost;

    public static IndexBuildServiceClient getInstance() throws IOException {
        IndexBuildServiceClient result = INSTANCE;
        if (result == null) {
            synchronized (S3Client.class) {
                result = INSTANCE;
                if (result == null) {
                    INSTANCE = result = new IndexBuildServiceClient();
                }
            }
        }
        return result;
    }

    private IndexBuildServiceClient() {
        this.httpClient = HttpClientBuilder.create().build();
        this.httpHost = new HttpHost(KNNSettings.getRemoteServiceEndpoint(), KNNSettings.getRemoteServicePort(), "http");
    }

    /**
     * API to be called to create the Vector Index using remote endpoint
     * @param createIndexRequest {@link CreateIndexRequest}
     * @throws IOException Exception called if createIndex request is not successful
     */
    public CreateIndexResponse createIndex(final CreateIndexRequest createIndexRequest) throws IOException {
        HttpPost request = new HttpPost();
        request.setHeader(CONTENT_TYPE, APPLICATION_JSON);
        request.setHeader(ACCEPT, APPLICATION_JSON);
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder = createIndexRequest.toXContent(builder, null);
        request.setEntity(new StringEntity(builder.toString()));

        HttpResponse response = makeHTTPRequest(request);
        HttpEntity httpEntity = response.getEntity();
        String responseString = EntityUtils.toString(httpEntity);
        return parseCreateIndexResponse(responseString);
    }

    // TODO: To be implemented
    public void checkIndexBuildStatus() {

    }

    private HttpResponse makeHTTPRequest(final HttpRequest request) throws IOException {
        HttpResponse response = httpClient.execute(httpHost, request);
        HttpEntity entity = response.getEntity();
        int statusCode = response.getStatusLine().getStatusCode();

        if (statusCode >= 400) {
            String errorBody = entity != null ? EntityUtils.toString(entity) : "No response body";
            throw new IOException("Request failed with status code: " + statusCode + ", body: " + errorBody);
        }

        return response;
    }

    // Keeping it package private for doing the unit testing for now.
    static CreateIndexResponse parseCreateIndexResponse(final String responseString) throws IOException {
        final XContent xContent = MediaTypeRegistry.getDefaultMediaType().xContent();
        final XContentParser parser = xContent.createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            responseString
        );
        return CreateIndexResponse.fromXContent(parser);
    }

}
