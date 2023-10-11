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

package org.opensearch.knn.index;

import lombok.NoArgsConstructor;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.shard.IndexSettingProvider;

@NoArgsConstructor
public class KNNIndexSettingProvider implements IndexSettingProvider {

    private static final Settings EF_SEARCH_SETTING = Settings.builder()
        .put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, KNNSettings.INDEX_KNN_NEW_DEFAULT_ALGO_PARAM_EF_SEARCH)
        .build();

    /**
     * Returns explicitly set default index {@link Settings} for the given index. This should not
     * return null.
     *
     * @param indexName {@link String} name of the index
     * @param isDataStreamIndex boolean: index is a datastream index or not
     * @param templateAndRequestSettings {@link Settings}
     */
    @Override
    public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
        if (isKNNIndex(templateAndRequestSettings) && isEfSearchNotSetByUser(templateAndRequestSettings)) {
            return EF_SEARCH_SETTING;
        }
        return IndexSettingProvider.super.getAdditionalIndexSettings(indexName, isDataStreamIndex, templateAndRequestSettings);
    }

    private boolean isKNNIndex(Settings templateAndRequestSettings) {
        return templateAndRequestSettings.getAsBoolean(KNNSettings.KNN_INDEX, false);
    }

    private boolean isEfSearchNotSetByUser(Settings templateAndRequestSettings) {
        return templateAndRequestSettings.hasValue(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH) == false;
    }
}
