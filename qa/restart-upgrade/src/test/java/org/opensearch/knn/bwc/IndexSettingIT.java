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

package org.opensearch.knn.bwc;

import org.junit.Assert;
import org.opensearch.knn.index.KNNSettings;

public class IndexSettingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;

    public void testOldIndexSettingsPersistedAfterUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            // create index with Old Values
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            int old_ef_search = Integer.parseInt(getIndexSettingByName(testIndex, KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, true));
            Assert.assertEquals(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH.intValue(), old_ef_search);
        } else {
            int old_ef_search = Integer.parseInt(getIndexSettingByName(testIndex, KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, true));
            Assert.assertEquals(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH.intValue(), old_ef_search);
            deleteKNNIndex(testIndex);
        }
    }

    // private void assertEfSearchOldDefaultValue(String indexName) {
    // if (Version.fromString(getBWCVersion().get()).onOrAfter(Version.V_2_10_0)) {
    //
    // }
    //
    // }
}
