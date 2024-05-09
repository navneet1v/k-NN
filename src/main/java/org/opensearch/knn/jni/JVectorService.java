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

package org.opensearch.knn.jni;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.Feature;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.graph.disk.InlineVectorValues;
import io.github.jbellis.jvector.graph.disk.InlineVectors;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.extern.log4j.Log4j2;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Log4j2
public class JVectorService {
    private static VectorTypeSupport vts;
    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            return null;
        });
    }


    public static void createIndex(int originalDimension, List<float[]> vectors, String path) {
        final List<VectorFloat<?>> baseVectors = new ArrayList<>();

        for(int i = 0 ; i < vectors.size(); i ++) {
            VectorFloat<?> vectorFloat = vts.createFloatVector(originalDimension);
            for(int j = 0 ; j < vectors.get(i).length; i ++) {
                vectorFloat.set(j, vectors.get(i)[j]);
            }
            baseVectors.add(vectorFloat);
        }

        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
        // compute the codebook, but don't encode any vectors yet
        ProductQuantization pq = ProductQuantization.compute(ravv, 16, 256, true);

        Path indexPath = null;
        Path pqPath = null;
        try {
            indexPath = Files.createFile(Path.of(path));
            pqPath = Files.createFile(Path.of(path +".pq"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // Builder creation looks mostly the same, but we need to set the BuildScoreProvider after the PQVectors are created
        try (GraphIndexBuilder builder = new GraphIndexBuilder(null,
                ravv.dimension(), 16, 100, 1.2f, 1.2f);
             // explicit Writer for the first time, this is what's behind OnDiskGraphIndex.write
             OnDiskGraphIndexWriter writer = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), indexPath)
                     .with(new InlineVectors(ravv.dimension()))
                     .withMapper(new OnDiskGraphIndexWriter.IdentityMapper())
                     .build();
             // you can use the partially written index as a source of the vectors written so far
             InlineVectorValues ivv = new InlineVectorValues(ravv.dimension(), writer);
             // output for the compressed vectors
             DataOutputStream pqOut = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(pqPath))))
        {
            List<ByteSequence<?>> incrementallyCompressedVectors = new ArrayList<>();
            PQVectors pqv = new PQVectors(pq, incrementallyCompressedVectors);

            // now we can create the actual BuildScoreProvider based on PQ + reranking
            BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, ivv, pqv);
            builder.setBuildScoreProvider(bsp);

            for (VectorFloat<?> v : baseVectors) {
                // compress the new vector and add it to the PQVectors (via incrementallyCompressedVectors)
                int ordinal = incrementallyCompressedVectors.size();
                incrementallyCompressedVectors.add(pq.encode(v));
                // write the full vector to disk
                writer.writeInline(ordinal, Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(v)));
                // now add it to the graph -- the previous steps must be completed first since the PQVectors
                // and InlineVectorValues are both used during the search that runs as part of addGraphNode construction
                builder.addGraphNode(ordinal, v);
            }

            // cleanup does a final enforcement of maxDegree and handles other scenarios like deleted nodes
            // that we don't need to worry about here
            builder.cleanup();

            // finish writing the index (by filling in the edge lists) and write our completed PQVectors
            writer.write(Map.of());
            pqv.write(pqOut);

        } catch (Exception e) {
            log.error("Error while creating index with JVector ", e);
        }
    }
}
