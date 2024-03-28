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

package org.opensearch.knn;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.opensearch.knn.jni.JNIService;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, timeUnit = TimeUnit.SECONDS, time = 300)
@Measurement(iterations = 1, timeUnit = TimeUnit.SECONDS, time = 300)
@Fork(3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)

public class TransferVectorsBenchmarks {
    private static final Random random = new Random(1212121212);
    private static final int TOTAL_NUMBER_OF_VECTOR_TO_BE_TRANSFERRED = 1000000;

    @Param({ "128", "256", "384", "512" })
    private int dimension;

    @Param({ "100000", "500000", "1000000" })
    private int vectorsPerTransfer;

    private List<float[]> vectorList;

    @Setup(Level.Invocation)
    public void setup() {
        vectorList = new ArrayList<>();
        for (int i = 0; i < TOTAL_NUMBER_OF_VECTOR_TO_BE_TRANSFERRED; i++) {
            vectorList.add(generateRandomVector(dimension));
        }
    }

    @Benchmark
    public void transferVectors() {
        long vectorsAddress = 0;
        List<float[]> vectorToTransfer = new ArrayList<>();
        long startTime = System.currentTimeMillis();
        for (float[] floats : vectorList) {
            if (vectorToTransfer.size() == vectorsPerTransfer) {
                vectorsAddress = JNIService.transferVectorsV2(vectorsAddress, vectorToTransfer.toArray(new float[][]{}));
                vectorToTransfer = new ArrayList<>();
            }
            vectorToTransfer.add(floats);
        }
        if(!vectorToTransfer.isEmpty()) {
            vectorsAddress = JNIService.transferVectorsV2(vectorsAddress, vectorToTransfer.toArray(new float[][]{}));
        }

        JNIService.freeVectors(vectorsAddress);
        long endTime = System.currentTimeMillis();
        System.out.println("Time taken to transfer vectors " + (endTime - startTime));
    }

    @Benchmark
    public void transferVectorsInitialCapacity() {
        long vectorsAddress = 0;
        long startingIndex = 0;
        long startTime = System.currentTimeMillis();
        List<float[]> vectorToTransfer = new ArrayList<>();
        for (float[] floats : vectorList) {
            if (vectorToTransfer.size() == vectorsPerTransfer) {
                vectorsAddress = JNIService.transferVectorsV3(vectorsAddress,
                        vectorToTransfer.toArray(new float[][]{}), startingIndex,
                        TOTAL_NUMBER_OF_VECTOR_TO_BE_TRANSFERRED * dimension);
                vectorToTransfer = new ArrayList<>();
                startingIndex += vectorsPerTransfer;
            }
            vectorToTransfer.add(floats);
        }
        if(!vectorToTransfer.isEmpty()) {
            vectorsAddress = JNIService.transferVectorsV3(vectorsAddress, vectorToTransfer.toArray(new float[][]{}),
                    startingIndex, TOTAL_NUMBER_OF_VECTOR_TO_BE_TRANSFERRED * dimension);
            startingIndex += vectorToTransfer.size();
        }
        JNIService.freeVectors(vectorsAddress);
        long endTime = System.currentTimeMillis();
        System.out.println("Time taken to transferVectorsInitialCapacity " + (endTime - startTime));
    }

    private float[] generateRandomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = -500 + (float) random.nextGaussian() * (1000);
        }
        return vector;
    }
}
