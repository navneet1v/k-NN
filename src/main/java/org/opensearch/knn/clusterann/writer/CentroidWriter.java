package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.writer.rotation.RotationWriter;

import java.io.IOException;

/**
 * Writes a field's centroids to {@code .clac}, the twin of {@code CentroidVectorValues}.
 *
 * <p>Each record is {@code dimension} floats followed by {@code ‖c‖²}, so records are fixed width and a centroid is
 * one seek away rather than a scan. The trailing norm is stored rather than recomputed because every score needs it:
 * derived per query it would cost a pass over the centroid, per posting one pass per cluster visited.
 *
 * <p>A rotated field carries the centroids <b>twice</b>, in both spaces, and both are needed:
 *
 * <ul>
 *   <li><b>raw</b> — what a planner ranks an unrotated query against
 *   <li><b>rotated</b> — the space the stored codes live in, so the space every residual and every ADC correction is
 *       measured in
 * </ul>
 *
 * <p>Offsets returned are relative to the start of this field's {@code .clac} region, because that is what the reader
 * slices before using them.
 */
public final class CentroidWriter {

    private CentroidWriter() {}

    /**
     * Where a field's centroid regions ended up, relative to the field's {@code .clac} region.
     *
     * @param rawOffset the centroids as clustering produced them
     * @param rotatedOffset the same centroids rotated, or {@link ClusterANNFieldMeta#NO_ROTATION} for an unrotated
     *     field, whose raw centroids already are the only ones there are
     * @param length total bytes the field contributed to {@code .clac}
     */
    public record Offsets(long rawOffset, long rotatedOffset, long length) {}

    /**
     * Writes the raw centroids and, when the field is rotated, the rotated ones after them.
     *
     * @param regionStart the absolute offset in {@code .clac} where this field's region begins; the returned offsets
     *     are relative to it
     */
    public static Offsets write(IndexOutput clac, long regionStart, float[][] centroids, RotationWriter rotation)
        throws IOException {

        long rawOffset = clac.getFilePointer() - regionStart;
        for (float[] centroid : centroids) {
            writeCentroid(clac, centroid);
        }

        long rotatedOffset = ClusterANNFieldMeta.NO_ROTATION;
        if (rotation.rotationId() != ClusterANNFieldMeta.ROTATION_NONE) {
            rotatedOffset = clac.getFilePointer() - regionStart;
            float[] rotated = new float[rotation.dimension()];
            for (float[] centroid : centroids) {
                rotation.rotate(centroid, rotated);
                writeCentroid(clac, rotated);
            }
        }

        return new Offsets(rawOffset, rotatedOffset, clac.getFilePointer() - regionStart);
    }

    /**
     * Rotates the centroids into a new array, for the caller that needs them in the stored space — distances in a
     * posting are measured to these, not to the raw ones.
     */
    public static float[][] rotate(float[][] centroids, RotationWriter rotation) throws IOException {
        float[][] rotated = new float[centroids.length][];
        for (int c = 0; c < centroids.length; c++) {
            rotated[c] = new float[rotation.dimension()];
            rotation.rotate(centroids[c], rotated[c]);
        }
        return rotated;
    }

    private static void writeCentroid(IndexOutput clac, float[] centroid) throws IOException {
        float normSq = 0f;
        for (float value : centroid) {
            clac.writeInt(Float.floatToIntBits(value));
            normSq += value * value;
        }
        clac.writeInt(Float.floatToIntBits(normSq));
    }
}
