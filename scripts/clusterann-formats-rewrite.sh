#!/usr/bin/env bash
#
# Shared by clusterann-drift.sh and clusterann-sync-from-formats.sh: how a ClusterANN source file in
# SearchServicesKnnVectorFormats maps onto this k-NN checkout. Sourced, not executed.
#
# The two trees are meant to be the same implementation. Everything below is the fixed, mechanical set of
# differences between the two homes, and nothing else should ever be in this list:
#
#   1. Package prefix: org.opensearch.knn.vectorformats.clusterann -> org.opensearch.knn.clusterann.
#   2. Formats keeps ScalarQuantizers and Int4DotProduct in a sibling `quantization` package; k-NN keeps them
#      inside clusterann/read/block/scalar.
#   3. Formats builds against a juno-patched Lucene 10.3 that has Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding;
#      upstream Lucene 10.3.2 (what k-NN builds against) does not, so k-NN carries its own copy of that enum
#      at clusterann/read/block/scalar/ScalarEncoding, and OptimizedScalarQuantizer.transposeDibit plus
#      VectorUtil.int4DotProductSinglePacked (also 10.4) are redirected to clusterann/read/block/scalar/Lucene104Backports.
#   4. javax.annotation is not on k-NN's classpath: Nullable becomes org.opensearch.common.Nullable, and the
#      thread-safety markers are dropped. Neither is spotbugs-annotations: @SuppressFBWarnings is dropped.
#   5. k-NN requires the OpenSearch SPDX header on every file and runs spotless; formats has neither.
#   6. The codec layer (the Lucene KnnVectorsFormat/Writer/Reader triple, the test codec and the suite-driven
#      component test) lives at org.opensearch.knn.vectorformats.codec in formats and at
#      org.opensearch.knn.index.codec.clusterann in k-NN, class names unchanged. Test resources (suite YAML,
#      baselines, the committed 1k corpus) are copied verbatim; the codec SPI file moves from formats'
#      src/test/java/META-INF to k-NN's src/test/resources/META-INF.

FMT_PKG='org.opensearch.knn.vectorformats.clusterann'
KNN_PKG='org.opensearch.knn.clusterann'
FMT_PKG_PATH='org/opensearch/knn/vectorformats/clusterann'
KNN_PKG_PATH='org/opensearch/knn/clusterann'
FMT_QUANT_PATH='org/opensearch/knn/vectorformats/quantization'
KNN_QUANT_PATH="$KNN_PKG_PATH/read/block/scalar"   # where formats' quantization/ files land
FMT_CODEC_PATH='org/opensearch/knn/vectorformats/codec'
KNN_CODEC_PATH='org/opensearch/knn/index/codec/clusterann'

# Every k-NN package root the synced set lands in, relative to src/<set>/java/.
KNN_ROOTS=("$KNN_PKG_PATH" "$KNN_CODEC_PATH")

SPDX_HEADER='/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
'

# Map a formats source path (relative to src/<set>/java/) to a k-NN path relative to src/<set>/java/,
# or print nothing when the file is outside the synced set.
fmt_rel_to_knn_rel() {
  local p="$1"
  case "$p" in
    "$FMT_PKG_PATH"/*) echo "$KNN_PKG_PATH/${p#"$FMT_PKG_PATH"/}" ;;
    "$FMT_QUANT_PATH"/*) echo "$KNN_QUANT_PATH/${p#"$FMT_QUANT_PATH"/}" ;;
    "$FMT_CODEC_PATH"/*) echo "$KNN_CODEC_PATH/${p#"$FMT_CODEC_PATH"/}" ;;
    *) ;;
  esac
}

# Map a formats test resource path (relative to src/test/resources/, or META-INF/... from src/test/java/) to
# its k-NN path relative to src/test/resources/, or print nothing when it is not part of the synced set.
fmt_res_to_knn_res() {
  local p="$1"
  case "$p" in
    META-INF/services/*) echo "$p" ;;
    baselines/*|dataset/*|*-suite.yml) echo "$p" ;;
    *) ;;
  esac
}

# Rewrite formats file content (stdin) into what the k-NN copy should contain, minus header and formatting.
rewrite_fmt_to_knn() {
  perl -pe '
    s/\borg\.opensearch\.knn\.vectorformats\.clusterann\b/org.opensearch.knn.clusterann/g;
    s/\borg\.opensearch\.knn\.vectorformats\.quantization\b/org.opensearch.knn.clusterann.read.block.scalar/g;
    s/\borg\.opensearch\.knn\.vectorformats\.codec\b/org.opensearch.knn.index.codec.clusterann/g;
    $_ = "" if /^import edu\.umd\.cs\.findbugs\.annotations\.SuppressFBWarnings;$/;
    s/^import org\.apache\.lucene\.codecs\.lucene103\.Lucene103ScalarQuantizedVectorsFormat(\.ScalarEncoding)?;$/import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;/;
    s/\bLucene103ScalarQuantizedVectorsFormat\.ScalarEncoding\b/ScalarEncoding/g;
    s/\bOptimizedScalarQuantizer\.transposeDibit\b/Lucene104Backports.transposeDibit/g;
    s/\bVectorUtil\.int4DotProductSinglePacked\b/Lucene104Backports.int4DotProductSinglePacked/g;
    s/^import javax\.annotation\.Nullable;$/import org.opensearch.common.Nullable;/;
    $_ = "" if /^import javax\.annotation\.concurrent\.(Not)?ThreadSafe;$/;
    $_ = "" if /^\@(Not)?ThreadSafe$/;
  ' | awk '!(/^import / && seen[$0]++)' | drop_same_package_imports | drop_fb_annotation | add_backport_import
}

# spotbugs-annotations is not on k-NN's classpath; the annotation is a spotbugs-only hint, so it goes (its
# import is dropped line-wise above). The block may span lines with one level of nested parentheses.
drop_fb_annotation() {
  perl -0pe 's/^\@SuppressFBWarnings\((?:[^()]|\([^()]*\))*\)\n//mg'
}

# The rewrite can turn a cross-package import into one naming the file's own package (formats imports
# ScalarEncoding from Lucene into read.block.scalar; k-NN keeps it there). Spotless would strip it anyway.
drop_same_package_imports() {
  perl -0pe '
    if (/^package ([\w.]+);/m) {
      my $pkg = quotemeta($1);
      s/^import $pkg\.\w+;\n//mg;
    }'
}

# A file that now calls Lucene104Backports but lives outside read.block.scalar needs the import; put it next to
# the Lucene import it replaced (OptimizedScalarQuantizer for transposeDibit, VectorUtil for int4DotProductSinglePacked)
# so spotless leaves it in the same group.
add_backport_import() {
  perl -0pe '
    if (/\bLucene104Backports\./ && !/^package org\.opensearch\.knn\.clusterann\.read\.block\.scalar;/m
        && !/^import org\.opensearch\.knn\.clusterann\.read\.block\.scalar\.Lucene104Backports;/m) {
      s/^(import org\.apache\.lucene\.util\.(?:quantization\.OptimizedScalarQuantizer|VectorUtil);\n)/$1import org.opensearch.knn.clusterann.read.block.scalar.Lucene104Backports;\n/m;
    }'
}

# Remove a leading SPDX/license comment block and squeeze blank-line runs (stdin -> stdout).
strip_header() {
  awk 'NR==1 && /^\/\*$/ {skip=1} skip { if (/^ \*\/$/) {skip=0; blank=1}; next } blank && /^$/ {blank=0; next} {blank=0; print}' | cat -s
}

# Fingerprint of a Java file for comparison (stdin -> stdout). Formats has no formatter and k-NN's spotless
# rewraps long signatures and drops unused imports, so byte identity is not the bar: tokens outside the import
# block are. An import that matters shows up as a changed reference in the body.
fingerprint() {
  grep -v '^import ' | tr -d '[:space:]'
}
