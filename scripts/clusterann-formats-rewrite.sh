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
#      at clusterann/read/block/scalar/ScalarEncoding, and OptimizedScalarQuantizer.transposeDibit (also 10.4) is
#      redirected to clusterann/read/block/scalar/Lucene104Backports.
#   4. javax.annotation is not on k-NN's classpath: Nullable becomes org.opensearch.common.Nullable, and the
#      thread-safety markers are dropped.
#   5. k-NN requires the OpenSearch SPDX header on every file and runs spotless; formats has neither.

FMT_PKG='org.opensearch.knn.vectorformats.clusterann'
KNN_PKG='org.opensearch.knn.clusterann'
FMT_PKG_PATH='org/opensearch/knn/vectorformats/clusterann'
KNN_PKG_PATH='org/opensearch/knn/clusterann'
FMT_QUANT_PATH='org/opensearch/knn/vectorformats/quantization'
KNN_QUANT_REL='read/block/scalar'   # where formats' quantization/ files land, relative to the k-NN clusterann root

SPDX_HEADER='/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
'

# Map a formats source path (relative to src/<set>/java/) to a k-NN path relative to the clusterann root,
# or print nothing when the file is outside the synced set.
fmt_rel_to_knn_rel() {
  local p="$1"
  case "$p" in
    "$FMT_PKG_PATH"/*) echo "${p#"$FMT_PKG_PATH"/}" ;;
    "$FMT_QUANT_PATH"/*) echo "$KNN_QUANT_REL/${p#"$FMT_QUANT_PATH"/}" ;;
    *) ;;
  esac
}

# Rewrite formats file content (stdin) into what the k-NN copy should contain, minus header and formatting.
rewrite_fmt_to_knn() {
  perl -pe '
    s/\borg\.opensearch\.knn\.vectorformats\.clusterann\b/org.opensearch.knn.clusterann/g;
    s/\borg\.opensearch\.knn\.vectorformats\.quantization\b/org.opensearch.knn.clusterann.read.block.scalar/g;
    s/^import org\.apache\.lucene\.codecs\.lucene103\.Lucene103ScalarQuantizedVectorsFormat(\.ScalarEncoding)?;$/import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;/;
    s/\bLucene103ScalarQuantizedVectorsFormat\.ScalarEncoding\b/ScalarEncoding/g;
    s/\bOptimizedScalarQuantizer\.transposeDibit\b/Lucene104Backports.transposeDibit/g;
    s/^import javax\.annotation\.Nullable;$/import org.opensearch.common.Nullable;/;
    $_ = "" if /^import javax\.annotation\.concurrent\.(Not)?ThreadSafe;$/;
    $_ = "" if /^\@(Not)?ThreadSafe$/;
  ' | awk '!(/^import / && seen[$0]++)' | drop_same_package_imports | add_backport_import
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
# the OptimizedScalarQuantizer import so spotless leaves it in the same group.
add_backport_import() {
  perl -0pe '
    if (/\bLucene104Backports\./ && !/^package org\.opensearch\.knn\.clusterann\.read\.block\.scalar;/m
        && !/^import org\.opensearch\.knn\.clusterann\.read\.block\.scalar\.Lucene104Backports;/m) {
      s/^(import org\.apache\.lucene\.util\.quantization\.OptimizedScalarQuantizer;\n)/$1import org.opensearch.knn.clusterann.read.block.scalar.Lucene104Backports;\n/m;
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
