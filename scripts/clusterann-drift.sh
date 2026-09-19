#!/usr/bin/env bash
#
# Reports drift between the ClusterANN sources in SearchServicesKnnVectorFormats (the reference) and this
# k-NN checkout. The two trees are meant to be identical apart from the package prefix, so a clean run
# prints nothing and exits 0.
#
#   scripts/clusterann-drift.sh                 compare against the formats working tree
#   scripts/clusterann-drift.sh <git-ref>       compare against that formats commit instead
#   FMT=/path/to/SearchServicesKnnVectorFormats scripts/clusterann-drift.sh
#
# Exit status: 0 no drift, 1 drift found, 2 usage/setup error.
set -euo pipefail

KNN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FMT="${FMT:-$KNN_ROOT/../SearchServicesKNNVectorsFormat/src/SearchServicesKnnVectorFormats}"
REF="${1:-}"

FMT_PKG_PATH="org/opensearch/knn/vectorformats/clusterann"
KNN_PKG_PATH="org/opensearch/knn/clusterann"
FMT_PKG="org.opensearch.knn.vectorformats.clusterann"
KNN_PKG="org.opensearch.knn.clusterann"

# Files k-NN is allowed to have without a formats counterpart (relative to src/{main,test}/java/<pkg path>).
KNN_ONLY_ALLOWLIST=(
)

if [[ ! -d "$FMT" ]]; then
  echo "formats package not found at $FMT (set FMT=...)" >&2
  exit 2
fi

# List formats files as paths relative to the clusterann package root, for one source set.
fmt_list() { # $1 = main|test
  if [[ -n "$REF" ]]; then
    git -C "$FMT" ls-tree -r --name-only "$REF" -- "src/$1/java/$FMT_PKG_PATH" \
      | sed "s|^src/$1/java/$FMT_PKG_PATH/||"
  else
    (cd "$FMT/src/$1/java/$FMT_PKG_PATH" 2>/dev/null && find . -name '*.java' | sed 's|^\./||') || true
  fi | sort
}

# Strip a leading OpenSearch SPDX license block: k-NN requires it on every file, formats carries it only on
# some, and it is not part of the format implementation being compared.
strip_header() {
  awk 'NR==1 && /^\/\*$/ {skip=1} skip { if (/^ \*\/$/) {skip=0; blank=1}; next } blank && /^$/ {blank=0; next} {blank=0; print}'
}

fmt_cat() { # $1 = main|test, $2 = relative path; prints the formats file with the k-NN package prefix
  if [[ -n "$REF" ]]; then
    git -C "$FMT" show "$REF:src/$1/java/$FMT_PKG_PATH/$2"
  else
    cat "$FMT/src/$1/java/$FMT_PKG_PATH/$2"
  fi | sed "s/${FMT_PKG//./\\.}/$KNN_PKG/g" | strip_header
}

knn_cat() { strip_header < "$KNN_ROOT/src/$1/java/$KNN_PKG_PATH/$2"; }

knn_list() { # $1 = main|test
  (cd "$KNN_ROOT/src/$1/java/$KNN_PKG_PATH" 2>/dev/null && find . -name '*.java' | sed 's|^\./||') | sort
}

is_allowlisted() {
  local f
  for f in "${KNN_ONLY_ALLOWLIST[@]:-}"; do [[ "$1" == "$f" ]] && return 0; done
  return 1
}

drift=0
for set in main test; do
  fmt_files="$(fmt_list "$set")"
  knn_files="$(knn_list "$set")"

  only_fmt="$(comm -23 <(echo "$fmt_files") <(echo "$knn_files") | sed '/^$/d')"
  only_knn="$(comm -13 <(echo "$fmt_files") <(echo "$knn_files") | sed '/^$/d')"
  both="$(comm -12 <(echo "$fmt_files") <(echo "$knn_files") | sed '/^$/d')"

  if [[ -n "$only_fmt" ]]; then
    drift=1
    echo "== $set: in formats, missing from k-NN =="
    echo "$only_fmt" | sed 's/^/  /'
  fi

  unexpected_knn=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    is_allowlisted "$set/$f" || unexpected_knn+="$f"$'\n'
  done <<< "$only_knn"
  if [[ -n "$unexpected_knn" ]]; then
    drift=1
    echo "== $set: in k-NN, missing from formats (not allowlisted) =="
    printf '%s' "$unexpected_knn" | sed 's/^/  /'
  fi

  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if ! diff -q <(fmt_cat "$set" "$f") <(knn_cat "$set" "$f") >/dev/null; then
      drift=1
      echo "== $set: content differs: $f =="
      diff -u --label "formats/$f" --label "k-NN/$f" <(fmt_cat "$set" "$f") <(knn_cat "$set" "$f") | sed -n "1,${DRIFT_DIFF_LINES:-40}p" || true
    fi
  done <<< "$both"
done

exit $drift
