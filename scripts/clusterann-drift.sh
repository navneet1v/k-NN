#!/usr/bin/env bash
#
# Reports drift between the ClusterANN sources in SearchServicesKnnVectorFormats (the reference) and this
# k-NN checkout. The two trees are meant to be the same implementation apart from the fixed differences
# listed in clusterann-formats-rewrite.sh, so a clean run prints nothing and exits 0.
#
#   scripts/clusterann-drift.sh                 compare against the formats working tree
#   scripts/clusterann-drift.sh <git-ref>       compare against that formats commit instead
#   FMT=/path/to/SearchServicesKnnVectorFormats scripts/clusterann-drift.sh
#   DRIFT_DIFF_LINES=200 scripts/clusterann-drift.sh    show more of each differing file (default 40)
#
# Exit status: 0 no drift, 1 drift found, 2 usage/setup error.
set -euo pipefail

KNN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FMT="${FMT:-$KNN_ROOT/../SearchServicesKNNVectorsFormat/src/SearchServicesKnnVectorFormats}"
REF="${1:-}"
# shellcheck source=clusterann-formats-rewrite.sh
source "$KNN_ROOT/scripts/clusterann-formats-rewrite.sh"

# k-NN files allowed to have no formats counterpart, as "<set>/<path relative to the clusterann root>".
KNN_ONLY_ALLOWLIST=(
  # Copy of Lucene 10.4's Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding; formats gets it from its
  # juno-patched Lucene 10.3, k-NN's upstream Lucene 10.3.2 does not have it.
  main/read/block/scalar/ScalarEncoding.java
  # OptimizedScalarQuantizer.transposeDibit, same story.
  main/read/block/scalar/Lucene104Backports.java
)

if [[ ! -d "$FMT" ]]; then
  echo "formats package not found at $FMT (set FMT=...)" >&2
  exit 2
fi

# Formats files for one source set, as "<formats path under src/<set>/java>|<k-NN rel path>", sorted by k-NN path.
fmt_list() { # $1 = main|test
  local p rel
  {
    if [[ -n "$REF" ]]; then
      git -C "$FMT" ls-tree -r --name-only "$REF" -- "src/$1/java" | sed "s|^src/$1/java/||"
    else
      (cd "$FMT/src/$1/java" 2>/dev/null && find . -name '*.java' | sed 's|^\./||') || true
    fi
  } | while IFS= read -r p; do
    rel="$(fmt_rel_to_knn_rel "$p")"
    [[ -n "$rel" ]] && echo "$rel|$p"
  done | sort -t'|' -k1,1
}

fmt_cat() { # $1 = main|test, $2 = formats path under src/<set>/java
  if [[ -n "$REF" ]]; then
    git -C "$FMT" show "$REF:src/$1/java/$2"
  else
    cat "$FMT/src/$1/java/$2"
  fi | rewrite_fmt_to_knn | strip_header
}

knn_list() { # $1 = main|test
  (cd "$KNN_ROOT/src/$1/java/$KNN_PKG_PATH" 2>/dev/null && find . -name '*.java' | sed 's|^\./||') | sort
}

knn_cat() { strip_header < "$KNN_ROOT/src/$1/java/$KNN_PKG_PATH/$2"; }

is_allowlisted() {
  local f
  for f in "${KNN_ONLY_ALLOWLIST[@]}"; do [[ "$1" == "$f" ]] && return 0; done
  return 1
}

drift=0
for set in main test; do
  fmt_pairs="$(fmt_list "$set")"
  fmt_files="$(echo "$fmt_pairs" | cut -d'|' -f1)"
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
    fmt_path="$(echo "$fmt_pairs" | awk -F'|' -v k="$f" '$1==k {print $2; exit}')"
    if [[ "$(fmt_cat "$set" "$fmt_path" | fingerprint)" != "$(knn_cat "$set" "$f" | fingerprint)" ]]; then
      drift=1
      echo "== $set: content differs: $f =="
      diff -u --label "formats/$f" --label "k-NN/$f" <(fmt_cat "$set" "$fmt_path") <(knn_cat "$set" "$f") | sed -n "1,${DRIFT_DIFF_LINES:-40}p" || true
    fi
  done <<< "$both"
done

exit $drift
