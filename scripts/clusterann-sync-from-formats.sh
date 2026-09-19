#!/usr/bin/env bash
#
# Replays one SearchServicesKnnVectorFormats commit into this k-NN checkout's ClusterANN tree.
#
#   scripts/clusterann-sync-from-formats.sh <formats-commit>              files that commit touched, at that commit
#   scripts/clusterann-sync-from-formats.sh <formats-commit> --at <ref>   same file set, but content as of <ref>
#   scripts/clusterann-sync-from-formats.sh --all <ref>                   every ClusterANN file as of <ref>
#
# Only files under the synced packages are touched (see clusterann-formats-rewrite.sh); the formats codec and
# build files are skipped and listed so the caller can port them by hand if needed. New files get the
# OpenSearch SPDX header. Nothing is committed and spotless is not run: review, `./gradlew spotlessApply`,
# build, test, then commit with the formats hash in the message.
set -euo pipefail

KNN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FMT="${FMT:-$KNN_ROOT/../SearchServicesKNNVectorsFormat/src/SearchServicesKnnVectorFormats}"
# shellcheck source=clusterann-formats-rewrite.sh
source "$KNN_ROOT/scripts/clusterann-formats-rewrite.sh"

usage() { sed -n '2,13p' "$0" >&2; exit 2; }

COMMIT=""; AT=""; ALL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --at) AT="$2"; shift 2 ;;
    --all) ALL="$2"; shift 2 ;;
    -*) usage ;;
    *) COMMIT="$1"; shift ;;
  esac
done
[[ -n "$COMMIT" || -n "$ALL" ]] || usage
[[ -d "$FMT/.git" || -f "$FMT/.git" ]] || { echo "formats package not found at $FMT" >&2; exit 2; }

# Split "src/<set>/java/<pkgpath>" into set + package-relative path.
split_path() { # $1 = formats path -> prints "<set> <fmt-rel>" or nothing
  local p="$1" set rest
  case "$p" in
    src/main/java/*) set=main; rest="${p#src/main/java/}" ;;
    src/test/java/*) set=test; rest="${p#src/test/java/}" ;;
    *) return ;;
  esac
  echo "$set $rest"
}

knn_path() { # $1 = set, $2 = k-NN rel -> absolute k-NN path
  echo "$KNN_ROOT/src/$1/java/$KNN_PKG_PATH/$2"
}

write_file() { # $1 = ref, $2 = formats path, $3 = destination
  local content
  content="$(git -C "$FMT" show "$1:$2" | rewrite_fmt_to_knn)"
  mkdir -p "$(dirname "$3")"
  if [[ "$content" == "/*"* ]]; then
    printf '%s\n' "$content" > "$3"
  else
    printf '%s\n%s\n' "$SPDX_HEADER" "$content" > "$3"
  fi
  git -C "$KNN_ROOT" add "$3"
  echo "  wrote   ${3#"$KNN_ROOT"/}"
}

remove_file() { # $1 = destination
  if [[ -f "$1" ]]; then
    git -C "$KNN_ROOT" rm -q "$1"
    echo "  removed ${1#"$KNN_ROOT"/}"
  fi
}

skipped=()

if [[ -n "$ALL" ]]; then
  echo "== materializing every ClusterANN file as of formats $ALL =="
  while IFS= read -r p; do
    read -r set rel <<< "$(split_path "$p")" || true
    [[ -z "${set:-}" ]] && continue
    knn_rel="$(fmt_rel_to_knn_rel "$rel")"
    [[ -z "$knn_rel" ]] && continue
    write_file "$ALL" "$p" "$(knn_path "$set" "$knn_rel")"
  done < <(git -C "$FMT" ls-tree -r --name-only "$ALL" -- src/main/java src/test/java)
  exit 0
fi

STATE="${AT:-$COMMIT}"
echo "== replaying formats $(git -C "$FMT" log --format='%h %s' -1 "$COMMIT") (content as of $STATE) =="
while IFS=$'\t' read -r status p1 p2; do
  kind="${status:0:1}"
  case "$kind" in
    D)
      read -r set rel <<< "$(split_path "$p1")" || true
      [[ -z "${set:-}" ]] && { skipped+=("$status $p1"); continue; }
      knn_rel="$(fmt_rel_to_knn_rel "$rel")"; [[ -z "$knn_rel" ]] && { skipped+=("$status $p1"); continue; }
      remove_file "$(knn_path "$set" "$knn_rel")"
      ;;
    R)
      read -r set rel <<< "$(split_path "$p1")" || true
      if [[ -n "${set:-}" ]]; then
        knn_rel="$(fmt_rel_to_knn_rel "$rel")"; [[ -n "$knn_rel" ]] && remove_file "$(knn_path "$set" "$knn_rel")"
      fi
      read -r set rel <<< "$(split_path "$p2")" || true
      [[ -z "${set:-}" ]] && { skipped+=("$status $p1 -> $p2"); continue; }
      knn_rel="$(fmt_rel_to_knn_rel "$rel")"; [[ -z "$knn_rel" ]] && { skipped+=("$status $p1 -> $p2"); continue; }
      write_file "$STATE" "$p2" "$(knn_path "$set" "$knn_rel")"
      ;;
    A|M)
      read -r set rel <<< "$(split_path "$p1")" || true
      [[ -z "${set:-}" ]] && { skipped+=("$status $p1"); continue; }
      knn_rel="$(fmt_rel_to_knn_rel "$rel")"; [[ -z "$knn_rel" ]] && { skipped+=("$status $p1"); continue; }
      if git -C "$FMT" cat-file -e "$STATE:$p1" 2>/dev/null; then
        write_file "$STATE" "$p1" "$(knn_path "$set" "$knn_rel")"
      else
        echo "  (gone by $STATE, not written) $p1"
      fi
      ;;
    *) skipped+=("$status $p1") ;;
  esac
done < <(git -C "$FMT" show --name-status --format= -M "$COMMIT")

if [[ ${#skipped[@]} -gt 0 ]]; then
  echo "== outside the synced packages, not touched (port by hand if k-NN needs it) =="
  printf '  %s\n' "${skipped[@]}"
fi
