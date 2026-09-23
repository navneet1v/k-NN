#!/usr/bin/env bash
#
# Replays one SearchServicesKnnVectorFormats commit into this k-NN checkout's ClusterANN tree.
#
#   scripts/clusterann-sync-from-formats.sh <formats-commit>              files that commit touched, at that commit
#   scripts/clusterann-sync-from-formats.sh <formats-commit> --at <ref>   same file set, but content as of <ref>
#   scripts/clusterann-sync-from-formats.sh --all <ref>                   every ClusterANN file as of <ref>
#
# Only files under the synced packages and test resources are touched (see clusterann-formats-rewrite.sh);
# formats' build files are skipped and listed so the caller can port them by hand if needed. New files get the
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

# Split "src/<set>/java/<pkgpath>" into set + package-relative path. HappierTrails copies non-Java files from
# the java source dirs, so formats keeps its META-INF service files under src/{main,test}/java/META-INF; k-NN
# wants them under src/{main,test}/resources/META-INF, so they are classed as resources.
split_path() { # $1 = formats path -> prints "<set> <fmt-rel>" or nothing
  local p="$1" set rest
  case "$p" in
    src/main/java/META-INF/*) set=main-res; rest="${p#src/main/java/}" ;;
    src/test/java/META-INF/*) set=test-res; rest="${p#src/test/java/}" ;;
    src/main/java/*) set=main; rest="${p#src/main/java/}" ;;
    src/test/java/*) set=test; rest="${p#src/test/java/}" ;;
    src/test/resources/*) set=test-res; rest="${p#src/test/resources/}" ;;
    *) return ;;
  esac
  echo "$set $rest"
}

# Resolve a formats-relative path to its k-NN destination, or print nothing when outside the synced set.
knn_path() { # $1 = set, $2 = formats rel
  local rel
  case "$1" in
    main-res)
      rel="$(fmt_res_to_knn_res "$2")"
      [[ -n "$rel" ]] && echo "$KNN_ROOT/src/main/resources/$rel" ;;
    test-res)
      rel="$(fmt_res_to_knn_res "$2")"
      [[ -n "$rel" ]] && echo "$KNN_ROOT/src/test/resources/$rel" ;;
    *)
      rel="$(fmt_rel_to_knn_rel "$2")"
      [[ -n "$rel" ]] && echo "$KNN_ROOT/src/$1/java/$rel" ;;
  esac
}

is_java() { [[ "$1" == *.java ]]; }

write_file() { # $1 = ref, $2 = formats path, $3 = destination
  local content
  mkdir -p "$(dirname "$3")"
  if is_java "$2"; then
    content="$(git -C "$FMT" show "$1:$2" | rewrite_fmt_to_knn)"
    if [[ "$content" == "/*"* ]]; then
      printf '%s\n' "$content" > "$3"
    else
      printf '%s\n%s\n' "$SPDX_HEADER" "$content" > "$3"
    fi
  elif [[ "$2" == src/main/java/META-INF/services/* && -f "$3" ]]; then
    # k-NN's main service files list its own providers too: merge formats' entries in, keep the rest.
    while IFS= read -r line; do
      [[ -z "$line" || "$line" == \#* ]] && continue
      grep -qxF "$line" "$3" || printf '%s\n' "$line" >> "$3"
    done < <(git -C "$FMT" show "$1:$2" | rewrite_fmt_to_knn)
  elif [[ "$2" == */META-INF/services/* ]]; then
    git -C "$FMT" show "$1:$2" | rewrite_fmt_to_knn > "$3"
  else
    git -C "$FMT" show "$1:$2" > "$3"   # non-Java resource: verbatim
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

# Destination for a formats path, or empty (and recorded as skipped when $2 is given) if outside the synced set.
dest_for() { # $1 = formats path, [$2 = label to record on skip]
  local set rel d
  read -r set rel <<< "$(split_path "$1")" || true
  d=""
  [[ -n "${set:-}" ]] && d="$(knn_path "$set" "$rel")"
  if [[ -z "$d" && -n "${2:-}" ]]; then skipped+=("$2"); fi
  echo "$d"
}

if [[ -n "$ALL" ]]; then
  echo "== materializing every ClusterANN file as of formats $ALL =="
  while IFS= read -r p; do
    d="$(dest_for "$p")"
    [[ -z "$d" ]] && continue
    write_file "$ALL" "$p" "$d"
  done < <(git -C "$FMT" ls-tree -r --name-only "$ALL" -- src/main/java src/test/java src/test/resources)
  exit 0
fi

STATE="${AT:-$COMMIT}"
echo "== replaying formats $(git -C "$FMT" log --format='%h %s' -1 "$COMMIT") (content as of $STATE) =="
while IFS=$'\t' read -r status p1 p2; do
  kind="${status:0:1}"
  case "$kind" in
    D)
      d="$(dest_for "$p1" "$status $p1")"; [[ -z "$d" ]] && continue
      remove_file "$d"
      ;;
    R)
      d="$(dest_for "$p1")"; [[ -n "$d" ]] && remove_file "$d"
      d="$(dest_for "$p2" "$status $p1 -> $p2")"; [[ -z "$d" ]] && continue
      write_file "$STATE" "$p2" "$d"
      ;;
    A|M)
      d="$(dest_for "$p1" "$status $p1")"; [[ -z "$d" ]] && continue
      if git -C "$FMT" cat-file -e "$STATE:$p1" 2>/dev/null; then
        write_file "$STATE" "$p1" "$d"
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
