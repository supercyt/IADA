#!/usr/bin/env bash
# Keep only AP30/AP50/AP70 in eval_intermediate*.yaml files.

set -euo pipefail

apply=false
backup=false
log_dir="opencood/logs"

usage() {
    cat <<'EOF'
Usage: ./trim_eval_intermediate.sh [--apply] [--backup] [LOG_DIR]

By default, only show what would be changed. Pass --apply to rewrite files.
Pass --backup to keep each original as FILE.bak (requires --apply).

Examples:
  ./trim_eval_intermediate.sh
  ./trim_eval_intermediate.sh --apply
  ./trim_eval_intermediate.sh --apply --backup opencood/logs
EOF
}

while (($#)); do
    case "$1" in
        --apply) apply=true ;;
        --backup) backup=true ;;
        -h|--help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        *) log_dir=$1 ;;
    esac
    shift
done

if $backup && ! $apply; then
    printf '%s\n' '--backup must be used together with --apply' >&2
    exit 2
fi

if [[ ! -d "$log_dir" ]]; then
    printf 'Log directory does not exist: %s\n' "$log_dir" >&2
    exit 1
fi

mapfile -d '' files < <(
    find "$log_dir" -type f -name 'eval_intermediate*.yaml' -print0
)

if ((${#files[@]} == 0)); then
    printf 'No eval_intermediate*.yaml files found under %s\n' "$log_dir"
    exit 0
fi

declare -a candidates=()
before_bytes=0

# Validate every candidate before modifying anything, so an unexpected YAML
# layout cannot leave the directory only partly processed.
for file in "${files[@]}"; do
    mapfile -t first_lines < <(sed -n '1,4p' "$file")

    if ((${#first_lines[@]} < 3)) \
        || [[ ! ${first_lines[0]} =~ ^[[:space:]]*ap30: ]] \
        || [[ ! ${first_lines[1]} =~ ^[[:space:]]*ap_50: ]] \
        || [[ ! ${first_lines[2]} =~ ^[[:space:]]*ap_70: ]]; then
        printf 'Unexpected header; refusing to process: %s\n' "$file" >&2
        exit 1
    fi

    # A file with exactly these three lines is already compact.
    if ((${#first_lines[@]} > 3)); then
        candidates+=("$file")
        size=$(stat -c '%s' "$file")
        ((before_bytes += size)) || true
    fi
done

printf 'Found %d matching files; %d need trimming (%.2f MiB).\n' \
    "${#files[@]}" "${#candidates[@]}" "$(awk -v n="$before_bytes" 'BEGIN {print n/1048576}')"

if ! $apply; then
    printf '%s\n' 'Dry run only. Re-run with --apply to keep only the first three lines.'
    exit 0
fi

if $backup; then
    for file in "${candidates[@]}"; do
        if [[ -e "$file.bak" ]]; then
            printf 'Backup already exists; refusing to overwrite: %s\n' "$file.bak" >&2
            exit 1
        fi
    done
fi

for file in "${candidates[@]}"; do
    if $backup; then
        cp -p -- "$file" "$file.bak"
    fi

    tmp=$(mktemp "${file}.tmp.XXXXXX")
    sed -n '1,3p' "$file" > "$tmp"
    chmod --reference="$file" "$tmp"
    mv -- "$tmp" "$file"
done

printf 'Trimmed %d files.\n' "${#candidates[@]}"
