#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/rust-demo-env.sh"

DEMO_DIR="${ROOT}/blast/blast-stress-demo-rs"
BINARY="${DEMO_DIR}/target/release/blast-stress-demo"

export BLAST_STRESS_DEMO_SCENARIO="${BLAST_STRESS_DEMO_SCENARIO:-wall}"
export BLAST_STRESS_DEMO_SHOW_MESHES="${BLAST_STRESS_DEMO_SHOW_MESHES:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/run-rust-demo.sh [build|run]

  build   cargo build --release for blast-stress-demo-rs
  run     launch the GUI (default)

Environment:
  BLAST_STRESS_DEMO_SCENARIO=wall|bridge|tower|fractured-bridge|...
  BLAST_STRESS_DEMO_SHOW_MESHES=1|0
  BLAST_STRESS_DEMO_GUI_SHOT_SCRIPT=auto_smoke|bridge_smoke|...  # scripted shots in GUI (same as headless)
  DISPLAY must be set for GUI mode (omit BLAST_STRESS_DEMO_HEADLESS).

Examples:
  npm run build:rust-demo
  npm run start:rust-demo
  BLAST_STRESS_DEMO_SCENARIO=bridge npm run start:rust-demo
EOF
}

cmd="${1:-run}"

case "${cmd}" in
  build)
    cd "${DEMO_DIR}"
    cargo build --release
    ;;
  run)
    if [[ ! -x "${BINARY}" ]]; then
      echo "[run-rust-demo] Binary missing; building release target..." >&2
      "${ROOT}/scripts/run-rust-demo.sh" build
    fi
    if [[ -z "${DISPLAY:-}" && -z "${BLAST_STRESS_DEMO_HEADLESS:-}" ]]; then
      echo "[run-rust-demo] WARNING: DISPLAY is unset; Bevy may fall back to headless mode." >&2
    fi
    cd "${DEMO_DIR}"
    exec "${BINARY}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
