#!/usr/bin/env bash
# One-shot local setup for Blast web (Chrome) and Rust Bevy GUI demos.
#
# Usage:
#   ./scripts/setup-local-demos.sh              # system deps + emsdk + rust + npm
#   ./scripts/setup-local-demos.sh --build      # above + npm run build:demos + rust release build
#   SKIP_APT=1 ./scripts/setup-local-demos.sh   # skip sudo apt (npm/rust/emsdk only)
#
# After setup:
#   npm start                                 # web demos @ http://localhost:8000
#   npm run start:rust-demo                   # native Bevy GUI

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMSDK_VERSION="${EMSDK_VERSION:-3.1.51}"
EMSDK_DIR="${EMSDK_DIR:-/opt/emsdk}"
RUST_MIN_MAJOR=1
RUST_MIN_MINOR=85

DO_BUILD=0
SKIP_APT="${SKIP_APT:-0}"
SKIP_EMSDK="${SKIP_EMSDK:-0}"
SKIP_RUST="${SKIP_RUST:-0}"
SKIP_NPM="${SKIP_NPM:-0}"

for arg in "$@"; do
  case "${arg}" in
    --build) DO_BUILD=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    --skip-emsdk) SKIP_EMSDK=1 ;;
    --skip-rust) SKIP_RUST=1 ;;
    --skip-npm) SKIP_NPM=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() { echo "[setup-local-demos] $*"; }

have_sudo() {
  command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null
}

install_apt_packages() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    log "Skipping apt packages (SKIP_APT=1)"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; install C++/X11/Vulkan packages manually for the Rust GUI"
    return 0
  fi
  if ! have_sudo; then
    log "sudo not available; set SKIP_APT=1 or install packages manually:"
    log "  build-essential g++ libstdc++-13-dev"
    log "  libx11-dev libxcursor-dev libxi-dev libxrandr-dev libasound2-dev pkg-config"
    log "  libxkbcommon-x11-0 libxkbcommon0 libvulkan1 mesa-vulkan-drivers"
    return 0
  fi

  log "Installing system packages (C++ FFI, Bevy/X11/Vulkan runtime)..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential \
    g++ \
    gcc \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libstdc++-13-dev \
    libx11-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libasound2-dev \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libvulkan1 \
    mesa-vulkan-drivers \
    >/dev/null
}

install_emscripten() {
  if [[ "${SKIP_EMSDK}" == "1" ]]; then
    log "Skipping Emscripten (SKIP_EMSDK=1)"
    return 0
  fi

  # shellcheck source=/dev/null
  if [[ -f "${EMSDK_DIR}/emsdk_env.sh" ]]; then
    source "${EMSDK_DIR}/emsdk_env.sh"
    if command -v emcc >/dev/null 2>&1; then
      log "Emscripten already active: $(emcc --version | head -1)"
      ensure_emsdk_shell_hook
      return 0
    fi
  fi

  if [[ ! -d "${EMSDK_DIR}/.git" ]]; then
  log "Installing Emscripten SDK ${EMSDK_VERSION} at ${EMSDK_DIR}..."
    if [[ "${EMSDK_DIR}" == /opt/* ]] && [[ ! -w "$(dirname "${EMSDK_DIR}")" ]]; then
      sudo git clone --depth 1 https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
      sudo chown -R "$(whoami):$(whoami)" "${EMSDK_DIR}"
    else
      git clone --depth 1 https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
    fi
  fi

  pushd "${EMSDK_DIR}" >/dev/null
  ./emsdk install "${EMSDK_VERSION}"
  ./emsdk activate "${EMSDK_VERSION}"
  popd >/dev/null

  # shellcheck source=/dev/null
  source "${EMSDK_DIR}/emsdk_env.sh"
  log "Emscripten ready: $(emcc --version | head -1)"
  ensure_emsdk_shell_hook
}

ensure_emsdk_shell_hook() {
  local hook="source \"${EMSDK_DIR}/emsdk_env.sh\""
  local rc="${HOME}/.bashrc"
  if [[ -f "${rc}" ]] && ! grep -Fq "${hook}" "${rc}" 2>/dev/null; then
    echo "${hook}" >>"${rc}"
    log "Appended Emscripten hook to ${rc}"
  fi
}

install_rust() {
  if [[ "${SKIP_RUST}" == "1" ]]; then
    log "Skipping Rust (SKIP_RUST=1)"
    return 0
  fi

  if ! command -v rustup >/dev/null 2>&1; then
    log "Installing rustup (stable toolchain)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  fi

  # shellcheck source=/dev/null
  if [[ -f "${HOME}/.cargo/env" ]]; then
    source "${HOME}/.cargo/env"
  fi
  export PATH="${HOME}/.cargo/bin:${PATH}"

  rustup update stable
  rustup default stable

  local version
  version="$(rustc --version)"
  log "Rust ready: ${version}"

  ensure_cargo_shell_hook
}

ensure_cargo_shell_hook() {
  local hook='export PATH="$HOME/.cargo/bin:$PATH"'
  local rc="${HOME}/.bashrc"
  if [[ -f "${rc}" ]] && ! grep -Fq '.cargo/bin' "${rc}" 2>/dev/null; then
    echo "${hook}" >>"${rc}"
    log "Appended cargo PATH to ${rc}"
  fi
}

install_npm_deps() {
  if [[ "${SKIP_NPM}" == "1" ]]; then
    log "Skipping npm install (SKIP_NPM=1)"
    return 0
  fi
  log "Installing npm dependencies (root + blast-stress-solver + js_stress_example)..."
  cd "${ROOT}"
  npm install
  npm --prefix blast/blast-stress-solver install --ignore-scripts
  npm --prefix blast/js_stress_example install
}

build_web_assets() {
  log "Building web demo assets (WASM + TypeScript + esbuild entrypoints)..."
  cd "${ROOT}"
  if [[ -f "${EMSDK_DIR}/emsdk_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${EMSDK_DIR}/emsdk_env.sh"
  fi
  npm run build:demos
}

build_rust_demo() {
  log "Building Rust Bevy GUI (release)..."
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/rust-demo-env.sh"
  "${ROOT}/scripts/run-rust-demo.sh" build
}

print_next_steps() {
  cat <<EOF

[setup-local-demos] Done.

Web demos (Google Chrome or any modern browser):
  cd ${ROOT}
  source ${EMSDK_DIR}/emsdk_env.sh   # if emcc not already on PATH
  npm start                          # build (if needed) + serve http://localhost:8000
  # Open in Chrome:
  #   http://localhost:8000/blast/js_stress_example/demo-index.html
  #   http://localhost:8000/blast/js_stress_example/wall-demolition.html

Rust Bevy GUI:
  cd ${ROOT}
  source scripts/rust-demo-env.sh
  npm run start:rust-demo
  # Controls: left-click shoot, R reset. Default scenario: wall (best mesh rendering).

Preflight checks:
  npm run check:demos
  npm run check:rust-demo

EOF
}

main() {
  install_apt_packages
  install_emscripten
  install_rust
  install_npm_deps

  if [[ "${DO_BUILD}" == "1" ]]; then
    build_web_assets
    build_rust_demo
  fi

  print_next_steps
}

main "$@"
