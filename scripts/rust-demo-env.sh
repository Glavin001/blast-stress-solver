#!/usr/bin/env bash
# Source this file before building or running blast-stress-demo-rs (Bevy GUI).
#   source scripts/rust-demo-env.sh

export PATH="${HOME}/.cargo/bin:${PATH}"

# Prefer rustup's stable toolchain over image-bundled /usr/local/cargo (often too old for Bevy 0.18).
if [[ -x "${HOME}/.cargo/bin/rustc" ]]; then
  export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
fi

export CXX="${CXX:-g++}"
export CC="${CC:-gcc}"

_blast_libstdcxx_search_path() {
  local dir paths=()
  if [[ -d /usr/lib/gcc/x86_64-linux-gnu ]]; then
    for dir in /usr/lib/gcc/x86_64-linux-gnu/*/; do
      if [[ -f "${dir}libstdc++.so" ]]; then
        paths+=("${dir%/}")
      fi
    done
  fi
  if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
    paths+=("/usr/lib/x86_64-linux-gnu")
  fi
  if ((${#paths[@]} == 0)); then
    return 1
  fi
  local IFS=:
  echo "${paths[*]}"
}

if _blast_paths="$(_blast_libstdcxx_search_path)"; then
  export LIBRARY_PATH="${_blast_paths}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

unset _blast_paths _blast_libstdcxx_search_path
