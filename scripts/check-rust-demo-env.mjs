import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const envScript = resolve(here, 'rust-demo-env.sh');
const demoDir = resolve(root, 'blast/blast-stress-demo-rs');
const binary = resolve(demoDir, 'target/release/blast-stress-demo');

const MIN_RUST_VERSION = [1, 85, 0];
const failures = [];

function run(command, args) {
  return spawnSync(command, args, { encoding: 'utf8' });
}

function shell(command) {
  return run('bash', ['-lc', `export PATH="$HOME/.cargo/bin:$PATH" && source "${envScript}" && ${command}`]);
}

function parseVersion(text) {
  const match = text.match(/(\d+)\.(\d+)\.(\d+)/);
  if (!match) return null;
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function versionAtLeast(actual, required) {
  for (let i = 0; i < 3; i += 1) {
    if (actual[i] > required[i]) return true;
    if (actual[i] < required[i]) return false;
  }
  return true;
}

function checkRust() {
  const cargo = shell('command -v cargo');
  if (cargo.status !== 0 || !cargo.stdout.trim()) {
    failures.push({
      message: 'Rust toolchain (cargo) is not installed or not on PATH.',
      fixCommands: ['  npm run setup:local', '  # or: curl https://sh.rustup.rs | sh -s -- -y && rustup update stable']
    });
    return;
  }

  const version = shell('rustc --version');
  const parsed = parseVersion(version.stdout ?? '');
  if (!parsed || !versionAtLeast(parsed, MIN_RUST_VERSION)) {
    failures.push({
      message: `Rust ${MIN_RUST_VERSION.join('.')}+ is required (Bevy 0.18 / edition2024). Found: ${(version.stdout ?? '').trim() || 'unknown'}.`,
      fixCommands: ['  rustup update stable', '  source "$HOME/.cargo/env"']
    });
  }
}

function checkGpp() {
  const gpp = run('g++', ['--version']);
  if (gpp.status !== 0) {
    failures.push({
      message: 'g++ is required to compile the Blast stress-solver C++ FFI (default c++ is often clang without libstdc++ headers).',
      fixCommands: ['  npm run setup:local', '  # or: sudo apt-get install -y build-essential g++ libstdc++-13-dev']
    });
    return;
  }

  const probe = shell('echo | g++ -x c++ -std=c++17 -c - -o /dev/null');
  if (probe.status !== 0) {
    failures.push({
      message: 'g++ cannot compile C++17 (<new> header missing or broken toolchain).',
      fixCommands: ['  sudo apt-get install -y build-essential g++ libstdc++-13-dev']
    });
  }
}

function checkLibstdcxxLink() {
  const probe = shell('ls /usr/lib/gcc/x86_64-linux-gnu/*/libstdc++.so 2>/dev/null | head -1');
  if (probe.status !== 0 || !probe.stdout.trim()) {
    failures.push({
      message: 'libstdc++.so development symlink not found (Rust link step will fail with "unable to find library -lstdc++").',
      fixCommands: ['  sudo apt-get install -y libstdc++-13-dev', '  source scripts/rust-demo-env.sh']
    });
  }
}

function libraryAvailable(patterns) {
  const ld = run('bash', ['-lc', 'ldconfig -p 2>/dev/null || true']);
  const cache = ld.stdout ?? '';
  for (const pattern of patterns) {
    if (cache.includes(pattern)) {
      return true;
    }
    const candidates = [
      `/usr/lib/x86_64-linux-gnu/${pattern}`,
      `/lib/x86_64-linux-gnu/${pattern}`
    ];
    if (candidates.some((path) => existsSync(path))) {
      return true;
    }
  }
  return false;
}

function checkGuiRuntimeLibs() {
  if (!libraryAvailable(['libxkbcommon-x11.so', 'libxkbcommon_x11.so'])) {
    failures.push({
      message: 'libxkbcommon-x11 is required for the Bevy window (missing at runtime).',
      fixCommands: ['  sudo apt-get install -y libxkbcommon-x11-0 libxkbcommon0 libvulkan1 mesa-vulkan-drivers']
    });
  }
  if (!libraryAvailable(['libxkbcommon.so'])) {
    failures.push({
      message: 'libxkbcommon is required for the Bevy window.',
      fixCommands: ['  sudo apt-get install -y libxkbcommon0']
    });
  }
}

function checkBinary() {
  if (existsSync(binary)) {
    return;
  }
  failures.push({
    message: 'Rust GUI binary is not built yet.',
    fixCommands: ['  npm run build:rust-demo']
  });
}

checkRust();
checkGpp();
checkLibstdcxxLink();
checkGuiRuntimeLibs();
checkBinary();

if (failures.length > 0) {
  console.error('');
  for (const failure of failures) {
    console.error('[rust-demo-preflight] ' + failure.message);
    if (failure.fixCommands.length > 0) {
      console.error('Suggested fix:');
      for (const command of failure.fixCommands) {
        console.error(command);
      }
    }
    console.error('');
  }
  console.error('Full local setup:');
  console.error('  npm run setup:local');
  process.exit(1);
}
