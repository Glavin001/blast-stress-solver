#!/usr/bin/env bash
# Build a component and audit it, in one command.
#
# The structural loop used to be: edit here, run node there, cd to the game
# repo, remember the LD_LIBRARY_PATH, run cargo with four env vars, grep the
# output. Six steps between having an idea and knowing if it was right, which
# is enough friction that you batch changes -- and batching is exactly how you
# end up unable to say which of them helped.
#
#   ./comp.sh room                    # build + audit one component
#   ./comp.sh stack --floors 8        # with an option
#   ./comp.sh wall-bay --mount wall   # against a mocked wall instead of ground
#   ./comp.sh room --ladder           # the whole composition ladder at once
#
# Options after the name are passed to the component as JSON, so anything the
# component takes works: --size 9 --thickness 0.4 --floors 6.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME="${VIBE_LAND_ROOT:-/root/workspace/vibe-land-2}"

NAME="${1:?usage: comp.sh <component> [--opt value ...] [--ladder]}"
shift

LADDER=0
OPTS="{}"
while [ $# -gt 0 ]; do
  case "$1" in
    --ladder) LADDER=1; shift ;;
    --mount) OPTS=$(node -e "const o=$OPTS;o.mountKind='$2';console.log(JSON.stringify(o))"); shift 2 ;;
    --*)
      key="${1#--}"
      OPTS=$(node -e "const o=$OPTS;const v='$2';o['$key']=isNaN(+v)?v:+v;console.log(JSON.stringify(o))")
      shift 2 ;;
    *) echo "unexpected argument '$1'" >&2; exit 1 ;;
  esac
done

if [ "$LADDER" = 1 ]; then
  PACKS="comp-wall-bay comp-room comp-storey comp-stack"
else
  PACKS="comp-${NAME}"
fi

# Ad-hoc options mean an ad-hoc pack, written under the same name so the audit
# can find it. Without options this is just the registered build.
if [ "$OPTS" != "{}" ]; then
  node -e "
    import('./components.mjs').then(async (m) => {
      const { writeFile } = await import('node:fs/promises');
      const { pack } = m.standalone('$NAME', $OPTS);
      const json = JSON.stringify(pack);
      await writeFile('$GAME/destruction/assets/scenes/comp-$NAME.json', json);
      console.log('  built comp-$NAME with $OPTS');
    }).catch((e) => { console.error(e.message); process.exit(1); });
  " || exit 1
else
  (cd "$HERE" && node build.mjs $PACKS --emit-vibe-land "$GAME" 2>&1 \
    | grep -E "===|FAIL|WARN|=> " ) || exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}/usr/local/cuda/lib64:/root/PhysX/physx/install/linux-clang/PhysX/bin/linux.x86_64/release"
LIST=$(echo "$PACKS" | tr ' ' ',')
cd "$GAME"
AUDIT_MAX_SECS="${AUDIT_MAX_SECS:-60}" AUDIT_PACKS="$LIST" \
  cargo test -p vibe-land-destruction --features cuda-stress \
  --test structural_audit --release -- --ignored --nocapture 2>&1 \
  | grep -E "^=== |STABLE|^  peak |joint classes"
