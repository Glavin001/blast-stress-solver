#!/usr/bin/env bash
#
# Reproduce every published chunk-crushing A/B comparison, end to end.
#
# Each scenario runs the SAME simulation twice -- once with crushing enabled,
# once with --no-crush -- and composes a labelled side-by-side. Nothing else
# differs between the two runs of a pair, which is what makes them evidence
# rather than illustration.
#
#   wall-mid     an ordinary hit: bond fracture cracks the wall internally but
#                every block stays put; crushing is the only visible damage
#   wall-punch   a hard hit: bond-only blows a plug through at ~36 m/s, while
#                the crushable wall absorbs ~5 MJ and stops the ball
#   building     the calibrated reference building, dust bursts visible
#   scale-6k     144 buildings / ~6k chunks, GPU, production resim=1
#   scale-16k    400 buildings / ~16.5k chunks, the largest the CLI allows
#
# Usage:
#   ./record_crush_videos.sh [output-dir] [scenario ...]
#   ./record_crush_videos.sh /tmp/out wall-punch      # just one
#
# Requires a built blast_stress_demo and recorder (see README.md), plus ffmpeg.
# The scale scenarios need a CUDA GPU; the wall and building ones run on CPU.
set -euo pipefail

demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${demo_dir}/../.." && pwd)"
out_dir="${1:-/root/recordings/chunk-crush}"
shift || true
scenarios=("$@")
if [ ${#scenarios[@]} -eq 0 ]; then
  scenarios=(wall-mid wall-punch building scale-6k scale-16k)
fi

sim="${demo_dir}/build/blast_stress_demo"
recorder="${demo_dir}/recorder/target/release/blast-mini-city-recorder"
assets="${repo_root}/blast/blast-stress-solver/assets/reference"
wall_pack="${assets}/crush-wall.json"
building_pack="${assets}/reference-building-crush.json"
font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

for tool in "${sim}" "${recorder}"; do
  [ -x "${tool}" ] || { echo "missing ${tool} -- see demos/blast-stress-demo/README.md" >&2; exit 1; }
done
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
mkdir -p "${out_dir}"

# Run one side of a pair. $1 stem, $2 scene, $3.. sim args.
run_side() {
  local stem="$1" scene="$2"; shift 2
  echo "  sim  ${stem}"
  "${sim}" --scene "${scene}" --snapshot-fps 30 \
    --state "${out_dir}/${stem}.twstate" \
    --frame-telemetry "${out_dir}/${stem}.frames.csv" \
    --metadata "${out_dir}/${stem}.metadata.json" "$@" \
    | grep -E "chunk crushing|mini-city finished" || true
}

render_side() {
  local stem="$1"; shift
  echo "  render ${stem}"
  "${recorder}" render \
    --state "${out_dir}/${stem}.twstate" \
    --frame-telemetry "${out_dir}/${stem}.frames.csv" \
    --output "${out_dir}/${stem}.mp4" "$@" >/dev/null
}

# Side-by-side, crush-off left / crush-on right. The HUD (top 200 px) is
# cropped away because halving the width makes its text unreadable -- read the
# individual full-resolution renders for the counters.
compose() {
  local out="$1" left="$2" right="$3" left_label="$4" right_label="$5"
  echo "  compose ${out}"
  ffmpeg -hide_banner -loglevel error -i "${out_dir}/${left}.mp4" -i "${out_dir}/${right}.mp4" \
    -filter_complex "\
[0:v]crop=1920:880:0:200,scale=960:440,pad=960:490:0:50:color=0x1b1f24,\
drawtext=fontfile=${font}:text='${left_label}':fontcolor=white:fontsize=19:x=(w-tw)/2:y=16[l];\
[1:v]crop=1920:880:0:200,scale=960:440,pad=960:490:0:50:color=0x1b1f24,\
drawtext=fontfile=${font}:text='${right_label}':fontcolor=0xffd070:fontsize=19:x=(w-tw)/2:y=16[r];\
[l][r]hstack=inputs=2[v]" \
    -map "[v]" -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p "${out_dir}/${out}.mp4" -y
}

# Production config everywhere: resim-passes 1. The wall scenarios use a single
# projectile wave so the exit-velocity readout in frames.csv is uncontaminated
# by later launches still travelling at muzzle speed.
wall_common=(--physics cpu --grid 1 --duration 8 --settle 2 --projectile-waves 1 --resim-passes 1)
gpu_common=(--physics gpu --require-gpu --gpu-stress --gpu-stress-min-bonds 540 \
            --duration 12 --settle 2 --projectile-mass-scale 2 --resim-passes 1)

for scenario in "${scenarios[@]}"; do
  echo "== ${scenario}"
  case "${scenario}" in
    wall-mid)
      run_side wall-mid-off "${wall_pack}" "${wall_common[@]}" --projectile-speed-scale 1.1 --no-crush
      run_side wall-mid-on  "${wall_pack}" "${wall_common[@]}" --projectile-speed-scale 1.1
      render_side wall-mid-off; render_side wall-mid-on
      compose wall-resim1-mid wall-mid-off wall-mid-on \
        "CRUSH OFF - cracks internally, looks untouched" \
        "CRUSH ON - impact zone bitten out as dust"
      ;;
    wall-punch)
      run_side wall-punch-off "${wall_pack}" "${wall_common[@]}" --projectile-speed-scale 1.4 --no-crush
      run_side wall-punch-on  "${wall_pack}" "${wall_common[@]}" --projectile-speed-scale 1.4
      render_side wall-punch-off; render_side wall-punch-on
      compose wall-resim1-punch wall-punch-off wall-punch-on \
        "CRUSH OFF - plug blown through, ball exits at 36 m per s" \
        "CRUSH ON - wall EATS the ball: 4.5 t to dust, ball stopped"
      ;;
    building)
      bc=(--physics cpu --grid 1 --duration 10 --settle 2 --projectile-mass-scale 2 --resim-passes 1)
      run_side building-off "${building_pack}" "${bc[@]}" --no-crush
      run_side building-on  "${building_pack}" "${bc[@]}"
      render_side building-off --chase-projectile; render_side building-on --chase-projectile
      compose building-comparison building-off building-on \
        "CRUSH OFF - bond fracture only" \
        "CRUSH ON - crushed chunks burst into dust"
      ;;
    scale-6k)
      run_side scale-6k-off "${building_pack}" "${gpu_common[@]}" --grid 12 --no-crush
      run_side scale-6k-on  "${building_pack}" "${gpu_common[@]}" --grid 12
      render_side scale-6k-off --chase-projectile; render_side scale-6k-on --chase-projectile
      compose scale-6k-comparison scale-6k-off scale-6k-on \
        "6k chunks - CRUSH OFF" "6k chunks - CRUSH ON"
      ;;
    scale-16k)
      # grid 20 is the CLI maximum and the largest scene reachable with the
      # calibrated reference building. Do NOT reach for a taller generated
      # tower instead: scaling FLOORS without re-sizing COL drops every joint
      # class below its safety-factor band and the whole city cascades on
      # contact (see the authoring skill's symptom table).
      run_side scale-16k-off "${building_pack}" "${gpu_common[@]}" --grid 20 --no-crush
      run_side scale-16k-on  "${building_pack}" "${gpu_common[@]}" --grid 20
      render_side scale-16k-off --chase-projectile; render_side scale-16k-on --chase-projectile
      compose scale-16k-comparison scale-16k-off scale-16k-on \
        "16k chunks, calibrated - CRUSH OFF" \
        "16k chunks, calibrated - CRUSH ON"
      ;;
    *) echo "unknown scenario: ${scenario}" >&2; exit 2 ;;
  esac
done

echo
echo "wrote to ${out_dir}:"
ls -1 "${out_dir}"/*.mp4 2>/dev/null || true
