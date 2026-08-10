mod diagnostics;
mod renderer;
mod state;
mod telemetry;

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Context, Result};
use chrono::Local;
use clap::{Args, Parser, Subcommand};
use telemetry::GpuTelemetry;

#[derive(Parser)]
#[command(
    name = "blast-mini-city-recorder",
    about = "Headless recording for Blast PhysX GPU Mini-City using wgpu/Vulkan and NVENC"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the configurable C++ simulation, capture TWSTATE1, render, and encode.
    Record(RecordArgs),
    /// Render an existing TWSTATE1 state stream without running simulation.
    Render(RenderArgs),
}

#[derive(Args)]
struct RecordArgs {
    /// Blast PhysX GPU Mini-City simulation executable.
    #[arg(long)]
    sim_bin: Option<PathBuf>,

    /// Square city size passed to the simulation as `--grid`.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
    grid: u32,

    /// Captured simulation duration in seconds.
    #[arg(long, default_value_t = 12.0)]
    duration: f32,

    /// Optional pre-capture settling period passed to the simulation.
    #[arg(long, default_value_t = 3.0)]
    settle: f32,

    /// Transform snapshots per second.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..=240))]
    snapshot_fps: u32,

    /// Width of each camera pane; the encoded mosaic is twice this width.
    #[arg(long, default_value_t = 960, value_parser = clap::value_parser!(u32).range(1..))]
    pane_width: u32,

    /// Height of each camera pane; the encoded mosaic is twice this height.
    #[arg(long, default_value_t = 540, value_parser = clap::value_parser!(u32).range(1..))]
    pane_height: u32,

    /// Require confirmed PhysX GPU activation; never permit CPU fallback.
    #[arg(long)]
    require_gpu: bool,

    /// Enable the hybrid resident CUDA stress backend for large graphs.
    #[arg(long)]
    gpu_stress: bool,

    /// Require projectile impacts and broad partial fracture with no near-total shatter.
    #[arg(long)]
    require_partial_destruction: bool,

    /// Encoded MP4 path. Defaults to a timestamped file under /root/recordings.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Persistent TWSTATE1 path. Without this, a temporary sidecar is removed.
    #[arg(long)]
    state: Option<PathBuf>,

    /// Keep an automatically named state stream after a successful render.
    #[arg(long)]
    keep_state: bool,

    /// Simulation metadata JSON path.
    #[arg(long)]
    metadata: Option<PathBuf>,

    /// Simulation telemetry sidecar path.
    #[arg(long)]
    sim_telemetry: Option<PathBuf>,

    /// Per-step simulation CSV used for diagnostics and the video overlay.
    #[arg(long)]
    frame_telemetry: Option<PathBuf>,

    /// Replace the fourth pane with the projectile chase camera.
    #[arg(long)]
    chase_projectile: bool,

    /// Additional argument passed verbatim to the simulation; may be repeated.
    #[arg(long = "sim-arg", allow_hyphen_values = true)]
    sim_args: Vec<OsString>,
}

#[derive(Args)]
struct RenderArgs {
    /// Existing TWSTATE1 state stream.
    #[arg(long)]
    state: PathBuf,

    /// Encoded MP4 path.
    #[arg(long)]
    output: PathBuf,

    /// Replace the fourth pane with a close projectile chase camera.
    #[arg(long)]
    chase_projectile: bool,

    /// Per-step simulation CSV to render as a synchronized diagnostics overlay.
    #[arg(long)]
    frame_telemetry: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Record(args) => record(args),
        Commands::Render(args) => render(&args),
    }
}

fn record(args: RecordArgs) -> Result<()> {
    if !args.duration.is_finite() || args.duration <= 0.0 {
        bail!("duration must be finite and positive");
    }
    if !args.settle.is_finite() || args.settle < 0.0 {
        bail!("settle must be finite and non-negative");
    }

    let stamp = Local::now().format("%Y%m%d-%H%M%S");
    let output = args.output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "/root/recordings/blast-physx-gpu-mini-city-{}x{}-{stamp}.mp4",
            args.grid, args.grid
        ))
    });
    ensure_parent(&output)?;

    let explicit_state = args.state.is_some();
    let state_path = args
        .state
        .unwrap_or_else(|| phase_path(&output, "tmp.twstate"));
    let metadata_path = args
        .metadata
        .unwrap_or_else(|| phase_path(&output, "metadata.json"));
    let simulation_telemetry_path = args
        .sim_telemetry
        .unwrap_or_else(|| phase_path(&output, "simulation.telemetry.json"));
    let frame_telemetry_path = args
        .frame_telemetry
        .unwrap_or_else(|| phase_path(&output, "simulation.frames.csv"));
    for path in [
        &state_path,
        &metadata_path,
        &simulation_telemetry_path,
        &frame_telemetry_path,
    ] {
        ensure_parent(path)?;
    }

    let sim_bin = args.sim_bin.unwrap_or_else(default_sim_bin);
    if !sim_bin.is_file() {
        bail!(
            "simulation executable not found at {}; build it or pass --sim-bin",
            sim_bin.display()
        );
    }

    println!("simulation={}", sim_bin.display());
    println!("state={}", state_path.display());
    println!("metadata={}", metadata_path.display());
    println!(
        "simulation_telemetry={}",
        simulation_telemetry_path.display()
    );
    println!("frame_telemetry={}", frame_telemetry_path.display());
    println!("output={}", output.display());

    let simulation_gpu_csv = phase_path(&output, "simulation.gpu.csv");
    let simulation_gpu_summary = phase_path(&output, "simulation.gpu-summary.txt");
    let telemetry = GpuTelemetry::start(simulation_gpu_csv, simulation_gpu_summary)?;

    let mut simulation = Command::new(&sim_bin);
    simulation
        .arg("--state")
        .arg(&state_path)
        .arg("--metadata")
        .arg(&metadata_path)
        .arg("--telemetry")
        .arg(&simulation_telemetry_path)
        .arg("--frame-telemetry")
        .arg(&frame_telemetry_path)
        .arg("--grid")
        .arg(args.grid.to_string())
        .arg("--duration")
        .arg(args.duration.to_string())
        .arg("--settle")
        .arg(args.settle.to_string())
        .arg("--snapshot-fps")
        .arg(args.snapshot_fps.to_string())
        .arg("--pane-width")
        .arg(args.pane_width.to_string())
        .arg("--pane-height")
        .arg(args.pane_height.to_string())
        .args(&args.sim_args)
        .env_remove("DISPLAY");
    if args.require_gpu {
        simulation.arg("--require-gpu");
    }
    if args.gpu_stress {
        simulation.arg("--gpu-stress");
    }
    if args.require_partial_destruction {
        simulation.arg("--require-partial-destruction");
    }
    let status = simulation
        .status()
        .with_context(|| format!("run simulation {}", sim_bin.display()))?;
    telemetry.stop()?;
    if !status.success() {
        bail!("simulation exited with {status}");
    }
    if !state_path.is_file() {
        bail!(
            "simulation succeeded but did not write state {}",
            state_path.display()
        );
    }

    render_with_telemetry(
        &state_path,
        &output,
        args.chase_projectile,
        Some(&frame_telemetry_path),
    )?;
    verify_video(&output)?;

    if !explicit_state && !args.keep_state {
        fs::remove_file(&state_path)
            .with_context(|| format!("remove temporary state {}", state_path.display()))?;
        println!("removed temporary state={}", state_path.display());
    } else {
        println!("kept state={}", state_path.display());
    }
    Ok(())
}

fn render(args: &RenderArgs) -> Result<()> {
    ensure_parent(&args.output)?;
    render_with_telemetry(
        &args.state,
        &args.output,
        args.chase_projectile,
        args.frame_telemetry.as_deref(),
    )?;
    verify_video(&args.output)
}

fn render_with_telemetry(
    state_path: &Path,
    output: &Path,
    chase_projectile: bool,
    simulation_frames: Option<&Path>,
) -> Result<()> {
    let render_csv = phase_path(output, "render.gpu.csv");
    let render_summary = phase_path(output, "render.gpu-summary.txt");
    let render_frames = phase_path(output, "render.frames.csv");
    let render_frame_summary = phase_path(output, "render.summary.json");
    let telemetry = GpuTelemetry::start(render_csv, render_summary)?;
    let result = renderer::render_recording(
        state_path,
        output,
        chase_projectile,
        simulation_frames,
        &render_frames,
        &render_frame_summary,
    );
    telemetry.stop()?;
    result
}

fn verify_video(path: &Path) -> Result<()> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .with_context(|| format!("probe {}", path.display()))?;
    if !probe.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&probe.stderr).trim()
        );
    }
    println!(
        "verified video={}\n{}",
        path.display(),
        String::from_utf8_lossy(&probe.stdout).trim()
    );
    Ok(())
}

fn default_sim_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../build/blast_stress_demo")
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
    }
    Ok(())
}

fn phase_path(output: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}", without_extension(output), suffix))
}

fn without_extension(path: &Path) -> String {
    path.with_extension("").to_string_lossy().into_owned()
}
