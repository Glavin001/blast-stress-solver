use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{bail, Context, Result};
use font8x8::{UnicodeFonts, BASIC_FONTS};

#[derive(Clone, Debug, Default)]
pub struct SimulationFrame {
    pub step: u32,
    pub simulation_seconds: f64,
    pub physics_step_ms: f64,
    pub contact_callback_ms: f64,
    pub contact_processing_ms: f64,
    pub gravity_ms: f64,
    pub stress_solve_ms: f64,
    pub gpu_stress_solve_ms: f64,
    pub gpu_stress_h2d_bytes: u64,
    pub gpu_stress_d2h_bytes: u64,
    pub fracture_topology_ms: f64,
    pub adapter_tick_ms: f64,
    pub mapping_validation_ms: f64,
    pub state_export_ms: f64,
    pub frame_host_ms: f64,
    pub realtime_factor: f64,
    pub bodies: u64,
    pub awake_bodies: u64,
    pub solver_islands: u64,
    pub solver_islands_skipped: u64,
    pub overstressed_bonds: u64,
    pub contacts_frame: u64,
    pub contacts_total: u64,
    pub contacts_dropped_total: u64,
    pub projectile_impacts_frame: u64,
    pub projectile_impacts_total: u64,
    pub projectile_impulse_frame: f64,
    pub projectile_impulse_total: f64,
    pub splits_frame: u64,
    pub splits_total: u64,
    pub shapes_migrated_frame: u64,
    pub shapes_migrated_total: u64,
    pub sleeping_actors_skipped: u64,
    pub max_position_drift: f64,
    pub max_point_velocity_drift: f64,
    pub budget_miss_frames: u64,
}

pub struct SimulationTelemetry {
    frames: Vec<SimulationFrame>,
}

impl SimulationTelemetry {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("open simulation frame telemetry {}", path.display()))?;
        let mut lines = BufReader::new(file).lines();
        let header = lines
            .next()
            .context("simulation frame telemetry is empty")??;
        let columns: HashMap<_, _> = header
            .split(',')
            .enumerate()
            .map(|(index, name)| (name.trim().to_owned(), index))
            .collect();
        for required in [
            "step",
            "simulation_seconds",
            "physics_step_ms",
            "stress_solve_ms",
            "frame_host_ms",
            "bodies",
            "awake_bodies",
            "splits_total",
            "contacts_total",
        ] {
            if !columns.contains_key(required) {
                bail!("simulation frame telemetry is missing column {required}");
            }
        }

        let mut frames = Vec::new();
        let mut budget_misses = 0_u64;
        for (line_index, line) in lines.enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            let parse_f64 = |name: &str| -> Result<f64> {
                let index = *columns
                    .get(name)
                    .with_context(|| format!("missing telemetry column {name}"))?;
                fields
                    .get(index)
                    .with_context(|| format!("telemetry row {} is truncated", line_index + 2))?
                    .parse::<f64>()
                    .with_context(|| format!("invalid {name} on telemetry row {}", line_index + 2))
            };
            let parse_u64 = |name: &str| -> Result<u64> { Ok(parse_f64(name)? as u64) };
            let parse_optional_f64 = |name: &str| -> Result<f64> {
                if columns.contains_key(name) {
                    parse_f64(name)
                } else {
                    Ok(0.0)
                }
            };
            let frame_host_ms = parse_f64("frame_host_ms")?;
            if frame_host_ms > 1000.0 / 60.0 {
                budget_misses += 1;
            }
            frames.push(SimulationFrame {
                step: parse_u64("step")? as u32,
                simulation_seconds: parse_f64("simulation_seconds")?,
                physics_step_ms: parse_f64("physics_step_ms")?,
                contact_callback_ms: parse_f64("contact_callback_ms")?,
                contact_processing_ms: parse_f64("contact_processing_ms")?,
                gravity_ms: parse_f64("gravity_ms")?,
                stress_solve_ms: parse_f64("stress_solve_ms")?,
                gpu_stress_solve_ms: parse_optional_f64("gpu_stress_solve_ms")?,
                gpu_stress_h2d_bytes: parse_optional_f64("gpu_stress_h2d_bytes")? as u64,
                gpu_stress_d2h_bytes: parse_optional_f64("gpu_stress_d2h_bytes")? as u64,
                fracture_topology_ms: parse_f64("fracture_topology_ms")?,
                adapter_tick_ms: parse_f64("adapter_tick_ms")?,
                mapping_validation_ms: parse_f64("mapping_validation_ms")?,
                state_export_ms: parse_f64("state_export_ms")?,
                frame_host_ms,
                realtime_factor: parse_f64("realtime_factor")?,
                bodies: parse_u64("bodies")?,
                awake_bodies: parse_u64("awake_bodies")?,
                solver_islands: parse_u64("solver_islands")?,
                solver_islands_skipped: parse_u64("solver_islands_skipped")?,
                overstressed_bonds: parse_u64("overstressed_bonds")?,
                contacts_frame: parse_u64("contacts_frame")?,
                contacts_total: parse_u64("contacts_total")?,
                contacts_dropped_total: parse_u64("contacts_dropped_total")?,
                projectile_impacts_frame: parse_optional_f64("projectile_impacts_frame")? as u64,
                projectile_impacts_total: parse_optional_f64("projectile_impacts_total")? as u64,
                projectile_impulse_frame: parse_optional_f64("projectile_impulse_frame")?,
                projectile_impulse_total: parse_optional_f64("projectile_impulse_total")?,
                splits_frame: parse_u64("splits_frame")?,
                splits_total: parse_u64("splits_total")?,
                shapes_migrated_frame: parse_u64("shapes_migrated_frame")?,
                shapes_migrated_total: parse_u64("shapes_migrated_total")?,
                sleeping_actors_skipped: parse_u64("sleeping_actors_skipped")?,
                max_position_drift: parse_f64("max_position_drift")?,
                max_point_velocity_drift: parse_f64("max_point_velocity_drift")?,
                budget_miss_frames: budget_misses,
            });
        }
        if frames.is_empty() {
            bail!("simulation frame telemetry contains no samples");
        }
        Ok(Self { frames })
    }

    pub fn for_output_frame(&self, output_frame: u32, output_fps: u32) -> Option<&SimulationFrame> {
        if output_frame == 0 {
            return self.frames.first();
        }
        let physics_step = (output_frame as u64 * 60 / output_fps as u64).saturating_sub(1);
        self.frames.get(physics_step as usize)
    }
}

pub fn simulation_overlay(frame: &SimulationFrame) -> Vec<String> {
    let simulated_fps = 1000.0 / frame.frame_host_ms.max(1.0e-6);
    vec![
        format!(
            "PHYSX GPU  t={:6.2}s  step={}  host={:5.2}ms ({:5.1} fps, {:4.2}x realtime)",
            frame.simulation_seconds,
            frame.step,
            frame.frame_host_ms,
            simulated_fps,
            frame.realtime_factor
        ),
        format!(
            "physics(wait)={:5.2}ms  adapter={:5.2}ms  export={:5.2}ms",
            frame.physics_step_ms,
            frame.adapter_tick_ms,
            frame.state_export_ms
        ),
        format!(
            "stress: contacts={:5.2}ms  gravity={:5.2}ms  total={:5.2}ms  gpu={:5.2}ms  fracture={:5.2}ms  validation={:5.2}ms",
            frame.contact_processing_ms + frame.contact_callback_ms,
            frame.gravity_ms,
            frame.stress_solve_ms,
            frame.gpu_stress_solve_ms,
            frame.fracture_topology_ms,
            frame.mapping_validation_ms
        ),
        format!(
            "gpu stress traffic: H2D={} B  D2H={} B",
            frame.gpu_stress_h2d_bytes, frame.gpu_stress_d2h_bytes
        ),
        format!(
            "bodies={}  awake={}  islands={} (skipped {})  overstressed={}  sleep-skips={}",
            frame.bodies,
            frame.awake_bodies,
            frame.solver_islands,
            frame.solver_islands_skipped,
            frame.overstressed_bonds,
            frame.sleeping_actors_skipped
        ),
        format!(
            "splits={} (+{})  migrated={} (+{})  contacts={} (+{})  projectile-impacts={} (+{})",
            frame.splits_total,
            frame.splits_frame,
            frame.shapes_migrated_total,
            frame.shapes_migrated_frame,
            frame.contacts_total,
            frame.contacts_frame,
            frame.projectile_impacts_total,
            frame.projectile_impacts_frame
        ),
        format!(
            "impact impulse={:.2e} (+{:.2e})  dropped={}  continuity: pos={:.2e} vel={:.2e}",
            frame.projectile_impulse_total,
            frame.projectile_impulse_frame,
            frame.contacts_dropped_total,
            frame.max_position_drift,
            frame.max_point_velocity_drift
        ),
        format!(
            "60Hz budget misses={}",
            frame.budget_miss_frames,
        ),
    ]
}

pub fn draw_overlay(frame: &mut [u8], width: u32, height: u32, lines: &[String], scale: u32) {
    if lines.is_empty() || width == 0 || height == 0 {
        return;
    }
    let margin = 12_u32;
    let line_height = 9 * scale;
    let panel_height = (margin * 2 + line_height * lines.len() as u32).min(height);
    let panel_width = lines
        .iter()
        .map(|line| line.chars().count() as u32 * 8 * scale)
        .max()
        .unwrap_or(0)
        .saturating_add(margin * 2)
        .min(width);
    blend_rect(
        frame,
        width,
        height,
        0,
        0,
        panel_width,
        panel_height,
        [8, 12, 18, 205],
    );

    for (line_index, line) in lines.iter().enumerate() {
        let y = margin + line_index as u32 * line_height;
        draw_text(
            frame,
            width,
            height,
            margin,
            y,
            line,
            scale,
            [235, 244, 250, 255],
        );
    }
}

fn blend_rect(
    frame: &mut [u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    rect_width: u32,
    rect_height: u32,
    color: [u8; 4],
) {
    let alpha = color[3] as u32;
    for py in y..(y + rect_height).min(height) {
        for px in x..(x + rect_width).min(width) {
            let offset = ((py * width + px) * 4) as usize;
            for channel in 0..3 {
                frame[offset + channel] = (((frame[offset + channel] as u32) * (255 - alpha)
                    + color[channel] as u32 * alpha)
                    / 255) as u8;
            }
            frame[offset + 3] = 255;
        }
    }
}

fn draw_text(
    frame: &mut [u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    text: &str,
    scale: u32,
    color: [u8; 4],
) {
    let mut cursor_x = x;
    for character in text.chars() {
        let Some(glyph) = BASIC_FONTS.get(character) else {
            cursor_x += 8 * scale;
            continue;
        };
        for (row, bits) in glyph.iter().enumerate() {
            for column in 0..8_u32 {
                if bits & (1 << column) == 0 {
                    continue;
                }
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = cursor_x + column * scale + sx;
                        let py = y + row as u32 * scale + sy;
                        if px >= width || py >= height {
                            continue;
                        }
                        let offset = ((py * width + px) * 4) as usize;
                        frame[offset..offset + 4].copy_from_slice(&color);
                    }
                }
            }
        }
        cursor_x += 8 * scale;
        if cursor_x >= width {
            break;
        }
    }
}
