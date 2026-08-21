use std::{
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroU64,
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    diagnostics::{draw_overlay, simulation_overlay, SimulationFrame, SimulationTelemetry},
    state::{Actor, Camera, Shape, StateReader, Transform},
};

const STAGING_BUFFER_COUNT: usize = 3;
const INITIAL_INSTANCE_CAPACITY: usize = 128_000;
const PROJECTILE_PART: u8 = 5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    2 => Float32x4,
    3 => Float32x4,
    4 => Float32x4,
    5 => Float32x4,
    6 => Float32x4
];

struct Mesh {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
}

struct InstanceBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
}

struct ReadbackLayout {
    padded_bytes_per_row: u32,
    unpadded_bytes_per_row: u32,
    width: u32,
    height: u32,
    pane_width: u32,
    pane_height: u32,
}

struct ImpactObserverCamera {
    camera: Option<Camera>,
}

#[derive(Clone, Copy)]
struct SceneBounds {
    minimum: Vec3,
    maximum: Vec3,
}

struct OverviewOrbitCamera {
    bounds: SceneBounds,
    direction: Vec3,
    aspect: f32,
    fov_degrees: f32,
}

impl OverviewOrbitCamera {
    fn new(camera: Camera, bounds: SceneBounds, aspect: f32) -> Self {
        Self {
            bounds,
            direction: camera.direction.normalize(),
            aspect,
            fov_degrees: camera.fov_degrees,
        }
    }

    fn at_time(&self, seconds: f32) -> Camera {
        let direction = Quat::from_rotation_y(seconds * 0.08) * self.direction;
        fit_camera_to_bounds(
            Camera {
                eye: Vec3::ZERO,
                direction,
                fov_degrees: self.fov_degrees,
            },
            self.bounds,
            self.aspect,
            1.0,
        )
    }
}

#[derive(Clone, Copy)]
struct RenderSubmitInfo {
    cpu_submit_ms: f64,
    boxes: usize,
    spheres: usize,
}

#[derive(Clone, Copy)]
struct OutputTiming {
    readback_ms: f64,
    overlay_ms: f64,
    encode_pipe_ms: f64,
    total_host_ms: f64,
}

impl Default for ImpactObserverCamera {
    fn default() -> Self {
        Self { camera: None }
    }
}

impl ImpactObserverCamera {
    fn update(&mut self, actors: &[Actor], fallback: Camera) -> Camera {
        if let Some(camera) = self.camera {
            return camera;
        }

        let mut minimum = Vec3::splat(f32::INFINITY);
        let mut maximum = Vec3::splat(f32::NEG_INFINITY);
        let mut found = false;
        for actor in actors {
            if !actor.visible
                || actor.part == PROJECTILE_PART
                || actor.pose.position.y < -200.0
                || actor.pose.position.abs().max_element() >= 5_000.0
            {
                continue;
            }
            minimum = minimum.min(actor.pose.position);
            maximum = maximum.max(actor.pose.position);
            found = true;
        }
        if !found {
            return fallback;
        }

        // Lock onto the front-center buildings before impacts begin. Keeping
        // this pane in one world-space reference frame makes fragment motion
        // distinguishable from camera motion.
        let target = Vec3::new((minimum.x + maximum.x) * 0.5, 6.0, minimum.z + 4.0);
        let eye = target + Vec3::new(15.0, 9.0, -22.0);
        let camera = Camera {
            eye,
            direction: (target - eye).normalize(),
            fov_degrees: 50.0,
        };
        self.camera = Some(camera);
        camera
    }
}

fn scene_bounds(actors: &[Actor]) -> Option<SceneBounds> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    let mut found = false;
    for actor in actors {
        if !actor.visible
            || actor.part == PROJECTILE_PART
            || actor.pose.position.y < -200.0
            || actor.pose.position.abs().max_element() >= 5_000.0
        {
            continue;
        }
        let actor_matrix = transform_matrix(actor.pose);
        for shape in &actor.shapes {
            let (center, extents) = match shape {
                Shape::Box {
                    half_extents,
                    local,
                } => {
                    let world = actor_matrix * transform_matrix(*local);
                    let extents = world.transform_vector3(Vec3::X * half_extents.x).abs()
                        + world.transform_vector3(Vec3::Y * half_extents.y).abs()
                        + world.transform_vector3(Vec3::Z * half_extents.z).abs();
                    (world.transform_point3(Vec3::ZERO), extents)
                }
                Shape::Sphere { radius, local } => {
                    let world = actor_matrix * transform_matrix(*local);
                    (world.transform_point3(Vec3::ZERO), Vec3::splat(*radius))
                }
                Shape::Mesh {
                    positions, local, ..
                } => {
                    let world = actor_matrix * transform_matrix(*local);
                    if positions.is_empty() {
                        (world.transform_point3(Vec3::ZERO), Vec3::ZERO)
                    } else {
                        let mut mesh_min = Vec3::splat(f32::INFINITY);
                        let mut mesh_max = Vec3::splat(f32::NEG_INFINITY);
                        for position in positions {
                            let world_position = world.transform_point3(*position);
                            mesh_min = mesh_min.min(world_position);
                            mesh_max = mesh_max.max(world_position);
                        }
                        ((mesh_min + mesh_max) * 0.5, (mesh_max - mesh_min) * 0.5)
                    }
                }
            };
            minimum = minimum.min(center - extents);
            maximum = maximum.max(center + extents);
            found = true;
        }
    }
    found.then_some(SceneBounds { minimum, maximum })
}

fn fit_camera_to_bounds(
    template: Camera,
    bounds: SceneBounds,
    aspect: f32,
    padding: f32,
) -> Camera {
    let direction = template.direction.normalize_or_zero();
    let direction = if direction.length_squared() > 0.0 {
        direction
    } else {
        Vec3::NEG_Z
    };
    let reference_up = if direction.dot(Vec3::Y).abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = direction.cross(reference_up).normalize();
    let up = right.cross(direction).normalize();
    let center = (bounds.minimum + bounds.maximum) * 0.5;
    let tangent_y = (template.fov_degrees.to_radians() * 0.5).tan();
    let tangent_x = tangent_y * aspect;
    let mut distance = 1.0_f32;
    for x in [bounds.minimum.x, bounds.maximum.x] {
        for y in [bounds.minimum.y, bounds.maximum.y] {
            for z in [bounds.minimum.z, bounds.maximum.z] {
                let relative = Vec3::new(x, y, z) - center;
                let forward = relative.dot(direction);
                distance = distance
                    .max(relative.dot(right).abs() / tangent_x - forward)
                    .max(relative.dot(up).abs() / tangent_y - forward)
                    .max(1.0 - forward);
            }
        }
    }
    distance *= padding.max(0.8);
    Camera {
        eye: center - direction * distance,
        direction,
        fov_degrees: template.fov_degrees,
    }
}

impl InstanceBuffer {
    fn new(device: &wgpu::Device, label: &str, capacity: usize) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity,
        }
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        instances: &[InstanceRaw],
    ) {
        if instances.len() > self.capacity {
            self.capacity = instances.len().next_power_of_two();
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (self.capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(instances));
        }
    }
}

pub fn render_recording(
    state_path: &Path,
    output_path: &Path,
    chase_projectile: bool,
    sleep_tint: bool,
    simulation_telemetry_path: Option<&Path>,
    render_frames_path: &Path,
    render_summary_path: &Path,
) -> Result<()> {
    pollster::block_on(render_recording_async(
        state_path,
        output_path,
        chase_projectile,
        sleep_tint,
        simulation_telemetry_path,
        render_frames_path,
        render_summary_path,
    ))
}

/// Whether the local FFmpeg advertises an encoder by name.
fn ffmpeg_has_encoder(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).contains(name))
        .unwrap_or(false)
}

async fn render_recording_async(
    state_path: &Path,
    output_path: &Path,
    chase_projectile: bool,
    sleep_tint: bool,
    simulation_telemetry_path: Option<&Path>,
    render_frames_path: &Path,
    render_summary_path: &Path,
) -> Result<()> {
    let mut state = StateReader::open(state_path)?;
    let simulation_telemetry = simulation_telemetry_path
        .map(SimulationTelemetry::load)
        .transpose()?;
    let width = state.header.pane_width * 2;
    let height = state.header.pane_height * 2;
    let fps = state.header.fps;
    println!(
        "state: scene_instances={} duration={:.3}s settle={:.3}s frames={} mosaic={}x{}",
        state.header.building_count,
        state.header.duration_seconds,
        state.header.settle_seconds,
        state.header.frame_count,
        width,
        height
    );

    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .context("request headless Vulkan adapter")?;
    let adapter_info = adapter.get_info();
    if adapter_info.device_type != wgpu::DeviceType::DiscreteGpu {
        bail!(
            "headless renderer selected non-discrete adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );
    }
    println!(
        "renderer adapter: {} ({:?}, {:?})",
        adapter_info.name, adapter_info.backend, adapter_info.device_type
    );
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("blast-mini-city-recorder-device"),
            ..Default::default()
        })
        .await
        .context("create Vulkan device")?;

    let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("four-camera-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&Default::default());
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("four-camera-depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&Default::default());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blast-mini-city-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<CameraUniform>() as u64),
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blast-mini-city-pipeline-layout"),
        bind_group_layouts: &[Some(&camera_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blast-mini-city-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &VERTEX_ATTRIBUTES,
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceRaw>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &INSTANCE_ATTRIBUTES,
                }),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let cube = create_cube_mesh(&device);
    let sphere = create_sphere_mesh(&device, 18, 12);
    let mut box_buffer = InstanceBuffer::new(&device, "box-instances", INITIAL_INSTANCE_CAPACITY);
    let mut sphere_buffer = InstanceBuffer::new(&device, "sphere-instances", 32);
    let (camera_buffers, camera_groups) = create_camera_bind_groups(
        &device,
        &camera_layout,
        &state.header.cameras,
        state.header.pane_width as f32 / state.header.pane_height as f32,
    );
    let mut impact_camera = chase_projectile.then(ImpactObserverCamera::default);
    let mut orbit_camera = None;
    let camera_aspect = state.header.pane_width as f32 / state.header.pane_height as f32;

    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging_size = padded_bytes_per_row as u64 * height as u64;
    let readback_layout = ReadbackLayout {
        padded_bytes_per_row,
        unpadded_bytes_per_row,
        width,
        height,
        pane_width: state.header.pane_width,
        pane_height: state.header.pane_height,
    };
    let staging_buffers: Vec<_> = (0..STAGING_BUFFER_COUNT)
        .map(|slot| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("readback-{slot}")),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();

    // NVENC when the local FFmpeg has it, libx264 otherwise. The renderer
    // itself is always GPU: this is only the encode step, and hard-requiring
    // h264_nvenc made the whole pipeline unusable on any FFmpeg built without
    // it -- including builds that ship av1_nvenc but not h264_nvenc.
    let codec_args: &[&str] = if ffmpeg_has_encoder("h264_nvenc") {
        &["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-cq", "18", "-b:v", "0"]
    } else {
        eprintln!("h264_nvenc unavailable; encoding with libx264 (CPU)");
        &["-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    };

    let mut encoder_process = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            "-",
            "-an",
        ])
        .args(codec_args)
        .args([
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(output_path)
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("start encoder for {}", output_path.display()))?;
    let mut ffmpeg_stdin = encoder_process.stdin.take().context("open FFmpeg stdin")?;
    let mut render_frames =
        BufWriter::new(File::create(render_frames_path).with_context(|| {
            format!("create render telemetry {}", render_frames_path.display())
        })?);
    writeln!(
        render_frames,
        "frame,cpu_submit_ms,readback_ms,overlay_ms,encode_pipe_ms,total_host_ms,boxes,spheres"
    )?;

    // Built lazily on the first decoded frame: all RECORD_ACTOR entries
    // precede any RECORD_FRAME in this format, so `state.actors` is complete
    // by then (see StateReader::next_frame).
    let mut mesh_actors: Vec<Option<MeshActor>> = Vec::new();
    let mut submitted_frames = 0_usize;
    let mut written_frames = 0_usize;
    let mut submit_info = Vec::with_capacity(state.header.frame_count as usize);
    let mut output_timings = Vec::with_capacity(state.header.frame_count as usize);
    let start = Instant::now();
    while let Some(frame_index) = state.next_frame()? {
        let submit_start = Instant::now();
        if frame_index as usize != submitted_frames {
            bail!("state frame order error: got {frame_index}, expected {submitted_frames}");
        }
        if submitted_frames >= STAGING_BUFFER_COUNT {
            let slot = submitted_frames % STAGING_BUFFER_COUNT;
            let simulation_frame = simulation_telemetry
                .as_ref()
                .and_then(|telemetry| telemetry.for_output_frame(written_frames as u32, fps));
            let timing = write_staging_frame(
                &device,
                &staging_buffers[slot],
                &readback_layout,
                &mut ffmpeg_stdin,
                simulation_frame,
                submit_info[written_frames],
            )?;
            write_render_telemetry(
                &mut render_frames,
                written_frames,
                submit_info[written_frames],
                &timing,
            )?;
            output_timings.push(timing);
            written_frames += 1;
        }
        if let Some(observer) = impact_camera.as_mut() {
            let camera = observer.update(&state.actors, state.header.cameras[3]);
            let uniform = camera_uniform(camera, camera_aspect);
            queue.write_buffer(&camera_buffers[3], 0, bytemuck::bytes_of(&uniform));
        }
        if orbit_camera.is_none() {
            let bounds =
                scene_bounds(&state.actors).context("initial frame has no visible scene bounds")?;
            orbit_camera = Some(OverviewOrbitCamera::new(
                state.header.cameras[0],
                bounds,
                camera_aspect,
            ));
            for camera_index in 1..=2 {
                let padding = if camera_index == 1 { 0.88 } else { 1.02 };
                let camera = fit_camera_to_bounds(
                    state.header.cameras[camera_index],
                    bounds,
                    camera_aspect,
                    padding,
                );
                let uniform = camera_uniform(camera, camera_aspect);
                queue.write_buffer(
                    &camera_buffers[camera_index],
                    0,
                    bytemuck::bytes_of(&uniform),
                );
            }
        }
        let orbit = orbit_camera
            .as_ref()
            .expect("overview camera initialized")
            .at_time(frame_index as f32 / fps as f32);
        let orbit_uniform = camera_uniform(orbit, camera_aspect);
        queue.write_buffer(&camera_buffers[0], 0, bytemuck::bytes_of(&orbit_uniform));
        if mesh_actors.is_empty() && !state.actors.is_empty() {
            mesh_actors = build_mesh_actors(&device, &state.actors);
        }
        let (boxes, spheres) = collect_instances(&state.actors, sleep_tint);
        box_buffer.upload(&device, &queue, "box-instances", &boxes);
        sphere_buffer.upload(&device, &queue, "sphere-instances", &spheres);
        let visible_meshes = update_mesh_instances(&queue, &state.actors, &mesh_actors, sleep_tint);

        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blast-mini-city-frame"),
        });
        {
            let mut pass = commands.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("four-camera-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.58,
                            g: 0.68,
                            b: 0.82,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            for (camera_index, camera_group) in camera_groups.iter().enumerate() {
                let x = (camera_index % 2) as f32 * state.header.pane_width as f32;
                let y = (camera_index / 2) as f32 * state.header.pane_height as f32;
                pass.set_viewport(
                    x,
                    y,
                    state.header.pane_width as f32,
                    state.header.pane_height as f32,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(
                    x as u32,
                    y as u32,
                    state.header.pane_width,
                    state.header.pane_height,
                );
                pass.set_bind_group(0, camera_group, &[]);
                draw_instances(&mut pass, &cube, &box_buffer.buffer, boxes.len() as u32);
                draw_instances(
                    &mut pass,
                    &sphere,
                    &sphere_buffer.buffer,
                    spheres.len() as u32,
                );
                for &index in &visible_meshes {
                    // Safe: `visible_meshes` only contains indices with a
                    // `Some` entry (see update_mesh_instances).
                    let mesh_actor = mesh_actors[index].as_ref().expect("visible mesh actor");
                    draw_instances(&mut pass, &mesh_actor.mesh, &mesh_actor.instances, 1);
                }
            }
        }
        commands.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffers[submitted_frames % STAGING_BUFFER_COUNT],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([commands.finish()]);
        submit_info.push(RenderSubmitInfo {
            cpu_submit_ms: submit_start.elapsed().as_secs_f64() * 1000.0,
            boxes: boxes.len(),
            spheres: spheres.len(),
        });
        submitted_frames += 1;
        if submitted_frames.is_multiple_of(fps as usize) {
            let wall = start.elapsed().as_secs_f64();
            println!(
                "render frame {}/{} wall={wall:.1}s factor={:.2}x boxes={} spheres={}",
                submitted_frames,
                state.header.frame_count,
                (submitted_frames as f64 / fps as f64) / wall,
                boxes.len(),
                spheres.len()
            );
        }
    }

    while written_frames < submitted_frames {
        let slot = written_frames % STAGING_BUFFER_COUNT;
        let simulation_frame = simulation_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.for_output_frame(written_frames as u32, fps));
        let timing = write_staging_frame(
            &device,
            &staging_buffers[slot],
            &readback_layout,
            &mut ffmpeg_stdin,
            simulation_frame,
            submit_info[written_frames],
        )?;
        write_render_telemetry(
            &mut render_frames,
            written_frames,
            submit_info[written_frames],
            &timing,
        )?;
        output_timings.push(timing);
        written_frames += 1;
    }
    render_frames
        .flush()
        .context("flush render frame telemetry")?;
    drop(ffmpeg_stdin);
    let status = encoder_process.wait().context("wait for FFmpeg")?;
    if !status.success() {
        bail!("FFmpeg/NVENC exited with {status}");
    }
    if written_frames != state.header.frame_count as usize {
        bail!(
            "rendered {written_frames} frames, expected {}",
            state.header.frame_count
        );
    }
    write_render_summary(
        render_summary_path,
        start.elapsed().as_secs_f64(),
        fps,
        &submit_info,
        &output_timings,
    )?;
    drop(camera_buffers);
    println!(
        "render done. frames={} wall={:.2}s output={}",
        written_frames,
        start.elapsed().as_secs_f64(),
        output_path.display()
    );
    Ok(())
}

fn draw_instances<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    mesh: &'a Mesh,
    instances: &'a wgpu::Buffer,
    instance_count: u32,
) {
    if instance_count == 0 {
        return;
    }
    pass.set_vertex_buffer(0, mesh.vertex.slice(..));
    pass.set_vertex_buffer(1, instances.slice(..));
    pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.index_count, 0, 0..instance_count);
}

fn collect_instances(
    actors: &[Actor],
    sleep_tint: bool,
) -> (Vec<InstanceRaw>, Vec<InstanceRaw>) {
    let mut boxes = Vec::with_capacity(INITIAL_INSTANCE_CAPACITY);
    let mut spheres = Vec::with_capacity(16);
    for actor in actors.iter().filter(|actor| actor.visible) {
        let actor_matrix = transform_matrix(actor.pose);
        let color = part_color(actor.part, actor.sleeping && sleep_tint);
        for shape in &actor.shapes {
            match shape {
                Shape::Box {
                    half_extents,
                    local,
                } => {
                    let model =
                        actor_matrix * transform_matrix(*local) * Mat4::from_scale(*half_extents);
                    boxes.push(InstanceRaw {
                        model: model.to_cols_array_2d(),
                        color,
                    });
                }
                Shape::Sphere { radius, local } => {
                    let model = actor_matrix
                        * transform_matrix(*local)
                        * Mat4::from_scale(Vec3::splat(*radius));
                    spheres.push(InstanceRaw {
                        model: model.to_cols_array_2d(),
                        color,
                    });
                }
                // Mesh actors have unique per-actor geometry (a fractured
                // shard's real hull), so they cannot share one instanced draw
                // call the way every box/sphere does. Handled separately by
                // `build_mesh_actors` + the per-frame mesh instance update.
                Shape::Mesh { .. } => {}
            }
        }
    }
    (boxes, spheres)
}

/// Per-actor GPU resources for a `Shape::Mesh` actor: its own vertex/index
/// buffers (geometry is static — defined once, like a fractured shard's real
/// shape) plus a 1-element instance buffer rewritten each frame with its
/// current pose. Unlike boxes/spheres, mesh actors are never batched into one
/// shared instanced draw call because every fragment's geometry differs.
struct MeshActor {
    mesh: Mesh,
    instances: wgpu::Buffer,
}

fn build_mesh_actors(device: &wgpu::Device, actors: &[Actor]) -> Vec<Option<MeshActor>> {
    actors
        .iter()
        .map(|actor| match actor.shapes.first() {
            Some(Shape::Mesh {
                positions,
                normals,
                indices,
                ..
            }) => {
                let vertices: Vec<Vertex> = positions
                    .iter()
                    .zip(normals.iter())
                    .map(|(position, normal)| Vertex {
                        position: position.to_array(),
                        normal: normal.to_array(),
                    })
                    .collect();
                let mesh = create_mesh(device, "fragment", &vertices, indices);
                let instances = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fragment-instance"),
                    size: std::mem::size_of::<InstanceRaw>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                Some(MeshActor { mesh, instances })
            }
            _ => None,
        })
        .collect()
}

/// Refresh every visible mesh actor's 1-element instance buffer for this
/// frame and return the actor indices to draw. Mirrors `collect_instances`
/// but per-actor rather than batched, since mesh geometry is unique per actor.
fn update_mesh_instances(
    queue: &wgpu::Queue,
    actors: &[Actor],
    mesh_actors: &[Option<MeshActor>],
    sleep_tint: bool,
) -> Vec<usize> {
    let mut visible = Vec::new();
    for (index, actor) in actors.iter().enumerate() {
        if !actor.visible {
            continue;
        }
        let Some(mesh_actor) = mesh_actors.get(index).and_then(|entry| entry.as_ref()) else {
            continue;
        };
        let Some(Shape::Mesh { local, .. }) = actor.shapes.first() else {
            continue;
        };
        let model = transform_matrix(actor.pose) * transform_matrix(*local);
        let raw = InstanceRaw {
            model: model.to_cols_array_2d(),
            color: part_color(actor.part, actor.sleeping && sleep_tint),
        };
        queue.write_buffer(&mesh_actor.instances, 0, bytemuck::bytes_of(&raw));
        visible.push(index);
    }
    visible
}

fn transform_matrix(transform: Transform) -> Mat4 {
    Mat4::from_rotation_translation(transform.rotation, transform.position)
}

fn part_color(part: u8, apply_sleep_tint: bool) -> [f32; 4] {
    // Part palette is authored by structural role / building index. Sleep tint
    // (when enabled) darkens settled bodies so awake debris reads brighter —
    // disable with --no-sleep-tint for uniform materials.
    let mut rgb = match part {
        0 => [0.72, 0.70, 0.64],
        1 => [0.28, 0.33, 0.42],
        2 => [0.55, 0.50, 0.44],
        3 => [0.45, 0.45, 0.43],
        4 => [0.62, 0.40, 0.20],
        5 => [0.15, 0.85, 0.35],
        // Dust motes from pulverized (crushed) chunks: warm sand, deliberately
        // unlike every structural grey, so "vanished into dust" and "detached
        // and tumbled" are distinguishable at a glance.
        6 => [0.95, 0.80, 0.45],
        7 => [0.42, 0.42, 0.40],
        _ => [0.70, 0.70, 0.70],
    };
    if apply_sleep_tint {
        for channel in &mut rgb {
            *channel *= 0.25;
        }
    }
    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn create_camera_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    cameras: &[Camera; 4],
    aspect: f32,
) -> (Vec<wgpu::Buffer>, Vec<wgpu::BindGroup>) {
    let mut buffers = Vec::with_capacity(4);
    let mut groups = Vec::with_capacity(4);
    for (index, camera) in cameras.iter().enumerate() {
        let uniform = camera_uniform(*camera, aspect);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("camera-{index}")),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("camera-group-{index}")),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        buffers.push(buffer);
        groups.push(group);
    }
    (buffers, groups)
}

fn camera_uniform(camera: Camera, aspect: f32) -> CameraUniform {
    let direction = camera.direction.normalize();
    let up = if direction.dot(Vec3::Y).abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = glam::camera::rh::view::look_at_mat4(camera.eye, camera.eye + direction, up);
    let projection = glam::camera::rh::proj::directx::perspective(
        camera.fov_degrees.to_radians(),
        aspect,
        1.0,
        8_000.0,
    );
    CameraUniform {
        view_projection: (projection * view).to_cols_array_2d(),
    }
}

fn write_staging_frame(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    layout: &ReadbackLayout,
    output: &mut impl Write,
    simulation: Option<&SimulationFrame>,
    submit: RenderSubmitInfo,
) -> Result<OutputTiming> {
    let output_start = Instant::now();
    let readback_start = Instant::now();
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("poll Vulkan readback")?;
    receiver
        .recv()
        .context("receive Vulkan map callback")?
        .context("map Vulkan readback buffer")?;
    let mapped = slice.get_mapped_range()?;
    let mut frame = Vec::with_capacity((layout.unpadded_bytes_per_row * layout.height) as usize);
    for row in mapped.chunks_exact(layout.padded_bytes_per_row as usize) {
        frame.extend_from_slice(&row[..layout.unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    buffer.unmap();
    let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;

    let overlay_start = Instant::now();
    draw_dividers(
        &mut frame,
        layout.width,
        layout.height,
        layout.pane_width,
        layout.pane_height,
    );
    let mut lines = simulation.map(simulation_overlay).unwrap_or_default();
    lines.push(format!(
        "offline renderer: cpu-submit={:5.2}ms  readback={:5.2}ms  visible={} boxes + {} spheres",
        submit.cpu_submit_ms, readback_ms, submit.boxes, submit.spheres
    ));
    draw_overlay(&mut frame, layout.width, layout.height, &lines, 2);
    let overlay_ms = overlay_start.elapsed().as_secs_f64() * 1000.0;

    let encode_start = Instant::now();
    output.write_all(&frame).context("write frame to NVENC")?;
    let encode_pipe_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
    Ok(OutputTiming {
        readback_ms,
        overlay_ms,
        encode_pipe_ms,
        total_host_ms: output_start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn write_render_telemetry(
    output: &mut impl Write,
    frame: usize,
    submit: RenderSubmitInfo,
    timing: &OutputTiming,
) -> Result<()> {
    writeln!(
        output,
        "{},{},{},{},{},{},{},{}",
        frame,
        submit.cpu_submit_ms,
        timing.readback_ms,
        timing.overlay_ms,
        timing.encode_pipe_ms,
        timing.total_host_ms,
        submit.boxes,
        submit.spheres
    )
    .context("write render frame telemetry")
}

#[derive(Default)]
struct SummaryStats {
    mean: f64,
    p95: f64,
    maximum: f64,
}

fn summarize(samples: impl Iterator<Item = f64>) -> SummaryStats {
    let mut values: Vec<_> = samples.collect();
    if values.is_empty() {
        return SummaryStats::default();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.sort_by(f64::total_cmp);
    let p95_index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    SummaryStats {
        mean,
        p95: values[p95_index],
        maximum: *values.last().unwrap_or(&0.0),
    }
}

fn write_render_summary(
    path: &Path,
    wall_seconds: f64,
    fps: u32,
    submit: &[RenderSubmitInfo],
    output: &[OutputTiming],
) -> Result<()> {
    let cpu = summarize(submit.iter().map(|value| value.cpu_submit_ms));
    let readback = summarize(output.iter().map(|value| value.readback_ms));
    let overlay = summarize(output.iter().map(|value| value.overlay_ms));
    let encode = summarize(output.iter().map(|value| value.encode_pipe_ms));
    let total = summarize(output.iter().map(|value| value.total_host_ms));
    let video_seconds = output.len() as f64 / fps as f64;
    let visible_boxes = submit.iter().map(|value| value.boxes).max().unwrap_or(0);
    let visible_spheres = submit.iter().map(|value| value.spheres).max().unwrap_or(0);
    let mut file =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    writeln!(
        file,
        concat!(
            "{{\n",
            "  \"format\": \"blast-render-summary-v1\",\n",
            "  \"frames\": {},\n",
            "  \"videoSeconds\": {},\n",
            "  \"wallSeconds\": {},\n",
            "  \"throughputFactor\": {},\n",
            "  \"maxVisibleBoxes\": {},\n",
            "  \"maxVisibleSpheres\": {},\n",
            "  \"cpuSubmitMilliseconds\": {{\"mean\": {}, \"p95\": {}, \"max\": {}}},\n",
            "  \"readbackMilliseconds\": {{\"mean\": {}, \"p95\": {}, \"max\": {}}},\n",
            "  \"overlayMilliseconds\": {{\"mean\": {}, \"p95\": {}, \"max\": {}}},\n",
            "  \"encodePipeMilliseconds\": {{\"mean\": {}, \"p95\": {}, \"max\": {}}},\n",
            "  \"outputHostMilliseconds\": {{\"mean\": {}, \"p95\": {}, \"max\": {}}}\n",
            "}}"
        ),
        output.len(),
        video_seconds,
        wall_seconds,
        video_seconds / wall_seconds.max(1.0e-9),
        visible_boxes,
        visible_spheres,
        cpu.mean,
        cpu.p95,
        cpu.maximum,
        readback.mean,
        readback.p95,
        readback.maximum,
        overlay.mean,
        overlay.p95,
        overlay.maximum,
        encode.mean,
        encode.p95,
        encode.maximum,
        total.mean,
        total.p95,
        total.maximum
    )?;
    file.flush().context("flush render summary")
}

fn draw_dividers(frame: &mut [u8], width: u32, height: u32, split_x: u32, split_y: u32) {
    let divider = [220_u8, 220, 220, 255];
    for x in 0..width {
        for y in split_y.saturating_sub(1)..=(split_y + 1).min(height - 1) {
            let offset = ((y * width + x) * 4) as usize;
            frame[offset..offset + 4].copy_from_slice(&divider);
        }
    }
    for y in 0..height {
        for x in split_x.saturating_sub(1)..=(split_x + 1).min(width - 1) {
            let offset = ((y * width + x) * 4) as usize;
            frame[offset..offset + 4].copy_from_slice(&divider);
        }
    }
}

fn create_cube_mesh(device: &wgpu::Device) -> Mesh {
    let faces = [
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, 1.0],
            [
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, positions) in faces {
        let base = vertices.len() as u32;
        vertices.extend(
            positions
                .into_iter()
                .map(|position| Vertex { position, normal }),
        );
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    create_mesh(device, "cube", &vertices, &indices)
}

fn create_sphere_mesh(device: &wgpu::Device, slices: u32, stacks: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for stack in 0..=stacks {
        let v = stack as f32 / stacks as f32;
        let phi = std::f32::consts::PI * v;
        for slice in 0..=slices {
            let u = slice as f32 / slices as f32;
            let theta = std::f32::consts::TAU * u;
            let normal = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            vertices.push(Vertex {
                position: normal,
                normal,
            });
        }
    }
    for stack in 0..stacks {
        for slice in 0..slices {
            let row = slices + 1;
            let a = stack * row + slice;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    create_mesh(device, "sphere", &vertices, &indices)
}

fn create_mesh(device: &wgpu::Device, label: &str, vertices: &[Vertex], indices: &[u32]) -> Mesh {
    Mesh {
        vertex: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-vertices")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        index: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-indices")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        index_count: indices.len() as u32,
    }
}
