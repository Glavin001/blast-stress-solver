use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use anyhow::{bail, Context, Result};
use glam::{Quat, Vec3};

const RECORD_ACTOR: u8 = 1;
const RECORD_FRAME: u8 = 2;
const RECORD_END: u8 = 255;

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };
}

#[derive(Clone, Debug)]
pub enum Shape {
    Box {
        half_extents: Vec3,
        local: Transform,
    },
    Sphere {
        radius: f32,
        local: Transform,
    },
}

#[derive(Debug)]
pub struct Actor {
    pub part: u8,
    pub shapes: Vec<Shape>,
    pub pose: Transform,
    pub sleeping: bool,
    pub visible: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub direction: Vec3,
    pub fov_degrees: f32,
}

#[derive(Debug)]
pub struct Header {
    pub fps: u32,
    pub frame_count: u32,
    pub pane_width: u32,
    pub pane_height: u32,
    pub building_count: u32,
    pub duration_seconds: f32,
    pub settle_seconds: f32,
    pub cameras: [Camera; 4],
}

pub struct StateReader {
    reader: BufReader<File>,
    pub header: Header,
    pub actors: Vec<Actor>,
    ended: bool,
}

impl StateReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(
            File::open(path).with_context(|| format!("open state file {}", path.display()))?,
        );
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"TWSTATE1" {
            bail!("{} is not a TWSTATE1 recording", path.display());
        }

        let version = read_u32(&mut reader)?;
        if version != 1 {
            bail!("unsupported TWSTATE state version {version}");
        }

        let fps = read_u32(&mut reader)?;
        let frame_count = read_u32(&mut reader)?;
        let pane_width = read_u32(&mut reader)?;
        let pane_height = read_u32(&mut reader)?;
        let building_count = read_u32(&mut reader)?;
        let camera_count = read_u32(&mut reader)?;
        if camera_count != 4 {
            bail!("expected four cameras, state file has {camera_count}");
        }
        if fps == 0 || frame_count == 0 || pane_width == 0 || pane_height == 0 {
            bail!("state header contains a zero frame rate, frame count, or pane dimension");
        }
        let duration_seconds = read_f32(&mut reader)?;
        let settle_seconds = read_f32(&mut reader)?;
        let mut cameras = [Camera {
            eye: Vec3::ZERO,
            direction: Vec3::NEG_Z,
            fov_degrees: 45.0,
        }; 4];
        for camera in &mut cameras {
            camera.eye = read_vec3(&mut reader)?;
            camera.direction = read_vec3(&mut reader)?;
            camera.fov_degrees = read_f32(&mut reader)?;
        }

        Ok(Self {
            reader,
            header: Header {
                fps,
                frame_count,
                pane_width,
                pane_height,
                building_count,
                duration_seconds,
                settle_seconds,
                cameras,
            },
            actors: Vec::new(),
            ended: false,
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<u32>> {
        if self.ended {
            return Ok(None);
        }
        loop {
            match read_u8(&mut self.reader)? {
                RECORD_ACTOR => self.read_actor()?,
                RECORD_FRAME => return self.read_frame().map(Some),
                RECORD_END => {
                    self.ended = true;
                    return Ok(None);
                }
                record => bail!("unknown state record type {record}"),
            }
        }
    }

    fn read_actor(&mut self) -> Result<()> {
        let id = read_u32(&mut self.reader)? as usize;
        if id != self.actors.len() {
            bail!(
                "actor IDs must be contiguous: got {id}, expected {}",
                self.actors.len()
            );
        }
        let part = read_u8(&mut self.reader)?;
        let shape_count = read_u32(&mut self.reader)? as usize;
        let mut shapes = Vec::with_capacity(shape_count);
        for _ in 0..shape_count {
            let kind = read_u8(&mut self.reader)?;
            let parameters = read_vec3(&mut self.reader)?;
            let local = read_transform(&mut self.reader)?;
            match kind {
                1 => shapes.push(Shape::Box {
                    half_extents: parameters,
                    local,
                }),
                2 => shapes.push(Shape::Sphere {
                    radius: parameters.x,
                    local,
                }),
                _ => bail!("unknown shape kind {kind} on actor {id}"),
            }
        }
        self.actors.push(Actor {
            part,
            shapes,
            pose: Transform::IDENTITY,
            sleeping: false,
            visible: false,
        });
        Ok(())
    }

    fn read_frame(&mut self) -> Result<u32> {
        let frame_index = read_u32(&mut self.reader)?;
        let update_count = read_u32(&mut self.reader)? as usize;
        for _ in 0..update_count {
            let id = read_u32(&mut self.reader)? as usize;
            let pose = read_transform(&mut self.reader)?;
            let sleeping = read_u8(&mut self.reader)? != 0;
            let actor = self
                .actors
                .get_mut(id)
                .with_context(|| format!("frame {frame_index} references undefined actor {id}"))?;
            actor.pose = pose;
            actor.sleeping = sleeping;
            actor.visible = true;
        }
        Ok(frame_index)
    }
}

fn read_transform(reader: &mut impl Read) -> Result<Transform> {
    let position = read_vec3(reader)?;
    let rotation = Quat::from_xyzw(
        read_f32(reader)?,
        read_f32(reader)?,
        read_f32(reader)?,
        read_f32(reader)?,
    );
    Ok(Transform { position, rotation })
}

fn read_vec3(reader: &mut impl Read) -> Result<Vec3> {
    Ok(Vec3::new(
        read_f32(reader)?,
        read_f32(reader)?,
        read_f32(reader)?,
    ))
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0_u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
