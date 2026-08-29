//! #[repr(C)] mirrors of the structs in kernel.metal.
//!
//! Layout rule: scalar f32/u32 fields and fixed f32 arrays ONLY — no
//! vector types — so Rust repr(C) and MSL agree with zero padding. The
//! const size assertions below lock this down; keep them in sync with any
//! field edit on either side.

use crate::geometry::PrimitiveKind;
use crate::raytracer::flatten::{FlatLightKind, FlatObject, FlatScene};

pub const MAX_DEPTH: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuObject {
    /// 0 sphere, 1 cylinder, 2 cone, 3 torus, 4 disk, 5 paraboloid,
    /// 6 hyperboloid, 7 triangle.
    pub kind: u32,
    /// Per-kind parameters (see gpu_object for the packing).
    pub params: [f32; 9],
    pub inv: [f32; 16], // row-major world->local
    pub fwd: [f32; 16], // row-major local->world
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuMaterial {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub ka: f32,
    pub kd: f32,
    pub ks: f32,
    pub shininess: f32,
    pub reflectivity: f32,
    pub is_metal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuLight {
    pub kind: u32, // 0 point (x,y,z = position), 1 distant (x,y,z = direction)
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuUniforms {
    pub width: u32,
    pub height: u32,
    pub samples_x: u32,
    pub samples_y: u32,
    pub object_count: u32,
    pub light_count: u32,
    pub max_depth: u32,
    pub y_offset: u32, // first image row of this dispatch band
    pub background: [f32; 3],
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
}

const _: () = assert!(std::mem::size_of::<GpuObject>() == 168);
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 36);
const _: () = assert!(std::mem::size_of::<GpuLight>() == 32);
const _: () = assert!(std::mem::size_of::<GpuUniforms>() == 100);

pub struct SceneBuffers {
    pub objects: Vec<GpuObject>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<GpuLight>,
    pub uniforms: GpuUniforms,
}

impl SceneBuffers {
    pub fn build(flat: &FlatScene) -> Self {
        let mut objects: Vec<GpuObject> = flat.objects.iter().map(gpu_object).collect();
        let mut materials: Vec<GpuMaterial> = flat
            .materials
            .iter()
            .map(|m| GpuMaterial {
                r: m.color[0],
                g: m.color[1],
                b: m.color[2],
                ka: m.ka,
                kd: m.kd,
                ks: m.ks,
                shininess: m.shininess,
                reflectivity: m.reflectivity,
                is_metal: m.is_metal as u32,
            })
            .collect();
        let mut lights: Vec<GpuLight> = flat
            .lights
            .iter()
            .map(|l| {
                let (kind, v) = match &l.kind {
                    FlatLightKind::Point { position } => (0u32, *position),
                    FlatLightKind::Distant { direction } => (1u32, *direction),
                };
                GpuLight {
                    kind,
                    x: v[0],
                    y: v[1],
                    z: v[2],
                    intensity: l.intensity,
                    r: l.color[0],
                    g: l.color[1],
                    b: l.color[2],
                }
            })
            .collect();

        // Metal buffers must be non-empty; the counts in uniforms keep the
        // kernel from ever reading a dummy element.
        if objects.is_empty() {
            objects.push(unsafe { std::mem::zeroed() });
        }
        if materials.is_empty() {
            materials.push(unsafe { std::mem::zeroed() });
        }
        if lights.is_empty() {
            lights.push(unsafe { std::mem::zeroed() });
        }

        let cam = &flat.camera;
        let uniforms = GpuUniforms {
            width: flat.width,
            height: flat.height,
            samples_x: flat.pixel_samples.0.max(1),
            samples_y: flat.pixel_samples.1.max(1),
            object_count: flat.objects.len() as u32,
            light_count: flat.lights.len() as u32,
            max_depth: MAX_DEPTH,
            y_offset: 0,
            background: flat.background,
            eye: cam.eye,
            forward: cam.forward,
            right: cam.right,
            up: cam.up,
            half_width: cam.half_width,
            half_height: cam.half_height,
        };

        Self { objects, materials, lights, uniforms }
    }

    pub fn objects_bytes(&self) -> &[u8] {
        as_bytes(&self.objects)
    }
    pub fn materials_bytes(&self) -> &[u8] {
        as_bytes(&self.materials)
    }
    pub fn lights_bytes(&self) -> &[u8] {
        as_bytes(&self.lights)
    }
}

pub fn gpu_object(o: &FlatObject) -> GpuObject {
    let mut params = [0.0f32; 9];
    let kind = match o.kind {
        PrimitiveKind::Sphere { radius, zmin, zmax, thetamax } => {
            params[..4].copy_from_slice(&[
                radius as f32,
                zmin as f32,
                zmax as f32,
                thetamax as f32,
            ]);
            0u32
        }
        PrimitiveKind::Cylinder { radius, zmin, zmax, thetamax } => {
            params[..4].copy_from_slice(&[
                radius as f32,
                zmin as f32,
                zmax as f32,
                thetamax as f32,
            ]);
            1
        }
        PrimitiveKind::Cone { height, radius, thetamax } => {
            params[..3].copy_from_slice(&[height as f32, radius as f32, thetamax as f32]);
            2
        }
        PrimitiveKind::Torus { major_radius, minor_radius, phimin, phimax, thetamax } => {
            params[..5].copy_from_slice(&[
                major_radius as f32,
                minor_radius as f32,
                phimin as f32,
                phimax as f32,
                thetamax as f32,
            ]);
            3
        }
        PrimitiveKind::Disk { height, radius, thetamax } => {
            params[..3].copy_from_slice(&[height as f32, radius as f32, thetamax as f32]);
            4
        }
        PrimitiveKind::Paraboloid { rmax, zmin, zmax, thetamax } => {
            params[..4].copy_from_slice(&[
                rmax as f32,
                zmin as f32,
                zmax as f32,
                thetamax as f32,
            ]);
            5
        }
        PrimitiveKind::Hyperboloid { p1, p2, thetamax } => {
            params[..7].copy_from_slice(&[
                p1[0] as f32,
                p1[1] as f32,
                p1[2] as f32,
                p2[0] as f32,
                p2[1] as f32,
                p2[2] as f32,
                thetamax as f32,
            ]);
            6
        }
        PrimitiveKind::Triangle { v0, v1, v2 } => {
            params.copy_from_slice(&[
                v0[0] as f32,
                v0[1] as f32,
                v0[2] as f32,
                v1[0] as f32,
                v1[1] as f32,
                v1[2] as f32,
                v2[0] as f32,
                v2[1] as f32,
                v2[2] as f32,
            ]);
            7
        }
    };
    GpuObject { kind, params, inv: o.inv, fwd: o.fwd }
}

// Sound: T is repr(C) POD with no padding (checked by the size asserts).
pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
    }
}
