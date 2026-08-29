//! GPU representation of the full path-traced scene (roadmap Phase 3):
//! quadrics, meshes with concatenated BLAS buffers, TLAS instances,
//! materials, and lights, all as #[repr(C)] scalar-field structs mirrored
//! byte-for-byte by the structs in kernel_pt.metal. Triangle indices and
//! instances are permuted into BVH leaf order at export so leaves address
//! contiguous ranges without an indirection buffer.

use super::scene_buffers::{as_bytes, gpu_object, GpuObject};
use crate::accel::Bvh;
use crate::raytracer::flatten::{matrix_to_f32, FlatObject};
use crate::scene::{LightType, Scene};
use anyhow::Result;

pub const MAX_BOUNCES: u32 = 8;
pub const RR_START: u32 = 3;
pub const FIREFLY_CLAMP: f32 = 60.0;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBvhNode {
    pub min: [f32; 3],
    /// Interior: left-child index (node-local to its BVH). Leaf: first
    /// primitive slot (leaf-ordered buffers).
    pub left_or_first: u32,
    pub max: [f32; 3],
    /// 0 = interior, >0 = leaf primitive count.
    pub count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuMeshInfo {
    /// Offset of this mesh's nodes within the concatenated BLAS buffer.
    pub node_offset: u32,
    /// Offset (in u32s) of this mesh's leaf-ordered triangle indices.
    pub index_offset: u32,
    /// Offset (in vertices) into the concatenated vertex/normal buffers.
    pub vertex_offset: u32,
    /// 1 when per-vertex normals are present.
    pub has_normals: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuInstance {
    pub inv: [f32; 16],
    pub fwd: [f32; 16],
    pub mesh_id: u32,
    pub material_id: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPtMaterial {
    pub diffuse_gain: f32,
    pub diffuse_color: [f32; 3],
    pub diffuse_sigma: f32,
    pub spec_f0: [f32; 3],
    pub spec_f90: [f32; 3],
    pub spec_alpha: f32,
    pub coat_gain: f32,
    pub coat_alpha: f32,
    pub fuzz_gain: f32,
    pub fuzz_color: [f32; 3],
    pub glass_gain: f32,
    pub glass_ior: f32,
    pub glass_alpha: f32,
    pub refr_color: [f32; 3],
    pub emission: [f32; 3],
    pub presence: f32,
    /// Precomputed under_layer_scale (energy layering).
    pub under_scale: f32,
    /// Precomputed normalized lobe-selection weights (d, s, c, f, g);
    /// all-zero when the material reflects nothing.
    pub weights: [f32; 5],
    pub area_light: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPtLight {
    /// 0 point, 1 distant, 2 rect, 3 sphere, 4 disk, 5 dome.
    /// distant reuses `area` as angular_radius; sphere reuses it as radius.
    pub kind: u32,
    /// point/sphere: position; distant: direction; rect: corner; disk: center.
    pub a: [f32; 3],
    pub e1: [f32; 3],
    pub e2: [f32; 3],
    pub normal: [f32; 3],
    pub area: f32,
    /// intensity * color (radiance for rects; the point/distant scale).
    pub radiance: [f32; 3],
    pub pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPtUniforms {
    pub width: u32,
    pub height: u32,
    pub sample_start: u32,
    pub sample_count: u32,
    pub object_count: u32,
    pub instance_count: u32,
    pub light_count: u32,
    pub max_bounces: u32,
    pub rr_start: u32,
    pub y_offset: u32,
    pub firefly_clamp: f32,
    pub background: [f32; 3],
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
    /// Index of the dome light, or u32::MAX.
    pub dome_index: u32,
    pub env_width: u32,
    pub env_height: u32,
    pub env_total: f32,
}

const _: () = assert!(std::mem::size_of::<GpuBvhNode>() == 32);
const _: () = assert!(std::mem::size_of::<GpuMeshInfo>() == 16);
const _: () = assert!(std::mem::size_of::<GpuInstance>() == 144);
const _: () = assert!(std::mem::size_of::<GpuPtMaterial>() == 140);
const _: () = assert!(std::mem::size_of::<GpuPtLight>() == 76);
const _: () = assert!(std::mem::size_of::<GpuPtUniforms>() == 128);

pub struct GpuPtScene {
    pub objects: Vec<GpuObject>,
    /// Material id per legacy object (parallel to `objects`).
    pub object_materials: Vec<u32>,
    pub materials: Vec<GpuPtMaterial>,
    pub lights: Vec<GpuPtLight>,
    pub tlas_nodes: Vec<GpuBvhNode>,
    pub instances: Vec<GpuInstance>,
    pub blas_nodes: Vec<GpuBvhNode>,
    pub tri_indices: Vec<u32>,
    /// xyz triples, all meshes concatenated.
    pub vertices: Vec<f32>,
    /// Parallel to vertices (zeros when a mesh has none).
    pub normals: Vec<f32>,
    pub mesh_infos: Vec<GpuMeshInfo>,
    pub env_pixels: Vec<f32>,
    pub env_marginal: Vec<f32>,
    pub env_conditional: Vec<f32>,
    pub uniforms: GpuPtUniforms,
}

fn export_bvh(bvh: &Bvh) -> Vec<GpuBvhNode> {
    bvh.node_views()
        .map(|(min, max, first, count)| GpuBvhNode {
            min: [min.x as f32, min.y as f32, min.z as f32],
            left_or_first: first,
            max: [max.x as f32, max.y as f32, max.z as f32],
            count,
        })
        .collect()
}

fn v3(v: &crate::math::Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

impl GpuPtScene {
    pub fn build(scene: &Scene) -> Result<Self> {
        // Legacy objects (quadrics + loose polygons) with global material ids.
        let mut objects = Vec::with_capacity(scene.objects.len());
        let mut object_materials = Vec::with_capacity(scene.objects.len());
        for object in &scene.objects {
            let desc = object.describe();
            objects.push(gpu_object(&FlatObject {
                kind: desc.kind,
                inv: matrix_to_f32(&desc.inverse_transform)?,
                fwd: matrix_to_f32(&desc.transform)?,
            }));
            object_materials.push(desc.material_id as u32);
        }

        // Materials: full PBR lobe parameters with values the kernel
        // would otherwise re-derive (F0, alphas, layering, weights)
        // precomputed on the CPU for exact agreement with pt::bxdf.
        let materials: Vec<GpuPtMaterial> = scene
            .materials
            .iter()
            .map(|m| {
                let p = &m.pbr;
                let f0 = p.specular_f0();
                let weights = p.lobe_weights().unwrap_or([0.0; 5]);
                let spec_take = 0.2126 * f0.x + 0.7152 * f0.y + 0.0722 * f0.z;
                let coat_take = 0.04 * p.clearcoat_gain;
                let under = ((1.0 - spec_take) * (1.0 - coat_take)).clamp(0.0, 1.0);
                let alpha = |r: f64| ((r * r).clamp(2.5e-5, 1.0)) as f32;
                GpuPtMaterial {
                    diffuse_gain: p.diffuse_gain as f32,
                    diffuse_color: v3(&p.diffuse_color),
                    diffuse_sigma: p.diffuse_roughness as f32,
                    spec_f0: v3(&f0),
                    spec_f90: v3(&p.specular_edge_color),
                    spec_alpha: alpha(p.specular_roughness),
                    coat_gain: p.clearcoat_gain as f32,
                    coat_alpha: alpha(p.clearcoat_roughness),
                    fuzz_gain: p.fuzz_gain as f32,
                    fuzz_color: v3(&p.fuzz_color),
                    glass_gain: p.glass_gain as f32,
                    glass_ior: p.glass_ior as f32,
                    glass_alpha: alpha(p.glass_roughness),
                    refr_color: v3(&p.refraction_color),
                    emission: v3(&(m.emission + p.glow)),
                    presence: p.presence as f32,
                    under_scale: under as f32,
                    weights: weights.map(|w| w as f32),
                    area_light: m.area_light.map(|i| i as u32).unwrap_or(u32::MAX),
                }
            })
            .collect();

        let mut env_pixels: Vec<f32> = Vec::new();
        let mut env_marginal: Vec<f32> = Vec::new();
        let mut env_conditional: Vec<f32> = Vec::new();
        let mut dome_index = u32::MAX;
        let mut env_dims = (0u32, 0u32, 0.0f32);

        let lights: Vec<GpuPtLight> = scene
            .lights
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let radiance = v3(&(l.color * l.intensity));
                let zero = GpuPtLight {
                    kind: 0,
                    a: [0.0; 3],
                    e1: [0.0; 3],
                    e2: [0.0; 3],
                    normal: [0.0; 3],
                    area: 0.0,
                    radiance,
                    pad: [0.0; 2],
                };
                match &l.light_type {
                    LightType::Point { position } => GpuPtLight {
                        a: [position.x as f32, position.y as f32, position.z as f32],
                        ..zero
                    },
                    LightType::Distant { direction, angular_radius } => GpuPtLight {
                        kind: 1,
                        a: v3(direction),
                        area: *angular_radius as f32,
                        ..zero
                    },
                    LightType::Rect { corner, edge1, edge2, normal, area } => GpuPtLight {
                        kind: 2,
                        a: [corner.x as f32, corner.y as f32, corner.z as f32],
                        e1: v3(edge1),
                        e2: v3(edge2),
                        normal: v3(normal),
                        area: *area as f32,
                        ..zero
                    },
                    LightType::SphereArea { center, radius } => GpuPtLight {
                        kind: 3,
                        a: [center.x as f32, center.y as f32, center.z as f32],
                        area: *radius as f32,
                        ..zero
                    },
                    LightType::DiskArea { center, e1, e2, normal, area } => GpuPtLight {
                        kind: 4,
                        a: [center.x as f32, center.y as f32, center.z as f32],
                        e1: v3(e1),
                        e2: v3(e2),
                        normal: v3(normal),
                        area: *area as f32,
                        ..zero
                    },
                    LightType::Dome => {
                        dome_index = i as u32;
                        if let Some(env) = &l.env {
                            let (w, h, px, marg, cond, total) = env.export();
                            env_dims = (w as u32, h as u32, total as f32);
                            env_pixels = px;
                            env_marginal = marg;
                            env_conditional = cond;
                        }
                        GpuPtLight { kind: 5, ..zero }
                    }
                }
            })
            .collect();

        // Meshes: concatenate, permuting triangles into BLAS leaf order.
        let mut blas_nodes = Vec::new();
        let mut tri_indices = Vec::new();
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut mesh_infos = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let node_offset = blas_nodes.len() as u32;
            let index_offset = tri_indices.len() as u32;
            let vertex_offset = (vertices.len() / 3) as u32;
            blas_nodes.extend(export_bvh(&mesh.blas));
            for &tri in mesh.blas.prim_order() {
                let base = tri as usize * 3;
                tri_indices.push(mesh.indices[base]);
                tri_indices.push(mesh.indices[base + 1]);
                tri_indices.push(mesh.indices[base + 2]);
            }
            for p in &mesh.positions {
                vertices.extend_from_slice(p);
            }
            match &mesh.normals {
                Some(ns) => {
                    for n in ns {
                        normals.extend_from_slice(n);
                    }
                }
                None => normals.extend(std::iter::repeat(0.0f32).take(mesh.positions.len() * 3)),
            }
            mesh_infos.push(GpuMeshInfo {
                node_offset,
                index_offset,
                vertex_offset,
                has_normals: mesh.normals.is_some() as u32,
            });
        }

        // TLAS: instances permuted into leaf order so leaves are contiguous.
        let tlas = {
            let bounds: Vec<_> = scene.instances.iter().map(|i| i.world_bounds).collect();
            Bvh::build(&bounds)
        };
        let instances: Vec<GpuInstance> = tlas
            .prim_order()
            .iter()
            .map(|&i| {
                let inst = &scene.instances[i as usize];
                Ok(GpuInstance {
                    inv: matrix_to_f32(&inst.inverse)?,
                    fwd: matrix_to_f32(&inst.transform)?,
                    mesh_id: inst.mesh_id,
                    material_id: inst.material_id as u32,
                    pad: [0; 2],
                })
            })
            .collect::<Result<_>>()?;
        // Leaf `first` values index positions in prim_order, and the
        // instance array above is permuted into exactly that order, so the
        // exported nodes address it directly. (Same for BLAS leaves and the
        // permuted triangle indices.)
        let tlas_nodes = export_bvh(&tlas);

        let cam = &scene.camera;
        let aspect_ratio = cam.width as f64 / cam.height as f64;
        let half_height = (cam.fov.to_radians() / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        let uniforms = GpuPtUniforms {
            width: cam.width,
            height: cam.height,
            sample_start: 0,
            sample_count: 0,
            object_count: objects.len() as u32,
            instance_count: instances.len() as u32,
            light_count: lights.len() as u32,
            max_bounces: MAX_BOUNCES,
            rr_start: RR_START,
            y_offset: 0,
            firefly_clamp: FIREFLY_CLAMP,
            background: v3(&scene.background_color),
            eye: [cam.eye.x as f32, cam.eye.y as f32, cam.eye.z as f32],
            forward: [cam.forward.x as f32, cam.forward.y as f32, cam.forward.z as f32],
            right: [cam.right.x as f32, cam.right.y as f32, cam.right.z as f32],
            up: [cam.up.x as f32, cam.up.y as f32, cam.up.z as f32],
            half_width: half_width as f32,
            half_height: half_height as f32,
            dome_index,
            env_width: env_dims.0,
            env_height: env_dims.1,
            env_total: env_dims.2,
        };

        let mut out = Self {
            objects,
            object_materials,
            materials,
            lights,
            tlas_nodes,
            instances,
            blas_nodes,
            tri_indices,
            vertices,
            normals,
            mesh_infos,
            env_pixels,
            env_marginal,
            env_conditional,
            uniforms,
        };
        out.ensure_nonempty();
        Ok(out)
    }

    /// Metal buffers cannot be zero-length; counts in uniforms keep the
    /// kernel from reading dummies.
    fn ensure_nonempty(&mut self) {
        if self.objects.is_empty() {
            self.objects.push(unsafe { std::mem::zeroed() });
            self.object_materials.push(0);
        }
        if self.materials.is_empty() {
            self.materials.push(unsafe { std::mem::zeroed() });
        }
        if self.lights.is_empty() {
            self.lights.push(unsafe { std::mem::zeroed() });
        }
        if self.tlas_nodes.is_empty() {
            self.tlas_nodes.push(unsafe { std::mem::zeroed() });
        }
        if self.instances.is_empty() {
            self.instances.push(unsafe { std::mem::zeroed() });
        }
        if self.blas_nodes.is_empty() {
            self.blas_nodes.push(unsafe { std::mem::zeroed() });
        }
        if self.tri_indices.is_empty() {
            self.tri_indices.extend_from_slice(&[0, 0, 0]);
        }
        if self.vertices.is_empty() {
            self.vertices.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        if self.normals.is_empty() {
            self.normals.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        if self.mesh_infos.is_empty() {
            self.mesh_infos.push(unsafe { std::mem::zeroed() });
        }
        if self.env_pixels.is_empty() {
            self.env_pixels.extend_from_slice(&[0.0; 3]);
        }
        if self.env_marginal.is_empty() {
            self.env_marginal.push(0.0);
        }
        if self.env_conditional.is_empty() {
            self.env_conditional.push(0.0);
        }
    }

    pub fn objects_bytes(&self) -> &[u8] {
        as_bytes(&self.objects)
    }
    pub fn object_materials_bytes(&self) -> &[u8] {
        as_bytes(&self.object_materials)
    }
    pub fn materials_bytes(&self) -> &[u8] {
        as_bytes(&self.materials)
    }
    pub fn lights_bytes(&self) -> &[u8] {
        as_bytes(&self.lights)
    }
    pub fn tlas_bytes(&self) -> &[u8] {
        as_bytes(&self.tlas_nodes)
    }
    pub fn instances_bytes(&self) -> &[u8] {
        as_bytes(&self.instances)
    }
    pub fn blas_bytes(&self) -> &[u8] {
        as_bytes(&self.blas_nodes)
    }
    pub fn tri_indices_bytes(&self) -> &[u8] {
        as_bytes(&self.tri_indices)
    }
    pub fn vertices_bytes(&self) -> &[u8] {
        as_bytes(&self.vertices)
    }
    pub fn normals_bytes(&self) -> &[u8] {
        as_bytes(&self.normals)
    }
    pub fn mesh_infos_bytes(&self) -> &[u8] {
        as_bytes(&self.mesh_infos)
    }
    pub fn env_pixels_bytes(&self) -> &[u8] {
        as_bytes(&self.env_pixels)
    }
    pub fn env_marginal_bytes(&self) -> &[u8] {
        as_bytes(&self.env_marginal)
    }
    pub fn env_conditional_bytes(&self) -> &[u8] {
        as_bytes(&self.env_conditional)
    }
}

