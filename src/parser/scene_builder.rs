//! Interprets the generalized RIB request stream into a Scene.
//!
//! Compliance policy (see COMPLIANCE.md): every request is accepted.
//! Implemented requests take effect; state-only requests are recorded in
//! the graphics state or passthrough dictionaries; everything else warns
//! once and is skipped.

use super::ast::{RibFile, RibRequest, RibValue};
use crate::geometry::{
    Cone, Cylinder, Disk, Hyperboloid, Instance, Intersectable, Mesh, Paraboloid, Sphere, Torus,
    Triangle,
};
use crate::math::{Matrix4, Point3, Vec3};
use crate::scene::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

const MAX_ARCHIVE_DEPTH: usize = 16;

/// Attribute state saved/restored by AttributeBegin/AttributeEnd.
#[derive(Clone)]
struct GraphicsState {
    color: Vec3,
    opacity: Vec3,
    surface: String,
    roughness: f64,
    reverse_orientation: bool,
    sides: u32,
    shading_rate: f64,
    /// Active AreaLightSource: (intensity, color). Subsequent geometry in
    /// this attribute block becomes emissive.
    area_light: Option<(f64, Vec3)>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            color: Vec3::one(),
            opacity: Vec3::one(),
            surface: "matte".to_string(),
            roughness: 0.1,
            reverse_orientation: false,
            sides: 2,
            shading_rate: 1.0,
            area_light: None,
        }
    }
}

/// Everything the builder accumulates into the Scene.
#[derive(Default)]
struct SceneData {
    objects: Vec<Arc<dyn Intersectable>>,
    meshes: Vec<Mesh>,
    instances: Vec<Instance>,
    lights: Vec<Light>,
    materials: Vec<Material>,
}

/// One placed mesh recorded inside an ObjectBegin block: mesh id, its
/// transform relative to the block start, and the material captured at
/// definition time.
struct ObjectDefEntry {
    mesh_id: u32,
    local_transform: Matrix4,
    material: Material,
}

pub struct SceneBuilder {
    width: u32,
    height: u32,
    fov: f64,
    pixel_samples: (u32, u32),
    state: GraphicsState,
    attribute_stack: Vec<GraphicsState>,
    transform_stack: TransformStack,
    /// Named coordinate systems (CoordinateSystem requests).
    coord_systems: HashMap<String, Matrix4>,
    /// Declare registry: name -> declaration string.
    declarations: HashMap<String, String>,
    /// Option/Attribute passthrough: "<category>:<name>" -> raw values.
    passthrough: HashMap<String, Vec<RibValue>>,
    warned: HashSet<String>,
    in_motion: bool,
    motion_sample_taken: bool,
    /// Base directory for resolving ReadArchive paths.
    base_dir: Option<PathBuf>,
    /// Inline archives (ArchiveBegin/End) by name.
    archives: HashMap<String, Vec<RibRequest>>,
    /// Name of an inline archive currently being recorded.
    recording_archive: Option<(String, Vec<RibRequest>)>,
    background: Vec3,
    /// ObjectBegin definitions by handle.
    object_defs: HashMap<String, Vec<ObjectDefEntry>>,
    /// Handle of the object definition currently open, with its entries.
    defining_object: Option<(String, Vec<ObjectDefEntry>)>,
    archive_depth: usize,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            width: 640,
            height: 480,
            fov: 45.0,
            pixel_samples: (1, 1),
            state: GraphicsState::default(),
            attribute_stack: Vec::new(),
            transform_stack: TransformStack::new(),
            coord_systems: HashMap::new(),
            declarations: HashMap::new(),
            passthrough: HashMap::new(),
            warned: HashSet::new(),
            in_motion: false,
            motion_sample_taken: false,
            base_dir: None,
            background: Vec3::zero(),
            archives: HashMap::new(),
            recording_archive: None,
            object_defs: HashMap::new(),
            defining_object: None,
            archive_depth: 0,
        }
    }

    /// Directory against which ReadArchive paths resolve.
    pub fn with_base_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    pub fn build(mut self, requests: &RibFile) -> Result<Scene> {
        let mut data = SceneData::default();
        self.run(requests, &mut data)?;

        let camera = Camera::new(self.width, self.height, self.fov);
        let mut scene = Scene::new(camera);
        scene.objects = data.objects;
        scene.meshes = data.meshes;
        scene.instances = data.instances;
        scene.lights = data.lights;
        scene.materials = data.materials;
        scene.pixel_samples = self.pixel_samples;
        scene.background_color = self.background;
        scene.build_tlas();
        Ok(scene)
    }

    fn run(&mut self, requests: &[RibRequest], data: &mut SceneData) -> Result<()> {
        for request in requests {
            // Inline archive recording captures requests verbatim.
            if let Some((_, recorded)) = &mut self.recording_archive {
                if request.name == "ArchiveEnd" {
                    let (name, recorded) = self.recording_archive.take().unwrap();
                    self.archives.insert(name, recorded);
                } else {
                    recorded.push(request.clone());
                }
                continue;
            }
            // Motion blocks: use the first time sample, skip the rest.
            if self.in_motion && request.name != "MotionEnd" {
                if self.motion_sample_taken {
                    continue;
                }
                self.motion_sample_taken = true;
            }
            self.process(request, data)?;
        }
        Ok(())
    }

    fn process(&mut self, req: &RibRequest, data: &mut SceneData) -> Result<()> {
        match req.name.as_str() {
            // ---- structure ------------------------------------------------
            "version" | "WorldBegin" | "WorldEnd" | "FrameBegin" | "FrameEnd" => {}
            "AttributeBegin" => {
                self.attribute_stack.push(self.state.clone());
                self.transform_stack.push();
            }
            "AttributeEnd" => {
                if let Some(saved) = self.attribute_stack.pop() {
                    self.state = saved;
                }
                self.transform_stack.pop();
            }
            "TransformBegin" => self.transform_stack.push(),
            "TransformEnd" => self.transform_stack.pop(),
            "MotionBegin" => {
                self.in_motion = true;
                self.motion_sample_taken = false;
            }
            "MotionEnd" => {
                self.in_motion = false;
                self.motion_sample_taken = false;
            }
            // Conditional RIB: conditions are not evaluated; bodies process.
            "IfBegin" | "ElseIf" | "Else" | "IfEnd" => {}

            // ---- archives & instancing ------------------------------------
            "ArchiveBegin" => {
                if let Some(name) = req.string(0) {
                    self.recording_archive = Some((name.to_string(), Vec::new()));
                }
            }
            "ArchiveEnd" => {} // handled in run(); stray ends are ignored
            "ReadArchive" => {
                if let Some(name) = req.string(0) {
                    self.read_archive(name, data)?;
                }
            }
            "ObjectBegin" => {
                let handle = req
                    .string(0)
                    .map(str::to_string)
                    .or_else(|| req.number(0).map(|n| n.to_string()))
                    .unwrap_or_else(|| "0".to_string());
                self.attribute_stack.push(self.state.clone());
                self.transform_stack.push();
                // Geometry inside the block records relative to the block
                // start: reset to identity while defining.
                self.transform_stack.set(Matrix4::identity());
                self.defining_object = Some((handle, Vec::new()));
            }
            "ObjectEnd" => {
                if let Some((handle, entries)) = self.defining_object.take() {
                    self.object_defs.insert(handle, entries);
                }
                if let Some(saved) = self.attribute_stack.pop() {
                    self.state = saved;
                }
                self.transform_stack.pop();
            }
            "ObjectInstance" => {
                let handle = req
                    .string(0)
                    .map(str::to_string)
                    .or_else(|| req.number(0).map(|n| n.to_string()))
                    .unwrap_or_else(|| "0".to_string());
                if let Some(entries) = self.object_defs.get(&handle) {
                    let placement = self.transform_stack.current();
                    for entry in entries {
                        data.materials.push(entry.material.clone());
                        let material_id = data.materials.len() - 1;
                        data.instances.push(Instance::new(
                            entry.mesh_id,
                            material_id,
                            placement * entry.local_transform,
                            &data.meshes[entry.mesh_id as usize],
                        ));
                    }
                } else {
                    self.warn_once(&format!("ObjectInstance \"{handle}\" has no definition"));
                }
            }

            // ---- options & camera ----------------------------------------
            "Format" => {
                if let (Some(w), Some(h)) = (req.number(0), req.number(1)) {
                    self.width = w as u32;
                    self.height = h as u32;
                }
            }
            "Projection" => {
                if req.string(0) == Some("perspective") {
                    if let Some(fov) = req.params_from(1).get_number("fov") {
                        self.fov = fov;
                    }
                }
            }
            "PixelSamples" => {
                if let (Some(x), Some(y)) = (req.number(0), req.number(1)) {
                    self.pixel_samples = ((x as u32).max(1), (y as u32).max(1));
                }
            }
            "Declare" => {
                if let (Some(name), Some(decl)) = (req.string(0), req.string(1)) {
                    self.declarations.insert(name.to_string(), decl.to_string());
                }
            }
            "Option" | "Attribute" => {
                if let Some(category) = req.string(0) {
                    // Extension: `Option "background" "color" [r g b]` sets
                    // the miss color (a dome light supersedes this at P4).
                    if req.name == "Option" && category == "background" {
                        if let Some(c) = req.params_from(1).get_numbers("color") {
                            if c.len() >= 3 {
                                self.background = Vec3::new(c[0], c[1], c[2]);
                            }
                        }
                    }
                    for (token, value) in req.params_from(1).iter() {
                        self.passthrough
                            .insert(format!("{category}:{token}"), vec![(*value).clone()]);
                    }
                }
            }
            // Recorded (no render effect yet); see COMPLIANCE.md.
            "Display" | "Clipping" | "ClippingPlane" | "CropWindow" | "ScreenWindow"
            | "FrameAspectRatio" | "Shutter" | "PixelFilter" | "PixelVariance" | "Exposure"
            | "Quantize" | "Hider" | "Integrator" | "Basis" | "TextureCoordinates"
            | "ShadingInterpolation" | "DepthOfField" | "RelativeDetail" => {
                self.record(req);
            }

            // ---- transforms & spaces -------------------------------------
            "Identity" => self.transform_stack.set(Matrix4::identity()),
            "Transform" => {
                if let Some(m) = req.values.first().and_then(RibValue::as_numbers) {
                    if let Some(mat) = matrix_from_rib(m) {
                        self.transform_stack.set(mat);
                    }
                }
            }
            "ConcatTransform" => {
                if let Some(m) = req.values.first().and_then(RibValue::as_numbers) {
                    if let Some(mat) = matrix_from_rib(m) {
                        self.transform_stack.apply(mat);
                    }
                }
            }
            "Translate" => {
                if let (Some(x), Some(y), Some(z)) = (req.number(0), req.number(1), req.number(2)) {
                    self.transform_stack.apply(Matrix4::translate(x, y, z));
                }
            }
            "Rotate" => {
                if let (Some(a), Some(x), Some(y), Some(z)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    self.transform_stack.apply(Matrix4::rotate(a, x, y, z));
                }
            }
            "Scale" => {
                if let (Some(x), Some(y), Some(z)) = (req.number(0), req.number(1), req.number(2)) {
                    self.transform_stack.apply(Matrix4::scale(x, y, z));
                }
            }
            "CoordinateSystem" | "ScopedCoordinateSystem" => {
                if let Some(name) = req.string(0) {
                    self.coord_systems
                        .insert(name.to_string(), self.transform_stack.current());
                }
            }

            // ---- attributes ----------------------------------------------
            "Color" => {
                if let Some(c) = color_from(req) {
                    self.state.color = c;
                }
            }
            "Opacity" => {
                if let Some(c) = color_from(req) {
                    self.state.opacity = c;
                }
            }
            "Surface" => {
                if let Some(name) = req.string(0) {
                    self.state.surface = name.to_string();
                    self.state.roughness =
                        req.params_from(1).get_number("roughness").unwrap_or(0.1);
                }
            }
            "Orientation" => {
                self.state.reverse_orientation = req.string(0) != Some("outside");
            }
            "ReverseOrientation" => {
                self.state.reverse_orientation = !self.state.reverse_orientation;
            }
            "Sides" => {
                if let Some(s) = req.number(0) {
                    self.state.sides = s as u32;
                }
            }
            "ShadingRate" => {
                if let Some(r) = req.number(0) {
                    self.state.shading_rate = r;
                }
            }

            // ---- lights ---------------------------------------------------
            "LightSource" => {
                if let Some(light) = self.parse_light(req) {
                    data.lights.push(light);
                }
            }
            "AreaLightSource" => {
                let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
                let params = req.params_from(params_start);
                let intensity = params.get_number("intensity").unwrap_or(1.0);
                let color = params
                    .get_numbers("lightcolor")
                    .and_then(|v| (v.len() >= 3).then(|| Vec3::new(v[0], v[1], v[2])))
                    .unwrap_or(Vec3::one());
                self.state.area_light = Some((intensity, color));
            }

            // ---- meshes ---------------------------------------------------
            "PointsPolygons" => {
                self.points_polygons(req, None, data);
            }
            "PointsGeneralPolygons" => {
                // With loop counts: use each polygon's first (outer) loop;
                // holes are not supported yet.
                let nloops = req.values.first().and_then(RibValue::as_numbers);
                if let Some(nloops) = nloops {
                    if nloops.iter().any(|n| *n as usize != 1) {
                        self.warn_once("PointsGeneralPolygons holes are not supported; using outer loops only");
                    }
                    let shifted = RibRequest {
                        name: req.name.clone(),
                        values: req.values[1..].to_vec(),
                    };
                    self.points_polygons(&shifted, None, data);
                }
            }

            // ---- quadric / polygon geometry ------------------------------
            "Sphere" => {
                if let (Some(r), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Sphere::new(
                        r, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Cylinder" => {
                if let (Some(r), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Cylinder::new(
                        r, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Cone" => {
                if let (Some(h), Some(r), Some(tm)) = (req.number(0), req.number(1), req.number(2))
                {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Cone::new(
                        h, r, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Torus" => {
                if let (Some(major), Some(minor), Some(phimin), Some(phimax), Some(tm)) = (
                    req.number(0),
                    req.number(1),
                    req.number(2),
                    req.number(3),
                    req.number(4),
                ) {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Torus::new(
                        major, minor, phimin, phimax, tm, id,
                        self.transform_stack.current(),
                    )));
                }
            }
            "Disk" => {
                if let (Some(h), Some(r), Some(tm)) = (req.number(0), req.number(1), req.number(2))
                {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Disk::new(
                        h, r, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Paraboloid" => {
                if let (Some(rmax), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Paraboloid::new(
                        rmax, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Hyperboloid" => {
                let n: Vec<Option<f64>> = (0..7).map(|i| req.number(i)).collect();
                if n.iter().all(Option::is_some) {
                    if self.reject_in_object_def(&req.name) {
                        return Ok(());
                    }
                    let v: Vec<f64> = n.into_iter().map(Option::unwrap).collect();
                    let id = self.push_material(&mut data.materials);
                    data.objects.push(Arc::new(Hyperboloid::new(
                        [v[0], v[1], v[2]],
                        [v[3], v[4], v[5]],
                        v[6],
                        id,
                        self.transform_stack.current(),
                    )));
                }
            }
            "Polygon" => {
                if let Some(p) = req.params_from(0).get_numbers("P") {
                    if p.len() >= 9 {
                        if self.reject_in_object_def(&req.name) {
                            return Ok(());
                        }
                        let transform = self.transform_stack.current();
                        let vertex =
                            |i: usize| -> [f64; 3] { [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]] };
                        let world = |v: [f64; 3]| {
                            transform.transform_point(&Point3::new(v[0], v[1], v[2]))
                        };

                        let mut material = self.make_material();
                        if let Some((intensity, color)) = self.state.area_light {
                            material.emission = color * intensity;
                            if p.len() == 12 {
                                let c = world(vertex(0));
                                let e1 = world(vertex(1)) - c;
                                let e2 = world(vertex(3)) - c;
                                material.area_light = Some(data.lights.len());
                                data.lights.push(Light::rect(c, e1, e2, intensity, color));
                            }
                        }
                        data.materials.push(material);
                        let id = data.materials.len() - 1;

                        for i in 1..(p.len() / 3 - 1) {
                            data.objects.push(Arc::new(Triangle::new(
                                vertex(0),
                                vertex(i),
                                vertex(i + 1),
                                id,
                                transform,
                            )));
                        }
                    }
                }
            }

            // ---- everything else: accept and warn once --------------------
            _ => {
                self.warn_once(&format!(
                    "RIB request '{}' not implemented yet (see COMPLIANCE.md); skipping",
                    req.name
                ));
            }
        }
        Ok(())
    }

    /// Quadrics inside ObjectBegin blocks are not yet instanced; warn+skip.
    fn reject_in_object_def(&mut self, name: &str) -> bool {
        if self.defining_object.is_some() {
            self.warn_once(&format!(
                "'{name}' inside ObjectBegin is not supported yet (meshes only); skipping"
            ));
            true
        } else {
            false
        }
    }

    fn warn_once(&mut self, message: &str) {
        if self.warned.insert(message.to_string()) {
            eprintln!("warning: {message}");
        }
    }

    /// Build a mesh from PointsPolygons-style values: [nverts] [indices]
    /// then "P"/"N"/"st" params.
    fn points_polygons(&mut self, req: &RibRequest, _basis: Option<()>, data: &mut SceneData) {
        let Some(nverts) = req.values.first().and_then(RibValue::as_numbers) else {
            return;
        };
        let Some(vert_indices) = req.values.get(1).and_then(RibValue::as_numbers) else {
            return;
        };
        let params = req.params_from(2);
        let Some(p) = params.get_numbers("P") else {
            self.warn_once("PointsPolygons without \"P\"; skipping");
            return;
        };

        let positions: Vec<[f32; 3]> = p
            .chunks_exact(3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect();
        let normals = params.get_numbers("N").and_then(|n| {
            if n.len() == p.len() {
                Some(
                    n.chunks_exact(3)
                        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                        .collect::<Vec<_>>(),
                )
            } else {
                self.warn_once("PointsPolygons \"N\" is not per-vertex; ignoring normals");
                None
            }
        });
        let st = params.get_numbers("st").and_then(|s| {
            (s.len() * 3 == p.len() * 2).then(|| {
                s.chunks_exact(2)
                    .map(|c| [c[0] as f32, c[1] as f32])
                    .collect::<Vec<_>>()
            })
        });

        // Fan-triangulate each polygon.
        let mut indices: Vec<u32> = Vec::new();
        let mut cursor = 0usize;
        for &nv in nverts {
            let nv = nv as usize;
            if cursor + nv > vert_indices.len() {
                break;
            }
            let poly = &vert_indices[cursor..cursor + nv];
            for i in 1..nv.saturating_sub(1) {
                indices.push(poly[0] as u32);
                indices.push(poly[i] as u32);
                indices.push(poly[i + 1] as u32);
            }
            cursor += nv;
        }
        if indices.is_empty() {
            return;
        }

        let mesh = Mesh::new(positions, indices, normals, st);
        let mesh_id = data.meshes.len() as u32;
        data.meshes.push(mesh);

        let mut material = self.make_material();
        if let Some((intensity, color)) = self.state.area_light {
            // Emissive meshes glow (BSDF hits); they are not yet sampleable
            // lights.
            material.emission = color * intensity;
        }

        if let Some((_, entries)) = &mut self.defining_object {
            entries.push(ObjectDefEntry {
                mesh_id,
                local_transform: self.transform_stack.current(),
                material,
            });
        } else {
            data.materials.push(material);
            let material_id = data.materials.len() - 1;
            data.instances.push(Instance::new(
                mesh_id,
                material_id,
                self.transform_stack.current(),
                &data.meshes[mesh_id as usize],
            ));
        }
    }

    fn read_archive(&mut self, name: &str, data: &mut SceneData) -> Result<()> {
        if self.archive_depth >= MAX_ARCHIVE_DEPTH {
            self.warn_once(&format!("ReadArchive \"{name}\": max depth exceeded; skipping"));
            return Ok(());
        }
        // Inline archives take precedence over files.
        if let Some(requests) = self.archives.get(name).cloned() {
            self.archive_depth += 1;
            self.run(&requests, data)?;
            self.archive_depth -= 1;
            return Ok(());
        }
        let path = match &self.base_dir {
            Some(dir) => dir.join(name),
            None => PathBuf::from(name),
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => match super::parse_rib(&content) {
                Ok(requests) => {
                    self.archive_depth += 1;
                    self.run(&requests, data)?;
                    self.archive_depth -= 1;
                }
                Err(e) => self.warn_once(&format!("ReadArchive \"{name}\": parse error: {e}")),
            },
            Err(e) => self.warn_once(&format!("ReadArchive \"{name}\": {e}")),
        }
        Ok(())
    }

    /// Store a recognized state-only request for later phases.
    fn record(&mut self, req: &RibRequest) {
        self.passthrough
            .insert(format!("request:{}", req.name), req.values.clone());
    }

    fn parse_light(&self, req: &RibRequest) -> Option<Light> {
        let light_type = req.string(0)?;
        // `LightSource "type" <handle> params...` — the handle (number or
        // string) is present when the value count is even (type + handle +
        // token/value pairs); classic files may omit it.
        let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
        let params = req.params_from(params_start);
        let intensity = params.get_number("intensity").unwrap_or(1.0);
        let color = Vec3::one();

        match light_type {
            "pointlight" => {
                let from = params
                    .get_numbers("from")
                    .and_then(point_from)
                    .unwrap_or(Point3::new(0.0, 0.0, 0.0));
                Some(Light::point(from, intensity, color))
            }
            "distantlight" => {
                let from = params
                    .get_numbers("from")
                    .and_then(point_from)
                    .unwrap_or(Point3::new(0.0, 0.0, -10.0));
                let to = params
                    .get_numbers("to")
                    .and_then(point_from)
                    .unwrap_or(Point3::new(0.0, 0.0, 0.0));
                Some(Light::distant((to - from).normalize(), intensity, color))
            }
            _ => None,
        }
    }

    /// Material from the current attribute state.
    fn make_material(&self) -> Material {
        match self.state.surface.as_str() {
            "plastic" => Material::plastic(self.state.color, self.state.roughness),
            "metal" => Material::metal(self.state.color, self.state.roughness),
            _ => Material::matte(self.state.color),
        }
    }

    /// Create a material from the current attribute state; returns its id.
    fn push_material(&self, materials: &mut Vec<Material>) -> usize {
        materials.push(self.make_material());
        materials.len() - 1
    }
}

fn color_from(req: &RibRequest) -> Option<Vec3> {
    if let (Some(r), Some(g), Some(b)) = (req.number(0), req.number(1), req.number(2)) {
        return Some(Vec3::new(r, g, b));
    }
    let v = req.values.first()?.as_numbers()?;
    if v.len() >= 3 {
        Some(Vec3::new(v[0], v[1], v[2]))
    } else {
        None
    }
}

fn point_from(v: &[f64]) -> Option<Point3> {
    if v.len() >= 3 {
        Some(Point3::new(v[0], v[1], v[2]))
    } else {
        None
    }
}

/// RIB matrices are 16 numbers in row-major premultiply convention; our
/// Matrix4 uses column-vector convention, so transpose on ingest.
fn matrix_from_rib(m: &[f64]) -> Option<Matrix4> {
    if m.len() != 16 {
        return None;
    }
    let mut rows = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            rows[r][c] = m[c * 4 + r];
        }
    }
    Some(Matrix4::new(rows))
}

impl Default for SceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_rib;

    #[test]
    fn test_single_pass_build_with_attribute_blocks() {
        let input = r#"
            Format 320 240 1.0
            PixelSamples 2 2
            Projection "perspective" "fov" [45]
            WorldBegin
                LightSource "distantlight" 1 "from" [5 10 -10] "to" [0 0 0] "intensity" [0.8]
                AttributeBegin
                    Color 0 0 1
                    Surface "plastic"
                    Translate -2 0 5
                    Sphere 1 -1 1 360
                AttributeEnd
                AttributeBegin
                    Color 1 0 0
                    Surface "matte"
                    Translate 2 0 5
                    Cylinder 0.5 -1 1 360
                AttributeEnd
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();

        assert_eq!(scene.objects.len(), 2);
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.materials.len(), 2);
        assert_eq!(scene.pixel_samples, (2, 2));
        assert_eq!(scene.camera.width, 320);
        assert_eq!(scene.materials[0].color, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(scene.materials[1].color, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_attribute_end_restores_state() {
        let input = r#"
            WorldBegin
                Color 1 1 1
                AttributeBegin
                    Color 0 1 0
                    Surface "metal"
                AttributeEnd
                Translate 0 0 5
                Sphere 1 -1 1 360
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();

        assert_eq!(scene.objects.len(), 1);
        assert_eq!(scene.materials[0].color, Vec3::one());
        assert!(matches!(
            scene.materials[0].material_type,
            crate::scene::MaterialType::Matte
        ));
    }

    #[test]
    fn test_new_quadrics_and_polygon() {
        let input = r#"
            WorldBegin
                Translate 0 0 5
                Torus 1 0.3 0 360 360
                Disk 0 1 360
                Paraboloid 1 0 1 360
                Hyperboloid 1 0 -1 1 0 1 360
                Polygon "P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        assert_eq!(scene.objects.len(), 6);
    }

    #[test]
    fn test_concat_transform_matches_translate() {
        let a = parse_rib("WorldBegin ConcatTransform [1 0 0 0 0 1 0 0 0 0 1 0 3 4 5 1] Sphere 1 -1 1 360 WorldEnd").unwrap();
        let b = parse_rib("WorldBegin Translate 3 4 5 Sphere 1 -1 1 360 WorldEnd").unwrap();
        let sa = SceneBuilder::new().build(&a).unwrap();
        let sb = SceneBuilder::new().build(&b).unwrap();
        let da = sa.objects[0].describe();
        let db = sb.objects[0].describe();
        assert_eq!(da.transform.rows(), db.transform.rows());
    }

    #[test]
    fn test_unknown_requests_warn_but_build() {
        let input = r#"
            Option "searchpath" "string shader" ["/tmp"]
            Attribute "identifier" "name" ["ball"]
            GeometricApproximation "flatness" 0.5
            WorldBegin
                SubdivisionMesh "catmull-clark" [3] [0 1 2] [] [] [] []
                Sphere 1 -1 1 360
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        assert_eq!(scene.objects.len(), 1);
    }

    #[test]
    fn test_motion_block_takes_first_sample() {
        let input = r#"
            WorldBegin
                MotionBegin [0 1]
                    Translate 0 0 5
                    Translate 0 0 9
                MotionEnd
                Sphere 1 -1 1 360
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        let desc = scene.objects[0].describe();
        assert!((desc.transform.rows()[2][3] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_points_polygons_builds_mesh() {
        let input = r#"
            WorldBegin
                Translate 0 0 5
                PointsPolygons [4 4] [0 1 2 3  4 5 6 7]
                    "P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0
                          -1 -1 1  1 -1 1  1 1 1  -1 1 1]
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.instances.len(), 1);
        assert_eq!(scene.meshes[0].triangle_count(), 4); // two quads
        assert_eq!(scene.triangle_count(), 4);
    }

    #[test]
    fn test_object_instance_shares_mesh() {
        let input = r#"
            WorldBegin
                ObjectBegin "quad"
                    PointsPolygons [4] [0 1 2 3] "P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]
                ObjectEnd
                Translate -2 0 5
                ObjectInstance "quad"
                Translate 4 0 0
                ObjectInstance "quad"
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        assert_eq!(scene.meshes.len(), 1, "one shared mesh");
        assert_eq!(scene.instances.len(), 2, "two instances");
        // Instance transforms differ.
        let a = scene.instances[0].transform.rows();
        let b = scene.instances[1].transform.rows();
        assert!((a[0][3] - -2.0).abs() < 1e-12);
        assert!((b[0][3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_inline_archive() {
        let input = r#"
            ArchiveBegin "ball"
                Sphere 1 -1 1 360
            ArchiveEnd
            WorldBegin
                Translate 0 0 5
                ReadArchive "ball"
                Translate 3 0 0
                ReadArchive "ball"
            WorldEnd
        "#;
        let requests = parse_rib(input).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        assert_eq!(scene.objects.len(), 2);
    }
}
