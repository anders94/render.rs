//! Interprets the generalized RIB request stream into a Scene.
//!
//! Compliance policy (see COMPLIANCE.md): every request is accepted.
//! Implemented requests take effect; state-only requests are recorded in
//! the graphics state or passthrough dictionaries; everything else warns
//! once and is skipped.

use super::ast::{ParamList, RibFile, RibRequest, RibValue};
use crate::geometry::displace::DisplaceParams;
use crate::geometry::patches::{
    basis_by_name, tessellate_bicubic, tessellate_bilinear, tessellate_nurbs, Basis4, NuPatchDef,
    PatchMeshDef, BEZIER,
};
use crate::geometry::subdiv::SubdivCage;
use crate::geometry::curves::{dice_curve, CurveSet};
use crate::geometry::{
    Cone, Cylinder, Disk, Hyperboloid, Instance, Intersectable, Mesh, Paraboloid, Sphere, Torus,
    Triangle,
};
use crate::math::{Matrix4, Point3, Vec3};
use crate::scene::*;
use crate::texture::cache::Wrap;
use crate::texture::pattern::{BoundField, PInput, PatternNode, TextureRef};
use std::sync::Arc as StdArc;
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
    /// Modern material from a `Bxdf` request; overrides `Surface` mapping.
    bxdf: Option<PbrParams>,
    /// Pattern connections on the active Bxdf ("reference" params).
    bxdf_bindings: Vec<(BoundField, u32)>,
    /// Displacement applied to tessellated geometry (Displace request).
    displace: Option<DisplaceParams>,
    /// Transform-motion endpoints from a MotionBegin block: the composed
    /// transform at shutter open / close when the block closed.
    motion_t0: Option<Matrix4>,
    motion_t1: Option<Matrix4>,
    /// Hair material from Bxdf "PxrMarschnerHair".
    hair: Option<crate::raytracer::pt::hair::HairParams>,
    /// Interior medium bound to subsequent geometry.
    interior: Option<u32>,
    /// Attribute "identifier" "name": groups geometry under one id.
    identifier: Option<String>,
    basis_u: (Basis4, usize),
    basis_v: (Basis4, usize),
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
            bxdf: None,
            bxdf_bindings: Vec::new(),
            displace: None,
            motion_t0: None,
            motion_t1: None,
            hair: None,
            interior: None,
            identifier: None,
            basis_u: (BEZIER, 3),
            basis_v: (BEZIER, 3),
        }
    }
}

/// Everything the builder accumulates into the Scene.
#[derive(Default)]
struct SceneData {
    media: Vec<Medium>,
    atmosphere: Option<u32>,
    objects: Vec<Arc<dyn Intersectable>>,
    meshes: Vec<Mesh>,
    curve_sets: Vec<CurveSet>,
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
    /// Shared by every instance of this definition (deduplicated so a
    /// million ObjectInstances cost one material, not a million).
    material_id: usize,
}

pub struct SceneBuilder {
    width: u32,
    height: u32,
    fov: f64,
    projection: Projection,
    /// DepthOfField positional args: (fstop, focallength, focaldistance).
    depth_of_field: Option<(f64, f64, f64)>,
    pixel_filter: PixelFilter,
    shutter: (f64, f64),
    screen_window: Option<(f64, f64, f64, f64)>,
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
    /// Requests collected inside the open MotionBegin block.
    motion_requests: Vec<RibRequest>,
    /// Deformation endpoint "P" captured for the next mesh build.
    pending_deform_p: Option<Vec<f64>>,
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
    /// Pattern graph accumulated from `Pattern` requests (scene-global).
    pattern_nodes: Vec<PatternNode>,
    /// Pattern handle -> node index.
    pattern_handles: HashMap<String, u32>,
    /// identifier name -> object id, and the id counter.
    object_ids: HashMap<String, u32>,
    next_object_id: u32,
    id_manifest: std::collections::BTreeMap<u32, String>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            width: 640,
            height: 480,
            fov: 45.0,
            projection: Projection::Perspective,
            depth_of_field: None,
            pixel_filter: PixelFilter::Box { width: 1.0 },
            shutter: (0.0, 0.0),
            screen_window: None,
            pixel_samples: (1, 1),
            state: GraphicsState::default(),
            attribute_stack: Vec::new(),
            transform_stack: TransformStack::new(),
            coord_systems: HashMap::new(),
            declarations: HashMap::new(),
            passthrough: HashMap::new(),
            warned: HashSet::new(),
            in_motion: false,
            motion_requests: Vec::new(),
            pending_deform_p: None,
            base_dir: None,
            background: Vec3::zero(),
            archives: HashMap::new(),
            recording_archive: None,
            object_defs: HashMap::new(),
            defining_object: None,
            archive_depth: 0,
            pattern_nodes: Vec::new(),
            pattern_handles: HashMap::new(),
            object_ids: HashMap::new(),
            next_object_id: 1,
            id_manifest: std::collections::BTreeMap::new(),
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

        let mut camera = Camera::new(self.width, self.height, self.fov);
        camera.projection = self.projection;
        camera.filter = self.pixel_filter;
        camera.shutter = self.shutter;
        if let Some((fstop, focal_len, focal_dist)) = self.depth_of_field {
            if fstop.is_finite() && fstop > 0.0 {
                camera.lens_radius = focal_len / (2.0 * fstop);
                camera.focal_distance = focal_dist;
            }
        }
        if let Some((l, r, b, t)) = self.screen_window {
            camera.ortho_half = ((r - l).abs() * 0.5, (t - b).abs() * 0.5);
        }
        let mut scene = Scene::new(camera);
        scene.objects = data.objects;
        scene.meshes = data.meshes;
        scene.curve_sets = data.curve_sets;
        scene.instances = data.instances;
        scene.lights = data.lights;
        scene.materials = data.materials;
        scene.pixel_samples = self.pixel_samples;
        scene.background_color = self.background;
        scene.patterns = self.pattern_nodes;
        scene.media = data.media.clone();
        scene.atmosphere = data.atmosphere;
        scene.id_manifest = self.id_manifest.clone();
        scene.has_motion = scene.instances.iter().any(|i| i.transform1.is_some())
            || scene.meshes.iter().any(|m| m.positions1.is_some());
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
            // Motion blocks: collect the samples, resolved at MotionEnd.
            if self.in_motion && request.name != "MotionEnd" {
                self.motion_requests.push(request.clone());
                continue;
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
                self.motion_requests.clear();
            }
            "MotionEnd" => {
                self.in_motion = false;
                let block = std::mem::take(&mut self.motion_requests);
                self.finish_motion_block(block, data)?;
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
            // Participating media (roadmap Phase 10). Extension params:
            // sigma_a/sigma_s [rgb], g, and for heterogeneous clouds
            // density "fbm" + frequency/octaves/gain/lacunarity/coverage/
            // sharpness. Exterior is accepted but unimplemented.
            "Interior" | "Atmosphere" => {
                let params = req.params_from(1);
                let get3 = |name: &str, d: Vec3| {
                    params
                        .get_numbers(name)
                        .and_then(|v| (v.len() >= 3).then(|| Vec3::new(v[0], v[1], v[2])))
                        .unwrap_or(d)
                };
                let density = match params.get_string("density") {
                    Some("fbm") | Some("noise") => Some(DensityField::Fbm {
                        params: DisplaceParams {
                            amplitude: 1.0,
                            frequency: params.get_number("frequency").unwrap_or(0.3),
                            octaves: params.get_number("octaves").unwrap_or(5.0) as u32,
                            gain: params.get_number("gain").unwrap_or(0.55),
                            lacunarity: params.get_number("lacunarity").unwrap_or(2.0),
                            offset: [0.0; 3],
                        },
                        coverage: params.get_number("coverage").unwrap_or(0.5),
                        sharpness: params.get_number("sharpness").unwrap_or(4.0),
                    }),
                    Some(other) => {
                        self.warn_once(&format!(
                            "Interior density \"{other}\" not implemented; homogeneous"
                        ));
                        None
                    }
                    None => None,
                };
                let default_extent = if req.name == "Atmosphere" { 1200.0 } else { f64::INFINITY };
                let medium = Medium {
                    sigma_a: get3("sigma_a", Vec3::new(0.1, 0.1, 0.1)),
                    sigma_s: get3("sigma_s", Vec3::new(0.5, 0.5, 0.5)),
                    g: params.get_number("g").unwrap_or(0.0),
                    density,
                    emission: get3("emission", Vec3::zero()),
                    max_distance: params.get_number("maxdistance").unwrap_or(default_extent),
                };
                data.media.push(medium);
                let idx = (data.media.len() - 1) as u32;
                if req.name == "Atmosphere" {
                    data.atmosphere = Some(idx);
                } else {
                    self.state.interior = Some(idx);
                }
            }
            "Exterior" => {
                self.warn_once("Exterior media not implemented; ignoring");
            }

            // Procedural geometry: DelayedReadArchive loads eagerly (we
            // have no lazy loading yet — bounds ignored); RunProgram
            // spawns the generator and parses its stdout as RIB.
            "Procedural" => {
                let kind = req.string(0).unwrap_or("");
                let args: Vec<String> = match req.values.get(1) {
                    Some(RibValue::Strings(v)) => v.clone(),
                    Some(RibValue::String(s)) => vec![s.clone()],
                    _ => Vec::new(),
                };
                match kind {
                    "DelayedReadArchive" => {
                        self.warn_once(
                            "Procedural \"DelayedReadArchive\" loads eagerly (no lazy loading)",
                        );
                        if let Some(file) = args.first() {
                            self.read_archive(&file.clone(), data)?;
                        }
                    }
                    "RunProgram" => {
                        let Some(program) = args.first() else {
                            self.warn_once("Procedural \"RunProgram\" without a program; skipping");
                            return Ok(());
                        };
                        let prog_path = self.resource_path(program);
                        let mut cmd = std::process::Command::new(&prog_path);
                        if let Some(extra) = args.get(1) {
                            cmd.args(extra.split_whitespace());
                        }
                        if let Some(dir) = &self.base_dir {
                            cmd.current_dir(dir);
                        }
                        match cmd.output() {
                            Ok(out) if out.status.success() => {
                                match super::parse_rib_bytes(&out.stdout) {
                                    Ok(requests) => {
                                        self.archive_depth += 1;
                                        self.run(&requests, data)?;
                                        self.archive_depth -= 1;
                                    }
                                    Err(e) => self.warn_once(&format!(
                                        "RunProgram {program}: output failed to parse: {e:#}"
                                    )),
                                }
                            }
                            Ok(out) => self.warn_once(&format!(
                                "RunProgram {program} exited with {}",
                                out.status
                            )),
                            Err(e) => {
                                self.warn_once(&format!("RunProgram {program}: {e}"))
                            }
                        }
                    }
                    other => self.warn_once(&format!(
                        "Procedural \"{other}\" not implemented (DynamicLoad needs FFI); skipping"
                    )),
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
                    let placement1 = self.motion_endpoint(&placement);
                    for entry in entries {
                        data.instances.push(Instance::with_motion(
                            entry.mesh_id,
                            entry.material_id,
                            placement * entry.local_transform,
                            placement1.map(|p1| p1 * entry.local_transform),
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
                match req.string(0) {
                    Some("perspective") => {
                        self.projection = Projection::Perspective;
                        if let Some(fov) = req.params_from(1).get_number("fov") {
                            self.fov = fov;
                        }
                    }
                    Some("orthographic") => self.projection = Projection::Orthographic,
                    other => {
                        if let Some(name) = other {
                            self.warn_once(&format!(
                                "Projection \"{name}\" not implemented; using perspective"
                            ));
                        }
                    }
                }
            }
            "DepthOfField" => {
                if let (Some(fstop), Some(fl), Some(fd)) =
                    (req.number(0), req.number(1), req.number(2))
                {
                    self.depth_of_field = Some((fstop, fl, fd));
                }
            }
            "Shutter" => {
                if let (Some(open), Some(close)) = (req.number(0), req.number(1)) {
                    self.shutter = (open, close);
                }
            }
            "PixelFilter" => {
                if let Some(name) = req.string(0) {
                    let xw = req.number(1).unwrap_or(2.0);
                    let yw = req.number(2).unwrap_or(xw);
                    self.pixel_filter = PixelFilter::from_name(name, xw, yw);
                }
            }
            "ScreenWindow" => {
                if let (Some(l), Some(r), Some(b), Some(t)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    self.screen_window = Some((l, r, b, t));
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
                    // Attribute "identifier" "name" ["x"]: id-AOV grouping.
                    if req.name == "Attribute" && category == "identifier" {
                        if let Some(name) = req.params_from(1).get_string("name") {
                            self.state.identifier = Some(name.to_string());
                        }
                    }
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
            "Display" | "Clipping" | "ClippingPlane" | "CropWindow"
            | "FrameAspectRatio" | "PixelVariance" | "Exposure"
            | "Quantize" | "Hider" | "Integrator" | "TextureCoordinates"
            | "ShadingInterpolation" | "RelativeDetail" => {
                self.record(req);
            }
            "Basis" => {
                // Basis <name-or-matrix> step <name-or-matrix> step
                let parse = |v: Option<&RibValue>, step: Option<f64>| -> Option<(Basis4, usize)> {
                    match v {
                        Some(RibValue::String(name)) => {
                            let (m, default_step) = basis_by_name(name)?;
                            Some((m, step.map(|s| s as usize).unwrap_or(default_step)))
                        }
                        Some(other) => {
                            let nums = other.as_numbers()?;
                            if nums.len() != 16 { return None; }
                            let mut m = [[0.0; 4]; 4];
                            for r in 0..4 {
                                for c in 0..4 { m[r][c] = nums[r * 4 + c]; }
                            }
                            Some((m, step.map(|s| s as usize).unwrap_or(1)))
                        }
                        None => None,
                    }
                };
                if let Some(b) = parse(req.values.first(), req.number(1)) {
                    self.state.basis_u = b;
                }
                if let Some(b) = parse(req.values.get(2), req.number(3)) {
                    self.state.basis_v = b;
                }
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
            "Bxdf" => {
                let bxdf_type = req.string(0).unwrap_or("");
                if bxdf_type == "PxrMarschnerHair" {
                    let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
                    let params = req.params_from(params_start);
                    let hp =
                        crate::raytracer::pt::hair::HairParams::from_bxdf_params(&params);
                    self.state.hair = Some(hp);
                    self.state.bxdf = None;
                    self.state.bxdf_bindings = Vec::new();
                    return Ok(());
                }
                self.state.hair = None;
                if bxdf_type != "PxrSurface" {
                    self.warn_once(&format!(
                        "Bxdf \"{bxdf_type}\" not implemented; treating as PxrSurface"
                    ));
                }
                let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
                let params = req.params_from(params_start);
                self.state.bxdf = Some(PbrParams::from_bxdf_params(&params));
                // "reference <type> <param>" ["node:output"] connections.
                let mut bindings = Vec::new();
                for (token, value) in params.iter() {
                    if !token.split_whitespace().any(|w| w == "reference") {
                        continue;
                    }
                    let Some(param) = token.split_whitespace().last() else { continue };
                    let Some(target) = value.as_str() else { continue };
                    let handle = target.split(':').next().unwrap_or(target);
                    let Some(field) = BoundField::from_param(param) else {
                        self.warn_once(&format!(
                            "reference to unsupported Bxdf param \"{param}\"; ignoring"
                        ));
                        continue;
                    };
                    match self.pattern_handles.get(handle) {
                        Some(node) => bindings.push((field, *node)),
                        None => self.warn_once(&format!(
                            "Bxdf references unknown Pattern \"{handle}\"; ignoring"
                        )),
                    }
                }
                self.state.bxdf_bindings = bindings;
            }
            "Pattern" => {
                self.pattern_request(req);
            }
            "Light" => {
                self.build_light(req, data);
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

            // ---- curves & points ------------------------------------------
            "Curves" => {
                self.curves_request(req, data);
            }
            "Points" => {
                self.points_request(req, data);
            }

            // ---- meshes ---------------------------------------------------
            "PointsPolygons" => {
                self.points_polygons(req, None, data);
            }
            "SubdivisionMesh" | "HierarchicalSubdivisionMesh" => {
                if req.name == "HierarchicalSubdivisionMesh" {
                    self.warn_once(
                        "HierarchicalSubdivisionMesh treated as SubdivisionMesh (string args ignored)",
                    );
                }
                self.subdivision_mesh(req, data);
            }
            "PatchMesh" | "Patch" => {
                self.patch_mesh(req, data);
            }
            "NuPatch" => {
                self.nu_patch(req, data);
            }
            "GeneralPolygon" => {
                self.general_polygon(req, data);
            }
            "Displace" => {
                let name = req.string(0).unwrap_or("");
                if name != "noise" {
                    self.warn_once(&format!(
                        "Displace \"{name}\": only the built-in \"noise\" displacement exists until Phase 6; using it"
                    ));
                }
                let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
                let params = req.params_from(params_start);
                let mut d = DisplaceParams::default();
                if let Some(v) = params.get_number("amplitude") { d.amplitude = v; }
                if let Some(v) = params.get_number("frequency") { d.frequency = v; }
                if let Some(v) = params.get_number("octaves") { d.octaves = v as u32; }
                if let Some(v) = params.get_number("gain") { d.gain = v; }
                if let Some(v) = params.get_number("lacunarity") { d.lacunarity = v; }
                if let Some(v) = params.get_numbers("offset") {
                    if v.len() >= 3 { d.offset = [v[0], v[1], v[2]]; }
                }
                self.state.displace = Some(d);
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

        let positions1 = self.pending_deform_p.take().and_then(|p1| {
            (p1.len() == p.len()).then(|| {
                p1.chunks_exact(3)
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                    .collect::<Vec<_>>()
            })
        });
        let mesh = Mesh::with_motion(positions, indices, normals, st, positions1);
        self.add_tessellated_mesh(mesh, data);
    }

    /// `Curves "cubic"|"linear" [nvertices] "nonperiodic" "P" [...]` with
    /// "width" (root/tip taper) or "constantwidth". Cubic curves use the
    /// v Basis (RiSpec); periodic wrap is treated as nonperiodic.
    fn curves_request(&mut self, req: &RibRequest, data: &mut SceneData) {
        if self.reject_in_object_def("Curves") {
            return;
        }
        let curve_type = req.string(0).unwrap_or("cubic");
        let Some(nvertices) = req.values.get(1).and_then(RibValue::as_numbers) else {
            self.warn_once("Curves without [nvertices]; skipping");
            return;
        };
        let wrap = req.string(2).unwrap_or("nonperiodic");
        if wrap == "periodic" {
            self.warn_once("Curves \"periodic\" not supported; treating as nonperiodic");
        }
        let params = req.params_from(3);
        let Some(p) = params.get_numbers("P") else {
            self.warn_once("Curves without \"P\"; skipping");
            return;
        };
        let (width_root, width_tip) = match params.get_numbers("width") {
            Some(w) if !w.is_empty() => (w[0], *w.last().unwrap()),
            _ => {
                let cw = params.get_number("constantwidth").unwrap_or(0.01);
                (cw, cw)
            }
        };
        let cubic = if curve_type == "linear" {
            None
        } else {
            let (basis, step) = &self.state.basis_v;
            Some((*basis, *step))
        };
        let segs = ((8.0 / self.state.shading_rate.max(0.1)) as usize).clamp(2, 24);

        let mut p0 = Vec::new();
        let mut p1 = Vec::new();
        let mut v0 = Vec::new();
        let mut v1 = Vec::new();
        let mut cursor = 0usize;
        for &nv in nvertices {
            let nv = nv as usize;
            if (cursor + nv) * 3 > p.len() {
                break;
            }
            let ctrl: Vec<[f64; 3]> = (0..nv)
                .map(|i| {
                    let b = (cursor + i) * 3;
                    [p[b], p[b + 1], p[b + 2]]
                })
                .collect();
            dice_curve(
                &ctrl,
                cubic.as_ref().map(|(b, s)| (b, *s)),
                width_root,
                width_tip,
                segs,
                &mut p0,
                &mut p1,
                &mut v0,
                &mut v1,
            );
            cursor += nv;
        }
        if p0.is_empty() {
            return;
        }
        self.add_curve_set(CurveSet::new(p0, p1, v0, v1), data);
    }

    /// `Points "P" [...] "width" [...]` — particles as spheres.
    fn points_request(&mut self, req: &RibRequest, data: &mut SceneData) {
        if self.reject_in_object_def("Points") {
            return;
        }
        let params = req.params_from(0);
        let Some(p) = params.get_numbers("P") else {
            self.warn_once("Points without \"P\"; skipping");
            return;
        };
        let widths = params.get_numbers("width");
        let cw = params.get_number("constantwidth").unwrap_or(0.05);
        let count = p.len() / 3;
        let mut p0 = Vec::with_capacity(count);
        let mut p1 = Vec::with_capacity(count);
        let mut v0 = vec![0.0f32; count];
        let mut v1 = vec![1.0f32; count];
        for i in 0..count {
            let w = widths
                .and_then(|w| w.get(i).copied())
                .unwrap_or(cw);
            let e = [
                p[i * 3] as f32,
                p[i * 3 + 1] as f32,
                p[i * 3 + 2] as f32,
                (w * 0.5) as f32,
            ];
            p0.push(e);
            p1.push(e);
        }
        v0.iter_mut().for_each(|v| *v = 0.5);
        v1.iter_mut().for_each(|v| *v = 0.5);
        self.add_curve_set(CurveSet::new(p0, p1, v0, v1), data);
    }

    /// Shared sink for curve sets: material + placement (with motion).
    fn add_curve_set(&mut self, set: CurveSet, data: &mut SceneData) {
        let material_id = self.push_material(&mut data.materials);
        let set_id = data.curve_sets.len() as u32;
        let placement = self.transform_stack.current();
        let instance = Instance::new_curves(
            set_id,
            material_id,
            placement,
            self.motion_endpoint(&placement),
            &set,
        );
        data.curve_sets.push(set);
        data.instances.push(instance);
    }

    /// Dice density per patch span, from ShadingRate (smaller rate = finer).
    fn dice_segments(&self) -> usize {
        (16.0 / self.state.shading_rate.max(0.05)).clamp(2.0, 64.0) as usize
    }

    /// Uniform subdivision depth from ShadingRate.
    fn subdiv_levels(&self) -> u32 {
        let sr = self.state.shading_rate;
        if sr <= 0.15 {
            5
        } else if sr <= 0.35 {
            4
        } else if sr <= 1.5 {
            3
        } else if sr <= 6.0 {
            2
        } else {
            1
        }
    }

    /// Shared sink for all tessellated geometry: applies displacement,
    /// emissive state, object-definition capture, and instancing.
    fn add_tessellated_mesh(&mut self, mesh: Mesh, data: &mut SceneData) {
        let mesh = match &self.state.displace {
            Some(d) => crate::geometry::displace::displace_mesh(mesh, d),
            None => mesh,
        };
        let mesh_id = data.meshes.len() as u32;
        data.meshes.push(mesh);

        let mut material = self.make_material();
        if let Some((intensity, color)) = self.state.area_light {
            // Emissive meshes glow (BSDF hits); they are not yet sampleable
            // lights.
            material.emission = color * intensity;
        }

        if self.defining_object.is_some() {
            data.materials.push(material);
            let material_id = data.materials.len() - 1;
            let (_, entries) = self.defining_object.as_mut().unwrap();
            entries.push(ObjectDefEntry {
                mesh_id,
                local_transform: self.transform_stack.current(),
                material_id,
            });
        } else {
            data.materials.push(material);
            let material_id = data.materials.len() - 1;
            let placement = self.transform_stack.current();
            data.instances.push(Instance::with_motion(
                mesh_id,
                material_id,
                placement,
                self.motion_endpoint(&placement),
                &data.meshes[mesh_id as usize],
            ));
        }
    }

    /// SubdivisionMesh "scheme" [nverts] [verts] (["tags"] [nargs] [ints]
    /// [floats])? params — uniform Catmull-Clark with crease/corner/hole
    /// tags and interpolateboundary.
    fn subdivision_mesh(&mut self, req: &RibRequest, data: &mut SceneData) {
        let Some(scheme) = req.string(0) else { return };
        if scheme != "catmull-clark" {
            self.warn_once(&format!(
                "SubdivisionMesh scheme \"{scheme}\" treated as catmull-clark"
            ));
        }
        let Some(nverts) = req.values.get(1).and_then(RibValue::as_numbers) else { return };
        let Some(verts) = req.values.get(2).and_then(RibValue::as_numbers) else { return };

        // Optional tag block.
        let mut crease_edges = Vec::new();
        let mut corners = Vec::new();
        let mut holes = Vec::new();
        let mut interpolate_boundary = false;
        let params_start = if let Some(RibValue::Strings(tags)) = req.values.get(3) {
            let nargs = req.values.get(4).and_then(RibValue::as_numbers).unwrap_or(&[]);
            let intargs = req.values.get(5).and_then(RibValue::as_numbers).unwrap_or(&[]);
            let floatargs = req.values.get(6).and_then(RibValue::as_numbers).unwrap_or(&[]);
            let mut int_cursor = 0usize;
            let mut float_cursor = 0usize;
            for (ti, tag) in tags.iter().enumerate() {
                let ni = nargs.get(ti * 2).copied().unwrap_or(0.0) as usize;
                let nf = nargs.get(ti * 2 + 1).copied().unwrap_or(0.0) as usize;
                let ints = &intargs[int_cursor.min(intargs.len())
                    ..(int_cursor + ni).min(intargs.len())];
                let floats = &floatargs[float_cursor.min(floatargs.len())
                    ..(float_cursor + nf).min(floatargs.len())];
                match tag.as_str() {
                    "crease" => {
                        let sharp = floats.first().copied().unwrap_or(10.0);
                        for pair in ints.windows(2) {
                            crease_edges.push((pair[0] as u32, pair[1] as u32, sharp));
                        }
                    }
                    "corner" => {
                        for (i, v) in ints.iter().enumerate() {
                            let sharp = floats
                                .get(i)
                                .or(floats.first())
                                .copied()
                                .unwrap_or(10.0);
                            corners.push((*v as u32, sharp));
                        }
                    }
                    "hole" => {
                        holes.extend(ints.iter().map(|f| *f as u32));
                    }
                    "interpolateboundary" => interpolate_boundary = true,
                    other => self.warn_once(&format!(
                        "SubdivisionMesh tag \"{other}\" not supported; ignoring"
                    )),
                }
                int_cursor += ni;
                float_cursor += nf;
            }
            7
        } else {
            3
        };

        let Some(p) = req.params_from(params_start).get_numbers("P") else {
            self.warn_once("SubdivisionMesh without \"P\"; skipping");
            return;
        };
        let positions: Vec<[f64; 3]> = p.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mut faces = Vec::new();
        let mut cursor = 0usize;
        for &nv in nverts {
            let nv = nv as usize;
            if cursor + nv > verts.len() {
                break;
            }
            faces.push(verts[cursor..cursor + nv].iter().map(|v| *v as u32).collect());
            cursor += nv;
        }
        let cage = SubdivCage {
            positions,
            faces,
            crease_edges,
            corners,
            holes,
            interpolate_boundary,
        };
        let mesh = cage.tessellate(self.subdiv_levels());
        self.add_tessellated_mesh(mesh, data);
    }

    /// PatchMesh "type" nu uwrap nv vwrap params / Patch "type" params.
    fn patch_mesh(&mut self, req: &RibRequest, data: &mut SceneData) {
        let Some(ptype) = req.string(0) else { return };
        let (nu, u_wrap, nv, v_wrap, params_start) = if req.name == "Patch" {
            let n = if ptype == "bilinear" { 2 } else { 4 };
            (n, false, n, false, 1)
        } else {
            let (Some(nu), Some(nv)) = (req.number(1), req.number(3)) else { return };
            (
                nu as usize,
                req.string(2) == Some("periodic"),
                nv as usize,
                req.string(4) == Some("periodic"),
                5,
            )
        };
        let Some(points) = req.params_from(params_start).get_numbers("P") else {
            self.warn_once(&format!("{} without \"P\"; skipping", req.name));
            return;
        };
        let def = PatchMeshDef { points, nu, nv, u_wrap, v_wrap };
        let segs = self.dice_segments();
        let mesh = match ptype {
            "bilinear" => tessellate_bilinear(&def, segs),
            "bicubic" => {
                let (bu, ustep) = self.state.basis_u;
                let (bv, vstep) = self.state.basis_v;
                tessellate_bicubic(&def, &bu, ustep, &bv, vstep, segs)
            }
            other => {
                self.warn_once(&format!("{} type \"{other}\" not supported", req.name));
                None
            }
        };
        match mesh {
            Some(m) => self.add_tessellated_mesh(m, data),
            None => self.warn_once(&format!("{}: invalid control grid; skipping", req.name)),
        }
    }

    /// NuPatch nu uorder [uknot] umin umax nv vorder [vknot] vmin vmax params.
    fn nu_patch(&mut self, req: &RibRequest, data: &mut SceneData) {
        let nums = |i: usize| req.number(i);
        let arr = |i: usize| req.values.get(i).and_then(RibValue::as_numbers);
        let (Some(nu), Some(uorder), Some(uknot), Some(umin), Some(umax)) =
            (nums(0), nums(1), arr(2), nums(3), nums(4))
        else {
            return;
        };
        let (Some(nv), Some(vorder), Some(vknot), Some(vmin), Some(vmax)) =
            (nums(5), nums(6), arr(7), nums(8), nums(9))
        else {
            return;
        };
        let params = req.params_from(10);
        let (points, rational) = match (params.get_numbers("Pw"), params.get_numbers("P")) {
            (Some(pw), _) => (pw, true),
            (None, Some(p)) => (p, false),
            _ => {
                self.warn_once("NuPatch without \"P\"/\"Pw\"; skipping");
                return;
            }
        };
        if params.get("trimcurve").is_some() {
            self.warn_once("NuPatch trim curves not supported yet; rendering untrimmed");
        }
        let def = NuPatchDef {
            nu: nu as usize,
            uorder: uorder as usize,
            uknot,
            umin,
            umax,
            nv: nv as usize,
            vorder: vorder as usize,
            vknot,
            vmin,
            vmax,
            points,
            rational,
        };
        let base = self.dice_segments();
        let segs_u = (base * (def.nu.saturating_sub(def.uorder) + 1)).clamp(4, 128);
        let segs_v = (base * (def.nv.saturating_sub(def.vorder) + 1)).clamp(4, 128);
        match tessellate_nurbs(&def, segs_u, segs_v) {
            Some(m) => self.add_tessellated_mesh(m, data),
            None => self.warn_once("NuPatch: invalid control data; skipping"),
        }
    }

    /// GeneralPolygon [loop nverts] "P" — ear-clipped, holes bridged.
    fn general_polygon(&mut self, req: &RibRequest, data: &mut SceneData) {
        let Some(nloops) = req.values.first().and_then(RibValue::as_numbers) else { return };
        let Some(p) = req.params_from(1).get_numbers("P") else {
            self.warn_once("GeneralPolygon without \"P\"; skipping");
            return;
        };
        let all: Vec<[f64; 3]> = p.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let mut loops = Vec::new();
        let mut cursor = 0usize;
        for &n in nloops {
            let n = n as usize;
            if cursor + n > all.len() {
                break;
            }
            loops.push(all[cursor..cursor + n].to_vec());
            cursor += n;
        }
        if loops.is_empty() || loops[0].len() < 3 {
            return;
        }
        match crate::geometry::earclip::triangulate_with_holes(&loops) {
            Some((positions, indices)) => {
                let positions: Vec<[f32; 3]> = positions
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                    .collect();
                let normals = crate::geometry::subdiv::smooth_normals(&positions, &indices);
                let mesh = Mesh::new(positions, indices, Some(normals), None);
                self.add_tessellated_mesh(mesh, data);
            }
            None => self.warn_once("GeneralPolygon: triangulation failed; skipping"),
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
        match std::fs::read(&path) {
            Ok(content) => match super::parse_rib_bytes(&content) {
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

    /// Modern `Light "PxrX" "handle" params...` — the light's shape comes
    /// from the current transform (unit primitives in light space, PRMan
    /// style), and area lights get matching emissive geometry so BSDF rays
    /// see them (with MIS via material.area_light).
    fn build_light(&mut self, req: &RibRequest, data: &mut SceneData) {
        let Some(light_type) = req.string(0) else { return };
        let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
        let params = req.params_from(params_start);
        let intensity = params.get_number("intensity").unwrap_or(1.0);
        let exposure = params.get_number("exposure").unwrap_or(0.0);
        let scale = intensity * exposure.exp2();
        let color = params
            .get_numbers("lightColor")
            .and_then(|v| (v.len() >= 3).then(|| Vec3::new(v[0], v[1], v[2])))
            .unwrap_or(Vec3::one());
        let xf = self.transform_stack.current();
        let origin = xf.transform_point(&Point3::new(0.0, 0.0, 0.0));

        // Emissive stand-in material for area lights: black lobes, pure
        // emitter, tagged with the light index for MIS.
        let mut emitter = Material::matte(Vec3::zero());
        emitter.emission = color * scale;
        let light_idx = data.lights.len();
        emitter.area_light = Some(light_idx);

        match light_type {
            "PxrRectLight" => {
                // Unit square in light-space XY.
                let corner = xf.transform_point(&Point3::new(-0.5, -0.5, 0.0));
                let e1 = xf.transform_vec(&Vec3::new(1.0, 0.0, 0.0));
                let e2 = xf.transform_vec(&Vec3::new(0.0, 1.0, 0.0));
                data.lights.push(Light::rect(corner, e1, e2, scale, color));
                data.materials.push(emitter);
                let id = data.materials.len() - 1;
                let v = |x: f64, y: f64| [x, y, 0.0];
                data.objects.push(StdArc::new(Triangle::new(
                    v(-0.5, -0.5),
                    v(0.5, -0.5),
                    v(0.5, 0.5),
                    id,
                    xf,
                )));
                data.objects.push(StdArc::new(Triangle::new(
                    v(-0.5, -0.5),
                    v(0.5, 0.5),
                    v(-0.5, 0.5),
                    id,
                    xf,
                )));
            }
            "PxrSphereLight" => {
                let radius = xf.transform_vec(&Vec3::new(0.5, 0.0, 0.0)).length();
                data.lights.push(Light::sphere_area(origin, radius, scale, color));
                data.materials.push(emitter);
                let id = data.materials.len() - 1;
                data.objects
                    .push(StdArc::new(Sphere::new(0.5, -0.5, 0.5, 360.0, id, xf)));
            }
            "PxrDiskLight" => {
                let e1 = xf.transform_vec(&Vec3::new(0.5, 0.0, 0.0));
                let e2 = xf.transform_vec(&Vec3::new(0.0, 0.5, 0.0));
                data.lights.push(Light::disk_area(origin, e1, e2, scale, color));
                data.materials.push(emitter);
                let id = data.materials.len() - 1;
                data.objects
                    .push(StdArc::new(crate::geometry::Disk::new(0.0, 0.5, 360.0, id, xf)));
            }
            "PxrDistantLight" => {
                let direction = xf.transform_vec(&Vec3::new(0.0, 0.0, -1.0)).normalize();
                let angle_deg = params.get_number("angleExtent").unwrap_or(0.53);
                let angular_radius = (angle_deg * 0.5).to_radians();
                data.lights
                    .push(Light::distant_soft(direction, angular_radius, scale, color));
            }
            "PxrDomeLight" => {
                let env = params.get_string("lightColorMap").and_then(|file| {
                    let path = match &self.base_dir {
                        Some(dir) => dir.join(file),
                        None => PathBuf::from(file),
                    };
                    match crate::scene::EnvMap::load(&path) {
                        Ok(map) => Some(StdArc::new(map)),
                        Err(e) => {
                            self.warn_once(&format!("PxrDomeLight map: {e:#}"));
                            None
                        }
                    }
                });
                data.lights.push(Light::dome(scale, color, env));
            }
            other => {
                self.warn_once(&format!(
                    "Light \"{other}\" not implemented (see COMPLIANCE.md); skipping"
                ));
            }
        }
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

    /// Resolve a closed MotionBegin block. Transform blocks record shutter
    /// open/close transforms (applied to geometry that follows);
    /// deformation blocks capture the last sample's "P" as the endpoint.
    /// Everything else takes the first sample and warns.
    fn finish_motion_block(
        &mut self,
        block: Vec<RibRequest>,
        data: &mut SceneData,
    ) -> Result<()> {
        let (Some(first), Some(last)) = (block.first(), block.last()) else {
            return Ok(());
        };
        const TRANSFORM_REQS: [&str; 6] =
            ["Translate", "Rotate", "Scale", "ConcatTransform", "Transform", "Identity"];
        if block.iter().all(|r| TRANSFORM_REQS.contains(&r.name.as_str())) {
            let base = self.transform_stack.current();
            let first = first.clone();
            let last = last.clone();
            self.process(&first, data)?;
            let t0 = self.transform_stack.current();
            if block.len() > 1 {
                self.transform_stack.set(base);
                self.process(&last, data)?;
                let t1 = self.transform_stack.current();
                self.transform_stack.set(t0);
                self.state.motion_t0 = Some(t0);
                self.state.motion_t1 = Some(t1);
            }
            return Ok(());
        }
        if block.iter().all(|r| r.name == first.name)
            && matches!(first.name.as_str(), "PointsPolygons" | "PointsGeneralPolygons")
        {
            if block.len() > 1 {
                self.pending_deform_p =
                    last.params_from(2).get_numbers("P").map(|p| p.to_vec());
            }
            let first = first.clone();
            self.process(&first, data)?;
            self.pending_deform_p = None;
            return Ok(());
        }
        self.warn_once(&format!(
            "MotionBegin over \"{}\" not supported (transforms and \
             PointsPolygons deformation only); using the first sample",
            first.name
        ));
        let first = first.clone();
        self.process(&first, data)
    }

    /// Shutter-close transform for geometry created now: the motion
    /// block's T1 endpoint with any post-block transforms composed on.
    fn motion_endpoint(&self, current: &Matrix4) -> Option<Matrix4> {
        let t0 = self.state.motion_t0?;
        let t1 = self.state.motion_t1?;
        let post = t0.inverse()? * *current;
        Some(t1 * post)
    }

    /// Resolve a resource filename against the RIB's directory.
    fn resource_path(&self, name: &str) -> PathBuf {
        match &self.base_dir {
            Some(dir) if !name.starts_with('/') => dir.join(name),
            _ => PathBuf::from(name),
        }
    }

    /// Open a texture through the global cache; `<UDIM>` filenames scan
    /// for tile files 1001-1100.
    fn open_texture(&mut self, filename: &str) -> TextureRef {
        if filename.is_empty() {
            self.warn_once("Pattern texture without \"filename\"; rendering magenta");
            return TextureRef::Missing;
        }
        if filename.contains("<UDIM>") {
            let mut tiles = HashMap::new();
            for tile in 1001u16..=1100 {
                let path = self.resource_path(&filename.replace("<UDIM>", &tile.to_string()));
                if !path.exists() {
                    continue;
                }
                match crate::texture::global_cache().open(&path) {
                    Ok(id) => {
                        tiles.insert(tile, id);
                    }
                    Err(e) => self.warn_once(&format!("texture {}: {e}", path.display())),
                }
            }
            if tiles.is_empty() {
                self.warn_once(&format!("no UDIM tiles found for \"{filename}\""));
                return TextureRef::Missing;
            }
            return TextureRef::Udim(tiles);
        }
        let path = self.resource_path(filename);
        match crate::texture::global_cache().open(&path) {
            Ok(id) => TextureRef::Single(id),
            Err(e) => {
                self.warn_once(&format!("texture {}: {e}; rendering magenta", path.display()));
                TextureRef::Missing
            }
        }
    }

    /// A pattern parameter that may be a constant or a reference to an
    /// earlier node ("reference color mix" ["noise:resultRGB"]).
    fn pattern_input(&self, params: &ParamList<'_>, name: &str, default: Vec3) -> PInput {
        for (token, value) in params.iter() {
            let mut words = token.split_whitespace();
            let is_ref = token.split_whitespace().any(|w| w == "reference");
            if is_ref && words.next_back() == Some(name) {
                if let Some(target) = value.as_str() {
                    let handle = target.split(':').next().unwrap_or(target);
                    if let Some(node) = self.pattern_handles.get(handle) {
                        return PInput::Node(*node);
                    }
                }
            }
        }
        match params.get_numbers(name) {
            Some(v) if v.len() >= 3 => PInput::Const(Vec3::new(v[0], v[1], v[2])),
            Some(v) if !v.is_empty() => PInput::Const(Vec3::new(v[0], v[0], v[0])),
            _ => PInput::Const(default),
        }
    }

    /// `Pattern "type" "handle" params...` — build a graph node.
    fn pattern_request(&mut self, req: &RibRequest) {
        let ptype = req.string(0).unwrap_or("");
        let params_start = if req.values.len() % 2 == 0 { 2 } else { 1 };
        let handle = if params_start == 2 { req.string(1) } else { None };
        let Some(handle) = handle.map(str::to_string) else {
            self.warn_once("Pattern without a handle; skipping");
            return;
        };
        let params = req.params_from(params_start);
        let color = |name: &str, default: Vec3| -> Vec3 {
            params
                .get_numbers(name)
                .and_then(|v| (v.len() >= 3).then(|| Vec3::new(v[0], v[1], v[2])))
                .unwrap_or(default)
        };
        let node = match ptype {
            "PxrTexture" | "texture" => {
                let filename = params.get_string("filename").unwrap_or("").to_string();
                let tex = self.open_texture(&filename);
                PatternNode::Texture {
                    tex,
                    wrap: Wrap::from_name(params.get_string("wrapMode").unwrap_or("periodic")),
                    scale: [
                        params.get_number("scaleS").unwrap_or(1.0),
                        params.get_number("scaleT").unwrap_or(1.0),
                    ],
                }
            }
            "PxrChecker" | "checker" => PatternNode::Checker {
                color_a: color("colorA", Vec3::one()),
                color_b: color("colorB", Vec3::zero()),
                scale: [
                    params.get_number("sScale").unwrap_or(8.0),
                    params.get_number("tScale").unwrap_or(8.0),
                ],
            },
            "PxrFractal" | "fractal" | "noise" => PatternNode::Fractal {
                frequency: params.get_number("frequency").unwrap_or(1.0),
                octaves: params.get_number("layers").unwrap_or(4.0) as u32,
                gain: params.get_number("gain").unwrap_or(0.5),
                lacunarity: params.get_number("lacunarity").unwrap_or(2.0),
            },
            "PxrMix" | "mix" => PatternNode::Mix {
                color1: self.pattern_input(&params, "color1", Vec3::zero()),
                color2: self.pattern_input(&params, "color2", Vec3::one()),
                mix: self.pattern_input(&params, "mix", Vec3::zero()),
            },
            "PxrColorCorrect" | "colorCorrect" => PatternNode::ColorCorrect {
                input: self.pattern_input(&params, "inputRGB", Vec3::new(0.5, 0.5, 0.5)),
                gain: color("gain", Vec3::one()),
                offset: color("offset", Vec3::zero()),
                gamma: params.get_number("gamma").unwrap_or(1.0),
                saturation: params.get_number("saturation").unwrap_or(1.0),
            },
            "PxrRamp" | "ramp" => {
                let positions = params
                    .get_numbers("positions")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![0.0, 1.0]);
                let colors = params
                    .get_numbers("colors")
                    .map(|v| v.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect())
                    .unwrap_or_else(|| vec![Vec3::zero(), Vec3::one()]);
                PatternNode::Ramp {
                    positions,
                    colors,
                    use_t: params.get_string("axis") == Some("t"),
                }
            }
            "triplanar" => {
                let filename = params.get_string("filename").unwrap_or("").to_string();
                let tex = self.open_texture(&filename);
                PatternNode::Triplanar {
                    tex,
                    wrap: Wrap::from_name(params.get_string("wrapMode").unwrap_or("periodic")),
                    frequency: params.get_number("frequency").unwrap_or(1.0),
                }
            }
            other => {
                self.warn_once(&format!("Pattern \"{other}\" not implemented; skipping"));
                return;
            }
        };
        let index = self.pattern_nodes.len() as u32;
        self.pattern_nodes.push(node);
        self.pattern_handles.insert(handle, index);
    }

    /// Material from the current attribute state.
    fn make_material(&mut self) -> Material {
        if let Some(hp) = &self.state.hair {
            // Whitted fallback: a plain brown matte so hair still shows up
            // outside the path tracer.
            let mut m = Material::matte(Vec3::new(0.35, 0.22, 0.12));
            m.hair = Some(hp.clone());
            m.interior = self.state.interior;
            m.id = self.assign_object_id();
            return m;
        }
        if let Some(pbr) = &self.state.bxdf {
            // Whitted fallback terms approximated from the lobe params.
            let mut m = Material::matte(pbr.diffuse_color);
            m.pbr = pbr.clone();
            m.emission = Vec3::zero();
            m.pattern_bindings = self.state.bxdf_bindings.clone();
            m.interior = self.state.interior;
            m.id = self.assign_object_id();
            return m;
        }
        let mut m = match self.state.surface.as_str() {
            "plastic" => Material::plastic(self.state.color, self.state.roughness),
            "metal" => Material::metal(self.state.color, self.state.roughness),
            _ => Material::matte(self.state.color),
        };
        m.interior = self.state.interior;
        m.id = self.assign_object_id();
        m
    }

    /// Create a material from the current attribute state; returns its id.
    fn push_material(&mut self, materials: &mut Vec<Material>) -> usize {
        materials.push(self.make_material());
        materials.len() - 1
    }

    /// Object id for the id AOV: identifier-named geometry shares an id;
    /// unnamed geometry gets a fresh auto id.
    fn assign_object_id(&mut self) -> u32 {
        let name = self
            .state
            .identifier
            .clone()
            .unwrap_or_else(|| format!("object_{}", self.next_object_id));
        if let Some(id) = self.object_ids.get(&name) {
            return *id;
        }
        let id = self.next_object_id;
        self.next_object_id += 1;
        self.object_ids.insert(name.clone(), id);
        self.id_manifest.insert(id, name);
        id
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
    fn motion_blocks_record_endpoints() {
        let input = r#"
            Format 64 64 1.0
            Shutter 0 1
            WorldBegin
                AttributeBegin
                    MotionBegin [0 1]
                        Translate 0 0 5
                        Translate 2 0 5
                    MotionEnd
                    PointsPolygons [4] [0 1 2 3]
                        "P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]
                AttributeEnd
                AttributeBegin
                    Translate 0 0 8
                    MotionBegin [0 1]
                        PointsPolygons [4] [0 1 2 3] "P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]
                        PointsPolygons [4] [0 1 2 3] "P" [-1 -1 2  1 -1 2  1 1 2  -1 1 2]
                    MotionEnd
                AttributeEnd
            WorldEnd
        "#;
        let scene = SceneBuilder::new().build(&parse_rib(input).unwrap()).unwrap();
        assert!(scene.has_motion);
        assert_eq!(scene.instances.len(), 2);
        // Transform motion: endpoint shifted +2 in x.
        let t1 = scene.instances[0].transform1.expect("transform motion");
        let p0 = scene.instances[0]
            .transform
            .transform_point(&Point3::new(0.0, 0.0, 0.0));
        let p1 = t1.transform_point(&Point3::new(0.0, 0.0, 0.0));
        assert!((p1.x - p0.x - 2.0).abs() < 1e-9, "endpoint {p1:?}");
        // Deformation motion: second mesh carries positions1 (+2 in z).
        let mesh = &scene.meshes[1];
        let d = mesh.positions1.as_ref().expect("deform endpoint");
        assert!((d[0][2] - mesh.positions[0][2] - 2.0).abs() < 1e-6);
        // World bounds cover both endpoints.
        assert!(scene.instances[0].world_bounds.max.x > 2.9);
    }

    #[test]
    fn pattern_graph_binds_to_bxdf() {
        let input = r#"
            Format 64 64 1.0
            WorldBegin
                Pattern "PxrChecker" "check" "colorA" [1 0 0] "colorB" [0 0 1] "sScale" [4] "tScale" [4]
                Pattern "PxrMix" "blend" "reference color color1" ["check:resultRGB"] "color2" [0 1 0] "mix" [0.25]
                Bxdf "PxrSurface" "mat" "reference color diffuseColor" ["blend:resultRGB"] "specularIor" [1]
                Sphere 1 -1 1 360
            WorldEnd
        "#;
        let scene = SceneBuilder::new().build(&parse_rib(input).unwrap()).unwrap();
        assert_eq!(scene.patterns.len(), 2);
        let material = scene.materials.last().unwrap();
        assert_eq!(material.pattern_bindings.len(), 1);
        let (field, node) = material.pattern_bindings[0];
        assert_eq!(field, crate::texture::pattern::BoundField::DiffuseColor);
        assert_eq!(node, 1);
        // Evaluate the bound graph: checker cell (0,0) red, mixed 25%
        // toward green.
        let ctx = crate::texture::pattern::ShadeCtx {
            st: [0.1, 0.1],
            p: [0.0; 3],
            n: [0.0, 0.0, 1.0],
            footprint: 0.0,
        };
        let pbr = material.resolved_pbr(&scene.patterns, &ctx);
        assert!((pbr.diffuse_color.x - 0.75).abs() < 1e-9, "{:?}", pbr.diffuse_color);
        assert!((pbr.diffuse_color.y - 0.25).abs() < 1e-9);
    }

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
