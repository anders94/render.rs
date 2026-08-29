//! Interprets the generalized RIB request stream into a Scene.
//!
//! Compliance policy (see COMPLIANCE.md): every request is accepted.
//! Implemented requests take effect; state-only requests are recorded in
//! the graphics state or passthrough dictionaries; everything else warns
//! once and is skipped.

use super::ast::{RibFile, RibRequest, RibValue};
use crate::geometry::{
    Cone, Cylinder, Disk, Hyperboloid, Intersectable, Paraboloid, Sphere, Torus, Triangle,
};
use crate::math::{Matrix4, Point3, Vec3};
use crate::scene::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
        }
    }
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
        }
    }

    pub fn build(mut self, requests: &RibFile) -> Result<Scene> {
        let mut objects: Vec<Arc<dyn Intersectable>> = Vec::new();
        let mut lights: Vec<Light> = Vec::new();
        let mut materials: Vec<Material> = Vec::new();

        for request in requests {
            // Motion blocks: use the first time sample, skip the rest.
            if self.in_motion && request.name != "MotionEnd" {
                if self.motion_sample_taken {
                    continue;
                }
                self.motion_sample_taken = true;
            }
            self.process(request, &mut objects, &mut lights, &mut materials)?;
        }

        let camera = Camera::new(self.width, self.height, self.fov);
        let mut scene = Scene::new(camera);
        scene.objects = objects;
        scene.lights = lights;
        scene.materials = materials;
        scene.pixel_samples = self.pixel_samples;
        Ok(scene)
    }

    fn process(
        &mut self,
        req: &RibRequest,
        objects: &mut Vec<Arc<dyn Intersectable>>,
        lights: &mut Vec<Light>,
        materials: &mut Vec<Material>,
    ) -> Result<()> {
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
                // "outside" is the default; anything else flips.
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
                    lights.push(light);
                }
            }

            // ---- geometry -------------------------------------------------
            "Sphere" => {
                if let (Some(r), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Sphere::new(
                        r, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Cylinder" => {
                if let (Some(r), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Cylinder::new(
                        r, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Cone" => {
                if let (Some(h), Some(r), Some(tm)) = (req.number(0), req.number(1), req.number(2))
                {
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Cone::new(
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
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Torus::new(
                        major, minor, phimin, phimax, tm, id,
                        self.transform_stack.current(),
                    )));
                }
            }
            "Disk" => {
                if let (Some(h), Some(r), Some(tm)) = (req.number(0), req.number(1), req.number(2))
                {
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Disk::new(
                        h, r, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Paraboloid" => {
                if let (Some(rmax), Some(zmin), Some(zmax), Some(tm)) =
                    (req.number(0), req.number(1), req.number(2), req.number(3))
                {
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Paraboloid::new(
                        rmax, zmin, zmax, tm, id, self.transform_stack.current(),
                    )));
                }
            }
            "Hyperboloid" => {
                let n: Vec<Option<f64>> = (0..7).map(|i| req.number(i)).collect();
                if n.iter().all(Option::is_some) {
                    let v: Vec<f64> = n.into_iter().map(Option::unwrap).collect();
                    let id = self.push_material(materials);
                    objects.push(Arc::new(Hyperboloid::new(
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
                        let id = self.push_material(materials);
                        let transform = self.transform_stack.current();
                        let vertex =
                            |i: usize| -> [f64; 3] { [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]] };
                        // Convex fan triangulation.
                        for i in 1..(p.len() / 3 - 1) {
                            objects.push(Arc::new(Triangle::new(
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
                if self.warned.insert(req.name.clone()) {
                    eprintln!(
                        "warning: RIB request '{}' not implemented yet (see COMPLIANCE.md); skipping",
                        req.name
                    );
                }
            }
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

    /// Create a material from the current attribute state; returns its id.
    fn push_material(&self, materials: &mut Vec<Material>) -> usize {
        let material = match self.state.surface.as_str() {
            "plastic" => Material::plastic(self.state.color, self.state.roughness),
            "metal" => Material::metal(self.state.color, self.state.roughness),
            _ => Material::matte(self.state.color),
        };
        materials.push(material);
        materials.len() - 1
    }
}

fn color_from(req: &RibRequest) -> Option<Vec3> {
    // Color 1 0 0  |  Color [1 0 0]
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
        // 4 quadrics + quad fan-triangulated into 2 triangles.
        assert_eq!(scene.objects.len(), 6);
    }

    #[test]
    fn test_concat_transform_matches_translate() {
        // RIB row-major premultiply: translation lives in elements 12..15.
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
        assert_eq!(scene.objects.len(), 1); // sphere lands, subdiv skipped
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
}
