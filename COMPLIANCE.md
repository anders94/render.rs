# RIB Compliance Matrix

"Fully compliant" means every request is **accepted**: implemented,
recorded-for-later, or deliberately skipped with a warning — never a parse
error. Status values: ✅ implemented · 🟡 parsed+stored (no render effect
yet) · ⏭ parse-and-warn, planned (phase noted) · 🚫 parse-and-warn,
skipped forever.

## Structure & blocks
| Request | Status | Notes |
|---|---|---|
| WorldBegin/WorldEnd | ✅ | |
| FrameBegin/FrameEnd | 🟡 | treated as grouping only |
| AttributeBegin/AttributeEnd | ✅ | full graphics-state save/restore |
| TransformBegin/TransformEnd | ✅ | |
| MotionBegin/MotionEnd | ✅ | transform motion (lerped endpoints, any prim via instances) + PointsPolygons deformation; other motion types warn and use the first sample |
| ObjectBegin/ObjectEnd/ObjectInstance | ✅ | mesh geometry instanced via TLAS; quadrics inside blocks warn+skip |
| SolidBegin/SolidEnd (CSG) | 🚫 | PRMan RIS barely supports it either |
| IfBegin/ElseIf/Else/IfEnd | 🟡 | condition ignored, body processed |
| version | ✅ | accepted and ignored |

## Options & camera
| Request | Status | Notes |
|---|---|---|
| Format | ✅ | |
| Projection | ✅ | perspective (fov) and orthographic (ScreenWindow sets extent) |
| Clipping / ClippingPlane | 🟡 | |
| CropWindow / ScreenWindow / FrameAspectRatio | 🟡 / ✅ / 🟡 | ScreenWindow drives the orthographic extent |
| Shutter | ✅ | enables per-ray shutter times over motion endpoints |
| PixelSamples | ✅ | |
| PixelFilter / PixelVariance | ✅ / 🟡 | box/triangle/gaussian via filter importance sampling; adaptive variance stopping via the --adaptive CLI flag (CPU) |
| DepthOfField | ✅ | thin-lens bokeh: fstop / focallength / focaldistance |
| Exposure / Quantize | 🟡 | |
| Display | 🟡 | filename/driver recorded; CLI overrides |
| Hider / Integrator | ⏭ P1/P4 | |
| Option (generic) | ✅ | passthrough dictionary; extension: `Option "background" "color" [r g b]` sets the miss color |
| Declare | ✅ | |

## Transforms & spaces
| Request | Status | Notes |
|---|---|---|
| Identity / Transform / ConcatTransform | ✅ | |
| Translate / Rotate / Scale | ✅ | |
| Skew | 🟡 | |
| Perspective | 🚫 | modeling-time perspective; no modern use |
| CoordinateSystem / ScopedCoordinateSystem | ✅ | named-space registry |
| CoordSysTransform | 🟡 | |

## Attributes & shading state
| Request | Status | Notes |
|---|---|---|
| Color / Opacity | ✅ / 🟡 | use PxrSurface "presence" for cutouts (transparent shadows) |
| Surface | ✅ | legacy matte/plastic/metal mapping |
| Bxdf "PxrSurface" | ✅ | full lobe stack: Oren-Nayar diffuse, GGX/VNDF specular (F0/F90 or IOR), clearcoat, fuzz, rough glass with refraction, glow, presence |
| Light "PxrRect/Sphere/Disk/Distant/DomeLight" | ✅ | shapes from the current transform; dome takes "lightColorMap" HDRI with 2D-CDF importance sampling |
| Pattern | ✅ | PxrTexture (UDIM `<UDIM>`), PxrChecker, PxrFractal, PxrMix, PxrColorCorrect, PxrRamp, triplanar (extension); `reference` param connections into Bxdf; CPU eval + Metal MSL codegen |
| Displace | ✅ | extension: `Displace "noise" "amplitude"/"frequency"/"octaves"/"gain"/"lacunarity"/"offset"` — fBm displacement at dice time; pattern-driven displacement arrives with P6 |
| LightSource / AreaLightSource | ✅ / ✅ | point+distant; quad polygons under AreaLightSource become sampleable rect lights (path integrator) |
| Illuminate (light linking) | ⏭ P6 | |
| Atmosphere / Interior / Exterior | ✅ / ✅ / ⏭ | extension params sigma_a/sigma_s/g/emission/maxdistance + density "fbm" (delta/ratio tracking); Interior binds to hulls (invisible when lobeless) and glass; Exterior warns |
| Displacement (RSL-era) | 🚫 | RSL-only; `Displace` is the modern path |
| Orientation / ReverseOrientation / Sides | 🟡 | honored for meshes at P2 |
| Basis | ✅ | named (bezier/b-spline/catmull-rom/hermite/power) or custom 4x4 matrices + steps |
| ShadingRate / ShadingInterpolation | ✅ / 🟡 | drives dice density: subdiv depth 1-5 and patch segments 2-64 |
| Attribute (generic) | ✅ | passthrough dictionary |
| Detail / DetailRange / GeometricApproximation | 🚫 | always highest detail |
| TextureCoordinates | 🟡 | |

## Geometry
| Request | Status | Notes |
|---|---|---|
| Sphere / Cylinder / Cone | ✅ | |
| Torus / Disk / Paraboloid / Hyperboloid | ✅ | P0 |
| Polygon | ✅ | P0, convex fan triangulation |
| PointsPolygons / PointsGeneralPolygons | ✅ | fan triangulation, vertex N carried; general-polygon holes warn (outer loops used) |
| GeneralPolygon | ✅ | ear-clipping with hole bridging |
| PatchMesh / Patch | ✅ | bilinear + bicubic with basis matrices; crack-free shared-grid dicing |
| NuPatch / TrimCurve | ✅ / ⏭ | NURBS (rational Pw or P) via Cox-de Boor; trims warn + render untrimmed |
| SubdivisionMesh | ✅ | uniform Catmull-Clark; crease/corner (semi-sharp decay), hole, interpolateboundary tags |
| HierarchicalSubdivisionMesh | 🟡 | treated as SubdivisionMesh (string args ignored) |
| Curves | ✅ | linear + cubic (v Basis) diced to rounded-cone segments; "width"/"constantwidth" with root-to-tip taper; periodic warns; Bxdf "PxrMarschnerHair" shades them |
| Points | ✅ | sphere particles ("width" per point or "constantwidth") |
| Blobby | 🚫 | revisit only if a real scene needs it |
| Volume | ⏭ P10 | |
| Procedural (DelayedReadArchive/RunProgram/DynamicLoad) | ✅ / ✅ / ⏭ | DelayedReadArchive loads eagerly (bounds ignored); RunProgram spawns the generator and parses its stdout; DynamicLoad needs FFI and warns |
| Geometry (renderer-specific) | 🚫 | |

## Archives & external
| Request | Status | Notes |
|---|---|---|
| ReadArchive | ✅ | file (relative to RIB dir) or inline archive; depth-capped |
| ArchiveBegin/ArchiveEnd | ✅ | inline archives |
| MakeTexture/MakeLatLongEnvironment/etc. | 🟡 | `render txmake in.png out.tex` CLI converts any image to the tiled-mip .tex format (in-RIB MakeTexture requests still warn) |
| Binary RIB encoding | ✅ | RISpec App. C decode (mixed ASCII+binary); `render catrib [--binary]` converts both ways; ~45% smaller archives |

## RSL-era shading (all skipped forever)
`ShadingModel`, `Imager`, RSL `Surface`/`Displacement`/`Volume` shader
compilation, `MakeShadow`, shadow/environment map requests — 🚫. The modern
path-traced pipeline (Bxdf/Pattern/Integrator + OSL) replaces all of it,
matching PRMan's own evolution.
