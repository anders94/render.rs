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
| MotionBegin/MotionEnd | ⏭ P7 | first sample used until then |
| ObjectBegin/ObjectEnd/ObjectInstance | ✅ | mesh geometry instanced via TLAS; quadrics inside blocks warn+skip |
| SolidBegin/SolidEnd (CSG) | 🚫 | PRMan RIS barely supports it either |
| IfBegin/ElseIf/Else/IfEnd | 🟡 | condition ignored, body processed |
| version | ✅ | accepted and ignored |

## Options & camera
| Request | Status | Notes |
|---|---|---|
| Format | ✅ | |
| Projection | ✅ | perspective fov; orthographic ⏭ P7 |
| Clipping / ClippingPlane | 🟡 | |
| CropWindow / ScreenWindow / FrameAspectRatio | 🟡 | render effect ⏭ P7 |
| Shutter | 🟡 | ⏭ P7 motion blur |
| PixelSamples | ✅ | |
| PixelFilter / PixelVariance | 🟡 | filters ⏭ P7 |
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
| Atmosphere / Interior / Exterior | ⏭ P10 | |
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
| Curves | ⏭ P8 | |
| Points | ⏭ P8 | |
| Blobby | 🚫 | revisit only if a real scene needs it |
| Volume | ⏭ P10 | |
| Procedural (DelayedReadArchive/RunProgram/DynamicLoad) | ⏭ P9 | |
| Geometry (renderer-specific) | 🚫 | |

## Archives & external
| Request | Status | Notes |
|---|---|---|
| ReadArchive | ✅ | file (relative to RIB dir) or inline archive; depth-capped |
| ArchiveBegin/ArchiveEnd | ✅ | inline archives |
| MakeTexture/MakeLatLongEnvironment/etc. | 🟡 | `render txmake in.png out.tex` CLI converts any image to the tiled-mip .tex format (in-RIB MakeTexture requests still warn) |
| Binary RIB encoding | ⏭ P9 | |

## RSL-era shading (all skipped forever)
`ShadingModel`, `Imager`, RSL `Surface`/`Displacement`/`Volume` shader
compilation, `MakeShadow`, shadow/environment map requests — 🚫. The modern
path-traced pipeline (Bxdf/Pattern/Integrator + OSL) replaces all of it,
matching PRMan's own evolution.
