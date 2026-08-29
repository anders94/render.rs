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
| ObjectBegin/ObjectEnd/ObjectInstance | ⏭ P2 | TLAS instancing |
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
| Option (generic) | ✅ | passthrough dictionary |
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
| Color / Opacity | ✅ / 🟡 | opacity render effect ⏭ P4 |
| Surface | ✅ | legacy matte/plastic/metal mapping |
| Bxdf / Pattern / Displace / Light | ⏭ P4/P6 | the modern requests |
| LightSource / AreaLightSource | ✅ / ✅ | point+distant; quad polygons under AreaLightSource become sampleable rect lights (path integrator) |
| Illuminate (light linking) | ⏭ P4 | |
| Atmosphere / Interior / Exterior | ⏭ P10 | |
| Displacement (RSL-era) | 🚫 | RSL-only; `Displace` is the modern path |
| Orientation / ReverseOrientation / Sides | 🟡 | honored for meshes at P2 |
| Basis | 🟡 | used at P5 patches / P8 curves |
| ShadingRate / ShadingInterpolation | 🟡 | dicing rate analog at P5 |
| Attribute (generic) | ✅ | passthrough dictionary |
| Detail / DetailRange / GeometricApproximation | 🚫 | always highest detail |
| TextureCoordinates | 🟡 | |

## Geometry
| Request | Status | Notes |
|---|---|---|
| Sphere / Cylinder / Cone | ✅ | |
| Torus / Disk / Paraboloid / Hyperboloid | ✅ | P0 |
| Polygon | ✅ | P0, convex fan triangulation |
| GeneralPolygon / PointsPolygons / PointsGeneralPolygons | ⏭ P2 | |
| PatchMesh / Patch | ⏭ P5 | |
| NuPatch / TrimCurve | ⏭ P5 | trims later |
| SubdivisionMesh / HierarchicalSubdivisionMesh | ⏭ P5 | |
| Curves | ⏭ P8 | |
| Points | ⏭ P8 | |
| Blobby | 🚫 | revisit only if a real scene needs it |
| Volume | ⏭ P10 | |
| Procedural (DelayedReadArchive/RunProgram/DynamicLoad) | ⏭ P9 | |
| Geometry (renderer-specific) | 🚫 | |

## Archives & external
| Request | Status | Notes |
|---|---|---|
| ReadArchive | ⏭ P2 | |
| ArchiveBegin/ArchiveEnd | ⏭ P2 | inline archives |
| MakeTexture/MakeLatLongEnvironment/etc. | ⏭ P6 | txmake-equivalent tool instead |
| Binary RIB encoding | ⏭ P9 | |

## RSL-era shading (all skipped forever)
`ShadingModel`, `Imager`, RSL `Surface`/`Displacement`/`Volume` shader
compilation, `MakeShadow`, shadow/environment map requests — 🚫. The modern
path-traced pipeline (Bxdf/Pattern/Integrator + OSL) replaces all of it,
matching PRMan's own evolution.
