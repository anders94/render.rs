mod traits;
mod sphere;
mod cone;
mod cylinder;
mod extra;
mod mesh;
pub mod curves;
pub mod displace;
pub mod earclip;
pub mod patches;
pub mod subdiv;

pub use traits::{Intersectable, PrimitiveDesc, PrimitiveKind};
pub use sphere::Sphere;
pub use cone::Cone;
pub use cylinder::Cylinder;
pub use extra::{Disk, Hyperboloid, Paraboloid, Torus, Triangle};
pub use curves::CurveSet;
pub use mesh::{to_intersection, GeomKind, Instance, Mesh, MeshHit};
