mod traits;
mod sphere;
mod cone;
mod cylinder;
mod extra;
mod mesh;

pub use traits::{Intersectable, PrimitiveDesc, PrimitiveKind};
pub use sphere::Sphere;
pub use cone::Cone;
pub use cylinder::Cylinder;
pub use extra::{Disk, Hyperboloid, Paraboloid, Torus, Triangle};
pub use mesh::{to_intersection, Instance, Mesh, MeshHit};
