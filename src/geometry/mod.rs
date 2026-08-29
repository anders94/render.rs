mod traits;
mod sphere;
mod cone;
mod cylinder;
mod extra;

pub use traits::{Intersectable, PrimitiveDesc, PrimitiveKind};
pub use sphere::Sphere;
pub use cone::Cone;
pub use cylinder::Cylinder;
pub use extra::{Disk, Hyperboloid, Paraboloid, Torus, Triangle};
