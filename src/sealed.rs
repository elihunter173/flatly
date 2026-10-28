/// Prevents downstream crates from implementing traits whose implementations
/// must be controlled by this crate.
pub trait Sealed {}
