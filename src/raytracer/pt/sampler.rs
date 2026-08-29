//! PCG32 random sampler: small, fast, statistically solid, and
//! deterministically seedable per (pixel, sample) so renders reproduce
//! exactly. Fancier low-discrepancy sequences (pmj02/Sobol) slot in behind
//! the same interface later.

pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Deterministic per-(pixel, sample) seeding.
    pub fn for_pixel_sample(pixel: u64, sample: u64) -> Self {
        // splitmix-style hash to decorrelate the two indices.
        let mix = |mut z: u64| {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self::new(mix(pixel.wrapping_mul(0x9e3779b97f4a7c15) ^ sample), mix(sample) | 1)
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) * (1.0 / 4294967296.0)
    }

    /// Two independent uniforms in [0, 1).
    pub fn next_2d(&mut self) -> (f64, f64) {
        (self.next_f64(), self.next_f64())
    }

    /// Uniform integer in [0, n).
    pub fn next_below(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_mean() {
        let mut rng = Pcg32::new(42, 54);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
    }

    #[test]
    fn deterministic_per_seed() {
        let a: Vec<u32> = {
            let mut r = Pcg32::for_pixel_sample(7, 3);
            (0..8).map(|_| r.next_u32()).collect()
        };
        let b: Vec<u32> = {
            let mut r = Pcg32::for_pixel_sample(7, 3);
            (0..8).map(|_| r.next_u32()).collect()
        };
        let c: Vec<u32> = {
            let mut r = Pcg32::for_pixel_sample(7, 4);
            (0..8).map(|_| r.next_u32()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
