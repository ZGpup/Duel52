//! A small, fully specified pseudo-random number generator.
//!
//! # Why not use the `rand` crate?
//!
//! `CLAUDE.md` requires that "same seed + same config → identical game", forever. The
//! popular Rust RNG crates explicitly reserve the right to change their output stream
//! between versions, which would silently invalidate every seed recorded in `FINDINGS.md`.
//! So the engine carries its own generator, ~40 lines of well-known published algorithms,
//! frozen here.
//!
//! - **Seeding** uses SplitMix64 (Steele, Lea & Flood 2014) to expand a single `u64` seed
//!   into the four words xoshiro needs. A single-word seed of `0` is a classic failure mode
//!   for xorshift-family generators; SplitMix64 has no bad seeds, which removes it.
//! - **Generation** uses xoshiro256\*\* (Blackman & Vigna 2018). Fast, 2^256 period, passes
//!   BigCrush. Statistically far better than anything a card shuffle needs.
//!
//! Nothing in the *game* is random once it has been dealt: every random draw the engine
//! makes happens during setup. The RNG is kept on the state anyway because Phase 3's
//! ISMCTS determinization (`DESIGN.md` §6) samples hidden state and wants a reproducible
//! stream attached to the position.

/// Reproducible RNG. Cloning it clones the stream position, which is what you want when
/// forking a search: the clone continues deterministically from the same point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a generator from a 64-bit seed. Every seed is usable, including `0`.
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        let mut next = || {
            // SplitMix64.
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            state: [next(), next(), next(), next()],
        }
    }

    /// Derive an independent generator from this one, tagged by `stream`.
    ///
    /// Used to keep unrelated sources of randomness from interfering: the deal, player 0's
    /// agent, and player 1's agent each get their own stream from one game seed, so
    /// changing how many random numbers an agent consumes cannot change the deal.
    pub fn derive(seed: u64, stream: u64) -> Rng {
        // Mixing the two words before seeding avoids correlated streams for nearby seeds.
        let mut z = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
        z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        Rng::new(z ^ (z >> 33))
    }

    /// Next raw 64-bit value. xoshiro256\*\*.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n`. Panics if `n == 0`.
    ///
    /// Uses Lemire's multiply-shift method with rejection, so the result is *exactly*
    /// uniform — no modulo bias. Bias here would be invisible but would quietly skew every
    /// statistic the project reports.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below(0) is undefined");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut low = m as u64;
        if low < n {
            // Only the rare "straddling" region needs rejection.
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform index in `0..len`.
    #[inline]
    pub fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// Pick a uniformly random element. Returns `None` for an empty slice.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.index(items.len())])
        }
    }

    /// Fisher–Yates shuffle, iterating downwards. This exact loop is part of the engine's
    /// reproducibility contract: changing the direction or the index formula would change
    /// every deal.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index(i + 1);
            items.swap(i, j);
        }
    }

    /// Uniform in the open-above interval `[0, 1)`, from the top 53 bits.
    ///
    /// Never returns exactly 1, and [`Rng::unit_open`] is the variant that also never
    /// returns 0 — which matters wherever a logarithm is taken.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        // 53 bits is the whole mantissa, so every representable value in [0,1) with that
        // spacing is reachable and none is doubled up.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform in `(0, 1)` — neither endpoint. For `ln(u)`.
    #[inline]
    pub fn unit_open(&mut self) -> f64 {
        // `unit()` yields k / 2^53 for integer k in [0, 2^53); shifting to (k + 0.5) / 2^53
        // keeps it uniform to the same precision and excludes both ends.
        self.unit() + 0.5 / 9_007_199_254_740_992.0
    }

    /// A standard normal, by Box–Muller.
    ///
    /// The second variate of the pair is discarded rather than cached. Caching it would make
    /// the number of `next_u64` calls depend on how many normals were drawn *earlier*, which
    /// is exactly the kind of hidden state that makes a "reproducible" run stop reproducing
    /// when an unrelated call site changes.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit_open();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A Gamma(`shape`, 1) variate, by Marsaglia–Tsang (2000).
    ///
    /// Used to build the Dirichlet noise `agents/net_mcts.rs` mixes into root priors.
    /// Normalising independent Gamma(α) draws is what makes that noise a *true* Dirichlet
    /// over whichever subset of actions is legal in the current determinization — the
    /// property a directly-sampled Dirichlet vector would lose the moment the legal set
    /// changed. Panics on a non-positive shape.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0, "gamma shape must be positive, got {shape}");
        if shape < 1.0 {
            // Marsaglia–Tsang's boost: Gamma(a) = Gamma(a + 1) · U^(1/a).
            let g = self.gamma(shape + 1.0);
            return g * self.unit_open().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.unit_open();
            if u.ln() < 0.5 * x * x + d - d * v + d * (v.ln()) {
                return d * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn zero_seed_is_not_degenerate() {
        let mut r = Rng::new(0);
        let first = r.next_u64();
        assert_ne!(first, 0);
        assert_ne!(first, r.next_u64());
    }

    #[test]
    fn derived_streams_are_independent() {
        let mut a = Rng::derive(7, 0);
        let mut b = Rng::derive(7, 1);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(xs, ys);
    }

    #[test]
    fn below_stays_in_range_and_is_roughly_uniform() {
        let mut r = Rng::new(99);
        let mut buckets = [0u32; 7];
        for _ in 0..70_000 {
            let v = r.below(7) as usize;
            buckets[v] += 1;
        }
        for b in buckets {
            // 10_000 expected per bucket; +-8% is a very loose bound that a biased
            // modulo implementation on a small modulus would still pass, so this test is
            // a smoke check, not a statistical proof. Uniformity is guaranteed by
            // construction (Lemire rejection), not by this assertion.
            assert!(b > 9_200 && b < 10_800, "bucket count {b} looks skewed");
        }
    }

    #[test]
    fn unit_stays_inside_its_interval() {
        let mut r = Rng::new(5);
        let mut sum = 0.0;
        for _ in 0..50_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "unit() returned {u}");
            let o = r.unit_open();
            assert!(o > 0.0 && o < 1.0, "unit_open() returned {o}");
            sum += u;
        }
        let mean = sum / 50_000.0;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} is not 0.5");
    }

    /// Marsaglia–Tsang has to be right on both sides of `shape = 1`, because the branch
    /// below 1 is a different algorithm wrapping the one above it. Mean and variance of
    /// Gamma(a, 1) are both `a`.
    #[test]
    fn gamma_has_the_right_mean_and_variance() {
        for shape in [0.3f64, 1.0, 2.5] {
            let mut r = Rng::derive(17, shape.to_bits());
            let n = 200_000;
            let xs: Vec<f64> = (0..n).map(|_| r.gamma(shape)).collect();
            assert!(xs.iter().all(|x| *x > 0.0), "gamma returned a non-positive");
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            // ~0.5% standard error at this sample size; 8% is loose enough never to flake
            // and tight enough that a wrong branch (which is off by a factor, not a
            // percent) fails.
            assert!((mean - shape).abs() < 0.08 * shape, "shape {shape}: mean {mean}");
            assert!((var - shape).abs() < 0.12 * shape, "shape {shape}: var {var}");
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::new(4);
        let mut v: Vec<usize> = (0..64).collect();
        r.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(v, sorted, "a 64-element shuffle should not be the identity");
    }
}
