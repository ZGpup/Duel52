//! Network weights, and the `.d52nn` checkpoint format.
//!
//! # The format
//!
//! ```text
//! magic        6 bytes   b"D52NN\0"
//! version      u16 LE
//! header_len   u32 LE
//! header       UTF-8, newline-delimited key=value, in this fixed key order:
//!                  obs_dim, action_dim, width, blocks, value_hidden,
//!                  obs_layout_hash, action_layout_hash, param_order
//! payload      for each name in param_order: a little-endian f32 array,
//!              its length implied by the architecture
//! ```
//!
//! Not ONNX (a large C++ dependency for a five-layer MLP) and not safetensors (which would
//! need a JSON parser on the Rust side, and `engine` has no dependencies). This is ~100
//! lines that both sides read and write with nothing but their standard library.
//!
//! `param_order` is explicit so neither side depends on dict or field iteration order.
//!
//! # The two hashes are the important part
//!
//! `obs_layout_hash` and `action_layout_hash` come from [`crate::encode`], which is the only
//! implementation of them; Python obtains them through PyO3 rather than recomputing them.
//! [`Weights::load`] recomputes them from *this build's* constants and refuses a checkpoint
//! that disagrees, naming the field that moved.
//!
//! That check is the whole reason the format has a header. Silent layout drift between the
//! function that was trained and the function that is evaluated does not crash anything — it
//! produces an agent that is merely bad, at which point the natural suspect is the training
//! run, and the real bug is a feature that moved three floats to the left. Making it a load
//! error makes it impossible rather than unlikely.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::Path;

use crate::config::GameConfig;
use crate::encode::{action_dim, action_layout_hash, obs_dim, obs_layout_hash};
use crate::rng::Rng;

pub const CHECKPOINT_MAGIC: &[u8; 6] = b"D52NN\0";
pub const CHECKPOINT_VERSION: u16 = 1;

/// The architecture, pinned by `PHASE3_STEP1.md` §1.5 and `DESIGN.md` §5.
///
/// Travels in the checkpoint header, so a saved network describes its own shape and a
/// config drift on either side is a load error rather than a silent reinterpretation of the
/// payload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Arch {
    pub obs_dim: usize,
    pub action_dim: usize,
    /// Residual trunk width. 512 in the default configuration.
    pub width: usize,
    /// Residual blocks. 5 in the default configuration.
    pub blocks: usize,
    /// Hidden width of the value head. 256 in the default configuration.
    pub value_hidden: usize,
}

impl Arch {
    /// The default architecture for a configuration: `DESIGN.md` §5's residual MLP at width
    /// 512, five blocks, value hidden 256. ≈5.1M parameters, ~20 MB of fp32.
    pub fn default_for(config: &GameConfig) -> Arch {
        Arch {
            obs_dim: obs_dim(config),
            action_dim: action_dim(config),
            width: 512,
            blocks: 5,
            value_hidden: 256,
        }
    }

    /// Every parameter tensor, in load order, as `(name, len)`.
    ///
    /// This list **is** `param_order`. Both sides walk it rather than trusting their own
    /// field or dict ordering, so adding a tensor in the middle is a format change that the
    /// header records rather than a silent shift of every array after it.
    pub fn params(&self) -> Vec<(String, usize)> {
        let (w, b) = (self.width, self.blocks);
        let mut out = vec![
            ("in.weight".to_string(), w * self.obs_dim),
            ("in.bias".to_string(), w),
            ("ln_in.weight".to_string(), w),
            ("ln_in.bias".to_string(), w),
        ];
        for i in 0..b {
            out.push((format!("block{i}.ln.weight"), w));
            out.push((format!("block{i}.ln.bias"), w));
            out.push((format!("block{i}.fc1.weight"), w * w));
            out.push((format!("block{i}.fc1.bias"), w));
            out.push((format!("block{i}.fc2.weight"), w * w));
            out.push((format!("block{i}.fc2.bias"), w));
        }
        out.push(("ln_out.weight".to_string(), w));
        out.push(("ln_out.bias".to_string(), w));
        out.push(("policy.weight".to_string(), self.action_dim * w));
        out.push(("policy.bias".to_string(), self.action_dim));
        out.push(("value1.weight".to_string(), self.value_hidden * w));
        out.push(("value1.bias".to_string(), self.value_hidden));
        out.push(("value2.weight".to_string(), self.value_hidden));
        out.push(("value2.bias".to_string(), 1));
        out
    }

    pub fn param_count(&self) -> usize {
        self.params().iter().map(|(_, n)| n).sum()
    }

    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("obs_dim", self.obs_dim),
            ("action_dim", self.action_dim),
            ("width", self.width),
            ("value_hidden", self.value_hidden),
        ] {
            if value == 0 {
                return Err(format!("{name} must be positive"));
            }
        }
        Ok(())
    }
}

/// A loaded network: the architecture plus one flat f32 array per tensor, in `param_order`.
///
/// Flat arrays rather than nested structs because the forward pass wants contiguous rows and
/// the file format wants contiguous bytes; a shape-aware representation would only be
/// converting between the two.
#[derive(Clone, PartialEq, Debug)]
pub struct Weights {
    pub arch: Arch,
    /// One entry per [`Arch::params`] tensor, same order.
    pub params: Vec<Vec<f32>>,
}

impl Weights {
    /// Look a tensor up by its position in [`Arch::params`].
    #[inline]
    pub fn tensor(&self, index: usize) -> &[f32] {
        &self.params[index]
    }

    /// Deterministic random initialisation from the engine's own [`Rng`].
    ///
    /// This exists so **every Rust test is self-contained and needs no Python-produced
    /// file**. Without it `phase3_netpolicy_plays_legal_games` and the forward-pass
    /// determinism tests could only run after a `maturin develop` and a `python -m
    /// duel52.nn init`, which means they would not run in plain `cargo test` — and a test
    /// that only runs on a fully set-up machine is a test that stops running.
    ///
    /// Linear layers get PyTorch's default bound, `U(-1/√fan_in, 1/√fan_in)`. LayerNorm
    /// affine parameters are drawn *near* 1 and 0 rather than exactly at them: an identity
    /// affine would make a transposed or swapped gamma/beta invisible to the parity test,
    /// which is one of the transcription bugs it exists to catch. `py/duel52/nn` perturbs
    /// them for the same reason.
    pub fn random(seed: u64, arch: Arch) -> Weights {
        arch.validate().expect("architecture must be valid");
        let mut rng = Rng::derive(seed, 0x4E4E_0000_0000_0001);
        let params = arch
            .params()
            .into_iter()
            .map(|(name, len)| {
                let mut v = Vec::with_capacity(len);
                if name.ends_with("ln.weight") || name.ends_with("ln_in.weight")
                    || name.ends_with("ln_out.weight")
                {
                    for _ in 0..len {
                        v.push(1.0 + 0.05 * uniform(&mut rng));
                    }
                } else if name.contains("ln") && name.ends_with(".bias") {
                    for _ in 0..len {
                        v.push(0.05 * uniform(&mut rng));
                    }
                } else {
                    let fan_in = fan_in_of(&name, &arch, len);
                    let bound = 1.0 / (fan_in as f32).sqrt();
                    for _ in 0..len {
                        v.push(bound * uniform(&mut rng));
                    }
                }
                v
            })
            .collect();
        Weights { arch, params }
    }

    /// Read a checkpoint, checking it against `config`.
    ///
    /// Returns a message naming the mismatched field rather than a generic parse error: the
    /// person reading it is holding a checkpoint they believed was compatible.
    pub fn load(path: &Path, config: &GameConfig) -> Result<Weights, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("cannot read checkpoint `{}`: {e}", path.display()))?;
        Weights::from_bytes(&bytes, config)
            .map_err(|e| format!("`{}`: {e}", path.display()))
    }

    /// Write a checkpoint, stamping `config`'s layout hashes into the header.
    pub fn save(&self, path: &Path, config: &GameConfig) -> Result<(), String> {
        let bytes = self.to_bytes(config);
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("cannot create `{}`: {e}", dir.display()))?;
            }
        }
        std::fs::File::create(path)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))
    }

    /// The header this build would stamp, as the checkpoint stores it.
    pub fn header_string(arch: &Arch, config: &GameConfig) -> String {
        let names: Vec<String> = arch.params().into_iter().map(|(n, _)| n).collect();
        let mut s = String::new();
        let _ = writeln!(s, "obs_dim={}", arch.obs_dim);
        let _ = writeln!(s, "action_dim={}", arch.action_dim);
        let _ = writeln!(s, "width={}", arch.width);
        let _ = writeln!(s, "blocks={}", arch.blocks);
        let _ = writeln!(s, "value_hidden={}", arch.value_hidden);
        let _ = writeln!(s, "obs_layout_hash={:016x}", obs_layout_hash(config));
        let _ = writeln!(s, "action_layout_hash={:016x}", action_layout_hash(config));
        let _ = writeln!(s, "param_order={}", names.join(","));
        s
    }

    pub fn to_bytes(&self, config: &GameConfig) -> Vec<u8> {
        let header = Weights::header_string(&self.arch, config);
        let mut out = Vec::with_capacity(16 + header.len() + 4 * self.arch.param_count());
        out.extend_from_slice(CHECKPOINT_MAGIC);
        out.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for tensor in &self.params {
            for v in tensor {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8], config: &GameConfig) -> Result<Weights, String> {
        if bytes.len() < 12 {
            return Err("truncated: not even a header".into());
        }
        if &bytes[..6] != CHECKPOINT_MAGIC {
            return Err("not a .d52nn checkpoint (bad magic)".into());
        }
        let version = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version != CHECKPOINT_VERSION {
            return Err(format!(
                "checkpoint format version {version}, but this build reads {CHECKPOINT_VERSION}"
            ));
        }
        let header_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let header_end = 12 + header_len;
        if bytes.len() < header_end {
            return Err("truncated: the header is shorter than it claims".into());
        }
        let header = std::str::from_utf8(&bytes[12..header_end])
            .map_err(|_| "the header is not valid UTF-8".to_string())?;

        let mut fields: Vec<(&str, &str)> = Vec::new();
        for line in header.lines().filter(|l| !l.trim().is_empty()) {
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("header line `{line}` is not `key=value`"))?;
            fields.push((k, v));
        }
        let get = |key: &str| -> Result<&str, String> {
            fields
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| *v)
                .ok_or_else(|| format!("header is missing `{key}`"))
        };
        let num = |key: &str| -> Result<usize, String> {
            get(key)?
                .parse::<usize>()
                .map_err(|_| format!("header `{key}` is not a number"))
        };

        let arch = Arch {
            obs_dim: num("obs_dim")?,
            action_dim: num("action_dim")?,
            width: num("width")?,
            blocks: num("blocks")?,
            value_hidden: num("value_hidden")?,
        };
        arch.validate()?;

        // --- the checks this format exists for ---------------------------------------
        expect(arch.obs_dim, obs_dim(config), "obs_dim")?;
        expect(arch.action_dim, action_dim(config), "action_dim")?;
        expect_hash(get("obs_layout_hash")?, obs_layout_hash(config), "obs_layout_hash")?;
        expect_hash(
            get("action_layout_hash")?,
            action_layout_hash(config),
            "action_layout_hash",
        )?;

        let declared: Vec<&str> = get("param_order")?.split(',').collect();
        let expected = arch.params();
        if declared.len() != expected.len() {
            return Err(format!(
                "param_order lists {} tensors, but a {}-block network has {}",
                declared.len(),
                arch.blocks,
                expected.len()
            ));
        }
        for (i, (name, _)) in expected.iter().enumerate() {
            if declared[i] != name {
                return Err(format!(
                    "param_order[{i}] is `{}`, expected `{name}`",
                    declared[i]
                ));
            }
        }

        // --- the payload -------------------------------------------------------------
        let need = 4 * arch.param_count();
        let payload = &bytes[header_end..];
        if payload.len() != need {
            return Err(format!(
                "payload is {} bytes, but the architecture needs {need}",
                payload.len()
            ));
        }
        let mut at = 0;
        let mut params = Vec::with_capacity(expected.len());
        for (_, len) in &expected {
            let mut tensor = Vec::with_capacity(*len);
            for _ in 0..*len {
                tensor.push(f32::from_le_bytes([
                    payload[at],
                    payload[at + 1],
                    payload[at + 2],
                    payload[at + 3],
                ]));
                at += 4;
            }
            params.push(tensor);
        }
        Ok(Weights { arch, params })
    }
}

fn expect(found: usize, want: usize, field: &str) -> Result<(), String> {
    if found == want {
        Ok(())
    } else {
        Err(format!(
            "{field} is {found} in the checkpoint but {want} in this build — the checkpoint \
             was trained against a different encoder"
        ))
    }
}

fn expect_hash(found: &str, want: u64, field: &str) -> Result<(), String> {
    let want = format!("{want:016x}");
    if found.trim() == want {
        Ok(())
    } else {
        Err(format!(
            "{field} is {found} in the checkpoint but {want} in this build — the observation \
             or action layout moved since the checkpoint was written, so its weights no \
             longer mean what they meant. Retrain, or check out the build that produced it."
        ))
    }
}

/// Uniform in `[-1, 1)`, from the engine's frozen generator.
fn uniform(rng: &mut Rng) -> f32 {
    // 24 bits is the f32 mantissa, so this covers every representable value in the range
    // without the rounding artefacts a wider draw would introduce.
    let bits = (rng.next_u64() >> 40) as u32; // 24 bits
    (bits as f32 / (1u32 << 23) as f32) - 1.0
}

/// Fan-in of a tensor, for the initialisation bound. Biases share their layer's fan-in,
/// which is what PyTorch's `nn.Linear` does.
fn fan_in_of(name: &str, arch: &Arch, len: usize) -> usize {
    if name.starts_with("in.") {
        arch.obs_dim
    } else if name.starts_with("policy.") || name.starts_with("value1.") {
        arch.width
    } else if name.starts_with("value2.") {
        arch.value_hidden
    } else if name.contains("fc1") || name.contains("fc2") {
        arch.width
    } else {
        len.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_architecture_is_about_five_million_parameters() {
        let arch = Arch::default_for(&GameConfig::default());
        let n = arch.param_count();
        assert!(
            (5_000_000..5_300_000).contains(&n),
            "expected ≈5.1M parameters, got {n}"
        );
        // 4 bytes each, ~20 MB.
        assert!((19..22).contains(&(4 * n / 1_000_000)));
    }

    #[test]
    fn random_init_is_reproducible_and_seed_dependent() {
        let arch = Arch {
            obs_dim: 16,
            action_dim: 8,
            width: 8,
            blocks: 2,
            value_hidden: 4,
        };
        assert_eq!(Weights::random(1, arch), Weights::random(1, arch));
        assert_ne!(Weights::random(1, arch), Weights::random(2, arch));
    }

    /// LayerNorm affine parameters must not initialise to exactly `1` and `0`, or a swapped
    /// gamma/beta would be invisible to the parity test.
    #[test]
    fn layernorm_affine_is_perturbed_so_a_swap_would_show() {
        let arch = Arch {
            obs_dim: 16,
            action_dim: 8,
            width: 8,
            blocks: 1,
            value_hidden: 4,
        };
        let w = Weights::random(3, arch);
        let names: Vec<String> = arch.params().into_iter().map(|(n, _)| n).collect();
        let gamma = &w.params[names.iter().position(|n| n == "ln_in.weight").unwrap()];
        let beta = &w.params[names.iter().position(|n| n == "ln_in.bias").unwrap()];
        assert!(gamma.iter().any(|&v| v != 1.0));
        assert!(beta.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn a_truncated_checkpoint_is_rejected_rather_than_read_as_garbage() {
        let config = GameConfig::default();
        let arch = Arch {
            obs_dim: obs_dim(&config),
            action_dim: action_dim(&config),
            width: 4,
            blocks: 1,
            value_hidden: 2,
        };
        let bytes = Weights::random(9, arch).to_bytes(&config);
        let short = &bytes[..bytes.len() - 8];
        assert!(Weights::from_bytes(short, &config)
            .unwrap_err()
            .contains("payload"));
    }

    #[test]
    fn a_file_that_is_not_a_checkpoint_is_rejected_by_magic() {
        let err = Weights::from_bytes(b"not a checkpoint at all", &GameConfig::default())
            .unwrap_err();
        assert!(err.contains("magic"));
    }
}
