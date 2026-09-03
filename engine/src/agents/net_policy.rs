//! The policy-only network agent — Phase 3's smoke rung.
//!
//! Encode the observation for the player to move, run one forward pass, mask, take the
//! argmax, decode, play it. No search: that arrives in step 2, and keeping it out here means
//! a failure in this rung is a failure in the *encoding path*, which is the thing step 1 is
//! trying to prove correct.
//!
//! Deterministic on purpose. Sampling from the policy belongs with self-play, and
//! determinism is what makes the anti-cheat assertion in
//! `engine/tests/agents.rs::phase2_no_agent_reads_hidden_information` exact for this agent
//! too: two states in one information set encode identically, so they must produce the same
//! action, with no random tie-break to explain a difference away.
//!
//! # Reading the observation off the real state is legitimate
//!
//! This looks like the Phase 2 greedy bug and is not. Greedy cheated because it **applied**
//! candidate actions to the real state, and applying reveals ranks: flipping your own base
//! card turns it face-up, and killing a face-down card sends its rank to the public discard.
//! This agent only **reads** a filtered projection, and
//! `phase3_observation_is_a_function_of_the_information_set` is what proves the projection is
//! filtered correctly. The two tests together are the whole argument.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::action::Action;
use crate::agents::Agent;
use crate::config::GameConfig;
use crate::encode::{action_dim, decode_action, encode_action, encode_observation, obs_dim};
use crate::nn::{Evaluator, MlpEvaluator, Weights};
use crate::state::GameState;

/// A checkpoint played greedily.
pub struct NetPolicyAgent {
    checkpoint: PathBuf,
    /// Resolved on the first decision, because [`crate::AgentSpec::build`] does not know the
    /// configuration the agent will be asked to play under and the layout hashes are
    /// config-derived.
    evaluator: Option<MlpEvaluator>,
    obs: Vec<f32>,
    logits: Vec<f32>,
    values: Vec<f32>,
    mask: Vec<bool>,
}

impl NetPolicyAgent {
    pub fn new(checkpoint: impl Into<PathBuf>) -> NetPolicyAgent {
        NetPolicyAgent {
            checkpoint: checkpoint.into(),
            evaluator: None,
            obs: Vec::new(),
            logits: Vec::new(),
            values: Vec::new(),
            mask: Vec::new(),
        }
    }

    fn ensure_loaded(&mut self, config: &GameConfig) {
        // Reload if the shape moved. An agent is normally built per game and plays one
        // configuration, so this fires only if one instance is reused across configs — but
        // silently encoding into a buffer sized for a different `obs_dim` is the kind of
        // thing that would surface as a bad agent rather than as an error.
        if self.evaluator.is_some() && self.obs.len() == obs_dim(config) {
            return;
        }
        let weights = load_cached(&self.checkpoint, config).unwrap_or_else(|e| {
            // `Agent::choose` cannot fail, and a checkpoint that will not load is a setup
            // error rather than a game state the agent has to cope with — so this is a
            // panic with the whole diagnosis in it, not a silent fallback to random play.
            panic!("netpolicy: {e}")
        });
        self.obs = vec![0.0; obs_dim(config)];
        self.logits = vec![0.0; action_dim(config)];
        self.values = vec![0.0; 1];
        self.mask = vec![false; action_dim(config)];
        self.evaluator = Some(MlpEvaluator::shared(weights));
    }
}

impl Agent for NetPolicyAgent {
    fn choose(&mut self, state: &GameState, legal: &[Action]) -> Action {
        self.ensure_loaded(&state.config);
        let evaluator = self.evaluator.as_ref().expect("loaded above");

        encode_observation(state, state.acting_player(), &mut self.obs);
        evaluator.eval_batch(&self.obs, 1, &mut self.logits, &mut self.values);

        // Mask from the actions we were handed rather than re-enumerating, so the agent can
        // only ever return something it was offered.
        self.mask.fill(false);
        for action in legal {
            self.mask[encode_action(action, state)] = true;
        }

        // Argmax over the masked logits, lowest index first on a tie. Deliberately *not* a
        // random tie-break: see the module docs.
        let mut best = usize::MAX;
        let mut best_logit = f32::NEG_INFINITY;
        for (i, &allowed) in self.mask.iter().enumerate() {
            if allowed && self.logits[i] > best_logit {
                best_logit = self.logits[i];
                best = i;
            }
        }
        debug_assert!(best != usize::MAX, "no legal action was masked in");

        let action = decode_action(best, state)
            .expect("a masked index was produced by encode_action, so it must decode");
        debug_assert!(
            legal.contains(&action),
            "netpolicy decoded `{action}`, which was not offered"
        );
        action
    }

    fn name(&self) -> String {
        format!("netpolicy:{}", self.checkpoint.display())
    }
}

/// Process-wide checkpoint cache.
///
/// A ladder builds a fresh agent per game — `duel52 ladder --games 400` across five rungs is
/// thousands of builds — and re-reading and re-parsing a 20 MB file each time would dominate
/// the run. Weights are immutable once loaded, so sharing them changes nothing about
/// reproducibility.
///
/// **The key includes the layout hashes, not just the path.** Keying on the path alone would
/// let a checkpoint loaded under one configuration be served to an agent playing under a
/// configuration whose layout it does not match: the second `Weights::load` would never run,
/// so the check the format exists for would be skipped exactly when it mattered.
type CacheKey = (PathBuf, u64, u64);
type Cache = Mutex<HashMap<CacheKey, Arc<Weights>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_cached(path: &Path, config: &GameConfig) -> Result<Arc<Weights>, String> {
    let key = (
        path.to_path_buf(),
        crate::encode::obs_layout_hash(config),
        crate::encode::action_layout_hash(config),
    );
    if let Some(found) = cache().lock().expect("checkpoint cache").get(&key) {
        return Ok(found.clone());
    }
    let weights = Arc::new(Weights::load(path, config)?);
    cache()
        .lock()
        .expect("checkpoint cache")
        .insert(key, weights.clone());
    Ok(weights)
}
