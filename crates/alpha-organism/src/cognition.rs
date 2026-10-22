//! Biological Thinking, Multi-Aspect Differentiation, Comparison, Episodic Experience
//! Formation, and Multi-Step Deliberative Solution Engine (`cognition.rs`).
//! Law 2: Zero string literals, zero text operations.

use crate::util::{dot, l2_norm, normalize_l2_in_place};

/// Prefrontal-Parietal Working Memory & Dendritic Comparator Circuit:
/// Holds a sustained NMDA working-memory trace `wm_trace` of the previous attended state
/// and decomposes any new incoming cortical state `x` into:
/// - `same_features`: Coincidence detection (`x * wm_trace`) isolating shared invariants
/// - `diff_features`: Dendritic subtraction (`x - wm_trace`) isolating what changed/differs
/// - `similarity` and `differentiation` scalars
#[derive(Clone, Debug)]
pub struct CorticalComparator {
    pub dim: usize,
    pub wm_trace: Vec<f32>,
    pub same_features: Vec<f32>,
    pub diff_features: Vec<f32>,
    pub last_similarity: f32,
    pub last_differentiation: f32,
}

impl CorticalComparator {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            wm_trace: vec![0.0; dim],
            same_features: vec![0.0; dim],
            diff_features: vec![0.0; dim],
            last_similarity: 0.0,
            last_differentiation: 1.0,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.wm_trace.capacity() + self.same_features.capacity() + self.diff_features.capacity()) * 4
    }

    /// Latch a state into working memory (ACh/DA-gated NMDA persistent activity).
    pub fn latch_reference(&mut self, x: &[f32]) {
        let n = self.dim.min(x.len());
        self.wm_trace[..n].copy_from_slice(&x[..n]);
        normalize_l2_in_place(&mut self.wm_trace);
    }

    /// Compare incoming state `x` against working-memory reference `wm_trace` without overwriting `wm_trace`.
    pub fn compare(&mut self, x: &[f32]) -> (f32, f32) {
        let mut x_norm = vec![0.0f32; self.dim];
        let n = self.dim.min(x.len());
        x_norm[..n].copy_from_slice(&x[..n]);
        normalize_l2_in_place(&mut x_norm);

        let sim = dot(&x_norm, &self.wm_trace) / (l2_norm(&x_norm) * l2_norm(&self.wm_trace) + 1e-8);
        for i in 0..self.dim {
            // Coincidence (AND-like dendritic spike) for shared aspect
            self.same_features[i] = (x_norm[i] * self.wm_trace[i]).max(0.0);
            // Apical-basal subtraction for differentiating aspect
            self.diff_features[i] = x_norm[i] - self.wm_trace[i];
        }
        self.last_similarity = sim;
        self.last_differentiation = (1.0 - sim).clamp(0.0, 2.0);
        (self.last_similarity, self.last_differentiation)
    }
}

/// A single grounded episodic experience trace formed via 1-shot Hippocampal BTSP.
#[derive(Clone, Debug)]
pub struct EpisodicTrace {
    pub state_before: Vec<f32>,
    pub action_code: usize,
    pub state_after: Vec<f32>,
    pub valence_outcome: f32,
    pub salience: f32,
}

/// Complementary Learning Systems Episodic Memory + Multi-Step Deliberative Thinker:
/// - Records salient `(state_before, action, state_after, valence)` experiences in 1 shot.
/// - Gradually consolidates causal transition matrices `w_transition[action]` (`state_before -> state_after`)
///   and `w_value[action]` (`state_before -> expected_valence`).
/// - Performs multi-step internal mental rollout ("thinking before reacting") to find a multi-step
///   action sequence that transforms an incoming problem state `s_0` into a high-valence / goal-aligned solution.
#[derive(Clone, Debug)]
pub struct DeliberativeMind {
    pub state_dim: usize,
    pub n_actions: usize,
    pub comparator: CorticalComparator,
    pub episodes: Vec<EpisodicTrace>,
    pub max_episodes: usize,
    /// Learned causal transition matrices: `n_actions` matrices of `state_dim * state_dim`
    pub w_transition: Vec<f32>,
    /// Learned state-outcome value weights: `n_actions * state_dim`
    pub w_value: Vec<f32>,
}

impl DeliberativeMind {
    pub fn new(state_dim: usize, n_actions: usize, max_episodes: usize) -> Self {
        Self {
            state_dim,
            n_actions,
            comparator: CorticalComparator::new(state_dim),
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
            w_transition: vec![0.0f32; n_actions * state_dim * state_dim],
            w_value: vec![0.0f32; n_actions * state_dim],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let ep_bytes: usize = self
            .episodes
            .iter()
            .map(|e| (e.state_before.capacity() + e.state_after.capacity()) * 4 + 16)
            .sum();
        self.comparator.memory_bytes() + ep_bytes + (self.w_transition.capacity() + self.w_value.capacity()) * 4
    }

    /// Pool full-dimensional IT shape (`s`, e.g. 256D) and C4 colour (`c`, e.g. 64D) assemblies
    /// into a compact prefrontal working-memory vector of dimension `self.state_dim` (e.g. 32D:
    /// 16 dendritic shape branches + 16 dendritic colour branches) so zero cortical spikes are lost.
    pub fn encode_multimodal_state(&self, s: &[f32], c: &[f32]) -> Vec<f32> {
        let half = (self.state_dim / 2).max(1);
        let mut out = vec![0.0f32; self.state_dim];
        if !s.is_empty() {
            for (idx, &val) in s.iter().enumerate() {
                let b = (idx * half) / s.len();
                out[b] += val;
            }
            normalize_l2_in_place(&mut out[..half]);
        }
        if !c.is_empty() {
            for (idx, &val) in c.iter().enumerate() {
                let b = half + (idx * (self.state_dim - half)) / c.len();
                out[b] += val;
            }
            normalize_l2_in_place(&mut out[half..]);
        }
        normalize_l2_in_place(&mut out);
        out
    }

    /// Record a live experience and update the causal transition & value models via local Hebbian/delta plasticity.
    pub fn record_experience(
        &mut self,
        s_before: &[f32],
        action: usize,
        s_after: &[f32],
        valence: f32,
        salience: f32,
        eta: f32,
    ) {
        let d = self.state_dim;
        let act = action % self.n_actions;
        let mut sb = vec![0.0f32; d];
        let mut sa = vec![0.0f32; d];
        sb[..d.min(s_before.len())].copy_from_slice(&s_before[..d.min(s_before.len())]);
        sa[..d.min(s_after.len())].copy_from_slice(&s_after[..d.min(s_after.len())]);
        normalize_l2_in_place(&mut sb);
        normalize_l2_in_place(&mut sa);

        // Update causal forward transition matrix W_a: sa_hat = W_a * sb
        let base_t = act * d * d;
        let eff_eta = eta * (1.0 + 2.0 * salience.clamp(0.0, 2.0));
        for i in 0..d {
            let row = &mut self.w_transition[base_t + i * d..base_t + (i + 1) * d];
            let pred_i = dot(row, &sb);
            let err_i = sa[i] - pred_i;
            for j in 0..d {
                row[j] += eff_eta * err_i * sb[j];
            }
        }

        // Update expected valence weights for (sb, act)
        let base_v = act * d;
        let v_row = &mut self.w_value[base_v..base_v + d];
        let pred_v = dot(v_row, &sb);
        let err_v = valence - pred_v;
        for j in 0..d {
            v_row[j] += eff_eta * err_v * sb[j];
        }

        if salience > 0.05 {
            if self.episodes.len() >= self.max_episodes {
                self.episodes.remove(0);
            }
            self.episodes.push(EpisodicTrace {
                state_before: sb,
                action_code: act,
                state_after: sa,
                valence_outcome: valence,
                salience,
            });
        }
    }

    /// Predict the next cortical state `s_{t+1}` and expected valence if `action` is taken from `state`.
    /// Combines Neocortical semantic transition weights (`w_transition`) with Hippocampal CA3
    /// precondition-gated episodic pattern completion (`self.episodes`). If the precondition for `action`
    /// is not met in `state` (`max_sim <= 0.75`), the state persists at `state` rather than teleporting.
    pub fn predict_step(&self, state: &[f32], action: usize) -> (Vec<f32>, f32) {
        let d = self.state_dim;
        let act = action % self.n_actions;
        let base_t = act * d * d;
        let mut cort_s = vec![0.0f32; d];
        for i in 0..d {
            let row = &self.w_transition[base_t + i * d..base_t + (i + 1) * d];
            cort_s[i] = dot(row, state);
        }

        let mut ep_s = vec![0.0f32; d];
        let mut ep_w_sum = 0.0f32;
        let mut ep_val_sum = 0.0f32;
        let mut max_sim = 0.0f32;
        for ep in &self.episodes {
            if ep.action_code == act {
                let sim = dot(&ep.state_before, state).max(0.0);
                if sim > max_sim {
                    max_sim = sim;
                }
                if sim > 0.75 {
                    let w = (sim - 0.75).powi(3);
                    for i in 0..d {
                        ep_s[i] += w * ep.state_after[i];
                    }
                    ep_val_sum += w * ep.valence_outcome;
                    ep_w_sum += w;
                }
            }
        }

        let gate = ((max_sim - 0.75) / 0.20).clamp(0.0, 1.0);
        let mut next_s = vec![0.0f32; d];
        if ep_w_sum > 1e-8 && gate > 0.0 {
            for i in 0..d {
                let target_i = 0.25 * cort_s[i] + 0.75 * (ep_s[i] / ep_w_sum);
                next_s[i] = (1.0 - gate) * state[i] + gate * target_i;
            }
        } else {
            for i in 0..d {
                next_s[i] = 0.85 * state[i] + 0.15 * cort_s[i];
            }
        }

        normalize_l2_in_place(&mut next_s);
        let base_v = act * d;
        let v_row = &self.w_value[base_v..base_v + d];
        let mut exp_val = gate * dot(v_row, state);
        if ep_w_sum > 1e-8 {
            exp_val = gate * (0.35 * dot(v_row, state) + 0.65 * (ep_val_sum / ep_w_sum));
        }
        (next_s, exp_val)
    }

    /// Deliberative Multi-Step Mental Rollout ("Thinking & Reasoning to Find a Solution"):
    /// Given current incoming state `s0` and a desired goal/resolution state `goal`,
    /// simulates all 2-step causal paths `(a1, a2)` internally:
    ///   `s0 --(a1)--> s1_hat --(a2)--> s2_hat`
    /// and returns `(best_first_action, best_second_action, predicted_final_state, utility_score)`.
    pub fn deliberate_solution(&self, s0: &[f32], goal: &[f32]) -> (usize, usize, Vec<f32>, f32) {
        let d = self.state_dim;
        let mut s_init = vec![0.0f32; d];
        let mut g_norm = vec![0.0f32; d];
        s_init[..d.min(s0.len())].copy_from_slice(&s0[..d.min(s0.len())]);
        g_norm[..d.min(goal.len())].copy_from_slice(&goal[..d.min(goal.len())]);
        normalize_l2_in_place(&mut s_init);
        normalize_l2_in_place(&mut g_norm);

        let mut best_a1 = 0usize;
        let mut best_a2 = 0usize;
        let mut best_state = s_init.clone();
        let mut best_score = f32::NEG_INFINITY;

        for a1 in 0..self.n_actions {
            let (s1, v1) = self.predict_step(&s_init, a1);
            for a2 in 0..self.n_actions {
                let (s2, v2) = self.predict_step(&s1, a2);
                let align2 = dot(&s2, &g_norm);
                let total_score = align2 + 0.25 * (v1 + v2);
                if total_score > best_score {
                    best_score = total_score;
                    best_a1 = a1;
                    best_a2 = a2;
                    best_state = s2;
                }
            }
        }
        (best_a1, best_a2, best_state, best_score)
    }

    /// Replay stored hippocampal episodes into the cortical causal transition matrices (offline learning).
    pub fn replay_episodes(&mut self, eta: f32) {
        let eps = self.episodes.clone();
        for ep in &eps {
            let d = self.state_dim;
            let act = ep.action_code % self.n_actions;
            let base_t = act * d * d;
            for i in 0..d {
                let row = &mut self.w_transition[base_t + i * d..base_t + (i + 1) * d];
                let pred_i = dot(row, &ep.state_before);
                let err_i = ep.state_after[i] - pred_i;
                for j in 0..d {
                    row[j] += eta * err_i * ep.state_before[j];
                }
            }
        }
    }
}
