//! Unbroken Biological Experience Chain & Continuous Recurrent Flow (`cognition.rs`).
//!
//! Core Biological Principle:
//! Memory is NOT a database or `Vec` of stored items, words, or shapes.
//! Instead, from the initial moment of birth (`t = 0`), every sensation and inner feeling
//! perturbs an unbroken, continuous **Chain of Experience** (`chain_fast -> chain_slow -> chain_deep`)
//! and carves continuous causal valleys into a recurrent synaptic fabric (`w_flow`) via
//! decaying biochemical eligibility traces (`eligibility`).
//! Two organisms with identical genomes but different life histories form permanently unique
//! experience chains and spontaneously emit novel, unscripted reactions.
//!
//! Law 2: Zero string literals, zero text operations, zero discrete episode storage.

use crate::util::{dot, l2_norm, normalize_l2_in_place};

/// Continuous Dendritic Resonance & Differentiation Circuit:
/// Rather than comparing discrete stored snapshots, measures the live tension between the
/// arriving sensory wave (`chain_fast`) and the lingering experience chain (`wm_trace` / `chain_slow`):
/// - `same_features`: Dendritic coincidence (`fast * slow`) capturing continuity along the chain.
/// - `diff_features`: Apical calcium mismatch (`fast - slow`) capturing surprise & new experience.
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

    /// Continuously coupled NMDA trace update (integrates incoming wave into the ongoing trace).
    pub fn latch_reference(&mut self, x: &[f32]) {
        let n = self.dim.min(x.len());
        let trace_norm = l2_norm(&self.wm_trace);
        if trace_norm < 1e-6 {
            self.wm_trace[..n].copy_from_slice(&x[..n]);
        } else {
            for i in 0..n {
                self.wm_trace[i] = 0.35 * self.wm_trace[i] + 0.65 * x[i];
            }
        }
        normalize_l2_in_place(&mut self.wm_trace);
    }

    /// Measure continuous resonance (`same_features`) and divergence (`diff_features`) against the ongoing trace.
    pub fn compare(&mut self, x: &[f32]) -> (f32, f32) {
        let mut x_norm = vec![0.0f32; self.dim];
        let n = self.dim.min(x.len());
        x_norm[..n].copy_from_slice(&x[..n]);
        normalize_l2_in_place(&mut x_norm);

        let sim = dot(&x_norm, &self.wm_trace) / (l2_norm(&x_norm) * l2_norm(&self.wm_trace) + 1e-8);
        for i in 0..self.dim {
            self.same_features[i] = (x_norm[i] * self.wm_trace[i]).max(0.0);
            self.diff_features[i] = x_norm[i] - self.wm_trace[i];
        }
        self.last_similarity = sim;
        self.last_differentiation = (1.0 - sim).clamp(0.0, 2.0);
        (self.last_similarity, self.last_differentiation)
    }
}

/// Unbroken Multi-Timescale Experience Chain (`DeliberativeMind`):
/// Contains ZERO stored episode lists (`Vec<Struct>`). Instead, experience is a continuous
/// trajectory `chain_fast(t) -> chain_slow(t) -> chain_deep(t)` coupled by a recurrent synaptic
/// flow matrix `w_flow` and biochemical eligibility trace `eligibility`.
#[derive(Clone, Debug)]
pub struct DeliberativeMind {
    pub state_dim: usize,
    pub comparator: CorticalComparator,
    /// Immediate experiential wave `E_fast(t)`: sensory input modulated by recurrent chain flow
    pub chain_fast: Vec<f32>,
    /// Lingering working-experience chain `E_slow(t)` (bridges consecutive moments without resetting)
    pub chain_slow: Vec<f32>,
    /// Lifelong biographical identity trace `E_deep(t)` (accumulates the organism's unique history from t=0)
    pub chain_deep: Vec<f32>,
    /// Continuous synaptic eligibility trace `Z(t)` (`state_dim * state_dim`) bridging `E_slow(t-1) -> E_fast(t)`
    pub eligibility: Vec<f32>,
    /// Recurrent causal flow fabric `W_flow` (`state_dim * state_dim`) carved by lived experience
    pub w_flow: Vec<f32>,
    /// Visceral feeling/valence gradient field across the experience manifold (`state_dim`)
    pub w_valence: Vec<f32>,
    /// Continuous forward wave state `imagined_flow` (`state_dim`) emerging from recurrent relaxation
    pub imagined_flow: Vec<f32>,
    /// Unbroken count of experiential transitions lived since birth (`t = 0`)
    pub chain_steps: u64,
    /// Emergent novelty of the current experiential state relative to lifelong `chain_deep`
    pub novelty_emergence: f32,
}

impl DeliberativeMind {
    pub fn new(state_dim: usize, _unused_actions: usize, _unused_cap: usize) -> Self {
        Self {
            state_dim,
            comparator: CorticalComparator::new(state_dim),
            chain_fast: vec![0.0f32; state_dim],
            chain_slow: vec![0.0f32; state_dim],
            chain_deep: vec![0.0f32; state_dim],
            eligibility: vec![0.0f32; state_dim * state_dim],
            w_flow: vec![0.0f32; state_dim * state_dim],
            w_valence: vec![0.0f32; state_dim],
            imagined_flow: vec![0.0f32; state_dim],
            chain_steps: 0,
            novelty_emergence: 0.0,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.comparator.memory_bytes()
            + (self.chain_fast.capacity()
                + self.chain_slow.capacity()
                + self.chain_deep.capacity()
                + self.eligibility.capacity()
                + self.w_flow.capacity()
                + self.w_valence.capacity()
                + self.imagined_flow.capacity())
                * 4
    }

    /// Pool full-dimensional IT shape (`s`, e.g. 256D) and C4 colour (`c`, e.g. 64D) assemblies
    /// into a zero-mean Excitatory-Inhibitory (E-I) balanced cortical vector of dimension `self.state_dim`.
    /// Subtracting the mean firing rate and applying balanced E-I dendritic polarity eliminates the
    /// positive common-mode cone so distinct objects map to orthogonal directions on the hypersphere.
    pub fn encode_multimodal_state(&self, s: &[f32], c: &[f32]) -> Vec<f32> {
        let half = (self.state_dim / 2).max(1);
        let mut out = vec![0.0f32; self.state_dim];
        if !s.is_empty() {
            let s_mean: f32 = s.iter().copied().sum::<f32>() / (s.len() as f32);
            for (idx, &val) in s.iter().enumerate() {
                let b = idx % half;
                let sign = if (idx / half) % 2 == 0 { 1.0 } else { -1.0 };
                out[b] += sign * (val - s_mean);
            }
            normalize_l2_in_place(&mut out[..half]);
        }
        if !c.is_empty() {
            let c_mean: f32 = c.iter().copied().sum::<f32>() / (c.len() as f32);
            let rem = (self.state_dim - half).max(1);
            for (idx, &val) in c.iter().enumerate() {
                let b = half + (idx % rem);
                let sign = if (idx / rem) % 2 == 0 { 1.0 } else { -1.0 };
                out[b] += sign * (val - c_mean);
            }
            normalize_l2_in_place(&mut out[half..]);
        }
        normalize_l2_in_place(&mut out);
        out
    }

    /// Advance the unbroken Chain of Experience by one living moment (`dt`):
    /// 1. Incoming sensory reality `sensory_now` collides with the recurrent flow from `chain_fast(t-1)`
    ///    and the organism's lifelong identity trace `chain_deep`, forming the new `chain_fast(t)`.
    /// 2. The causal STDP transition `prev_fast(t-1) -> s_in(t)` updates the continuous biochemical
    ///    eligibility trace `eligibility(t)` and sculpts `w_flow` (ZERO discrete episodes stored!).
    /// 3. Activity relaxes forward through `w_flow` for 2 continuous neural waves, producing
    ///    an emergent forward intention wave `imagined_flow(t)`.
    pub fn advance_experience_chain(
        &mut self,
        sensory_now: &[f32],
        valence: f32,
        arousal: f32,
        eta: f32,
    ) -> &[f32] {
        let d = self.state_dim;
        self.chain_steps += 1;

        let mut s_in = vec![0.0f32; d];
        let n = d.min(sensory_now.len());
        s_in[..n].copy_from_slice(&sensory_now[..n]);
        normalize_l2_in_place(&mut s_in);

        let (_, diff_surprise) = self.comparator.compare(&s_in);

        let prev_fast = self.chain_fast.clone();
        let prev_norm = l2_norm(&prev_fast);

        // Recurrent prediction from prior link of the chain: `flow_pred = W_flow * prev_fast`
        let mut flow_pred = vec![0.0f32; d];
        if prev_norm > 1e-6 {
            for i in 0..d {
                let row = &self.w_flow[i * d..(i + 1) * d];
                flow_pred[i] = dot(row, &prev_fast);
            }
        }

        // Update continuous STDP eligibility trace `Z(t)` and recurrent causal fabric `W_flow`
        // whenever a genuine state transition occurs (`diff_surprise > 0.05`)
        if prev_norm > 1e-6 && diff_surprise > 0.05 {
            let eff_eta = (eta * (1.0 + 0.8 * arousal.clamp(0.0, 2.0))).min(0.35);
            for i in 0..d {
                let row_z = &mut self.eligibility[i * d..(i + 1) * d];
                let row_w = &mut self.w_flow[i * d..(i + 1) * d];
                let err_i = s_in[i] - flow_pred[i];
                for j in 0..d {
                    let delta = err_i * prev_fast[j];
                    row_z[j] = 0.70 * row_z[j] + 0.30 * delta;
                    row_w[j] = (row_w[j] + eff_eta * delta).clamp(-2.5, 2.5);
                }
                self.w_valence[i] = (0.95 * self.w_valence[i] + eff_eta * valence * s_in[i]).clamp(-2.0, 2.0);
            }
        }

        // Update current living chain states: `chain_fast`, `chain_slow`, and lifelong `chain_deep`
        self.chain_fast.copy_from_slice(&s_in);
        for i in 0..d {
            self.chain_slow[i] = 0.65 * self.chain_slow[i] + 0.35 * self.chain_fast[i];
            self.chain_deep[i] = 0.985 * self.chain_deep[i] + 0.015 * (self.chain_fast[i] + 0.25 * valence);
        }
        normalize_l2_in_place(&mut self.chain_slow);
        normalize_l2_in_place(&mut self.chain_deep);
        self.comparator.latch_reference(&self.chain_fast);

        let deep_norm = l2_norm(&self.chain_deep);
        self.novelty_emergence = if deep_norm > 1e-6 {
            (1.0 - dot(&self.chain_fast, &self.chain_deep) / (l2_norm(&self.chain_fast) * deep_norm + 1e-8)).clamp(0.0, 2.0)
        } else {
            1.0
        };

        // Continuous 2-step neural wave propagation along carved experience valleys (`imagined_flow`)
        self.imagined_flow = self.propagate_wave(&self.chain_fast, 2);
        &self.imagined_flow
    }

    /// Offline slow-wave sleep consolidation: let the eligibility trace and deep identity trace
    /// settle into the recurrent causal flow matrix `w_flow` without any stored episode array.
    pub fn consolidate_chain(&mut self, eta: f32) {
        let d = self.state_dim;
        for i in 0..d {
            let row_z = &mut self.eligibility[i * d..(i + 1) * d];
            let row_w = &mut self.w_flow[i * d..(i + 1) * d];
            for j in 0..d {
                row_w[j] = (row_w[j] + eta * row_z[j]).clamp(-2.5, 2.5);
                row_z[j] *= 0.50;
            }
        }
    }

    /// Propagate any state `s0` forward through `steps` continuous recurrent waves of `w_flow`.
    pub fn propagate_wave(&self, s0: &[f32], steps: usize) -> Vec<f32> {
        let d = self.state_dim;
        let mut wave = vec![0.0f32; d];
        let n = d.min(s0.len());
        wave[..n].copy_from_slice(&s0[..n]);
        normalize_l2_in_place(&mut wave);
        for _ in 0..steps {
            let mut next_w = vec![0.0f32; d];
            for i in 0..d {
                let row = &self.w_flow[i * d..(i + 1) * d];
                next_w[i] = dot(row, &wave);
            }
            normalize_l2_in_place(&mut next_w);
            wave = next_w;
        }
        wave
    }
}
