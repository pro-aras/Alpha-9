//! Unbroken Biological Experience Chain & Whole-Body Cellular Epigenetic Memory (`cognition.rs`).
//!
//! Core Biological Principles:
//! 1. **Continuous River of Consciousness (`chain_fast -> chain_slow -> chain_deep` + `w_flow`)**:
//!    Every sensation, spoken word, bodily action, and inner feeling perturbs a multi-timescale
//!    experience chain and carves recurrent valleys into `w_flow` via decaying eligibility traces.
//! 2. **Multi-Scale Action Accountability & Self-Supervised Agency inside 256 Epigenetic Cells**:
//!    Each `EpigeneticCell` maintains multi-scale intracellular eligibility traces across three
//!    time horizons (`tag_fast` ~1-2 ticks, `tag_med` ~4-8 ticks, `tag_slow` ~12-25 ticks) along
//!    with the exact limb actions (`elig_action`) and vocalizations (`elig_vocal`) taken at those
//!    horizons:
//!    - **Self-Supervised Sensorimotor Agency (`world_delta_w`)**: Each cell measures how its
//!      actions actually changed the sensory world (`delta_s = s(t) - s(t-1)`), storing that
//!      causal transformation directly inside the cell even without external reward.
//!    - **Multi-Horizon D1/D2 Action Accountability (`motor_action_w`)**: When an outcome or
//!      feeling arrives at tick `t`, every cell with active fast, medium, or slow eligibility
//!      accounts for what it did: positive outcomes (D1) lock in the credited action across time
//!      horizons, while negative outcomes (D2) inhibit failed actions so the hands systematically
//!      explore new actions to solve single-step and multi-step puzzles.
//!    - **Homeostatic TET Chromatin Demethylation**: Prevents 100% epigenetic saturation over long
//!      lifetimes by gently recycling weakly-tagged chromatin marks when the cellular pool fills up,
//!      preserving a reserve of open plastic cells for lifelong learning.
//! 3. **Whole-Body 5-Tissue Cellular Memory (`retina_cells`, `cochlea_cells`, `limb_cells`, `organ_cells`, `muscle_cells`)**:
//!    Every sensory, motor, vocal, and visceral organ tissue in the body adjusts its intracellular
//!    molecular baseline alongside the brain and communicates bidirectionally with the cortex.
//!
//! Law 2: Zero string literals, zero text operations, zero discrete episode storage structs.

use crate::util::{dot, l2_norm, normalize_l2_in_place};

/// Continuous Dendritic Resonance & Differentiation Circuit:
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

/// Single Living Cell with Multi-Scale Action Eligibility, Self-Supervised Delta, and Epigenetics:
#[derive(Clone, Debug)]
pub struct EpigeneticCell {
    pub creb_excitability: f32,
    pub chromatin_locked: f32,
    /// Multi-scale intracellular eligibility tags (`fast` ~1-2 ticks, `med` ~4-8 ticks, `slow` ~12-25 ticks)
    pub synaptic_tag: f32,
    pub tag_med: f32,
    pub tag_slow: f32,
    pub erk_nuclear: f32,
    pub vesicle_ready: f32,
    pub receptor_w: Vec<f32>,
    pub context_w: Vec<f32>,
    pub axon_out_w: Vec<f32>,
    /// Self-supervised sensorimotor memory of how this cell's action changed the world (`s(t) - s(t-1)`)
    pub world_delta_w: Vec<f32>,
    pub somatic_feeling: f32,
    /// Multi-scale action eligibility traces (`[fast, med, slow]` x `4` limb actions) + pending outcome gate
    pub elig_action: [[f32; 4]; 3],
    pub action_pending: f32,
    /// Learned limb action synergy (`action_drive[0..4]`) + bitmask of actions that failed in this state
    pub motor_action_w: [f32; 4],
    pub failed_actions: [f32; 4],
    /// Vocal articulation synergy (`vocal_tract[0..8]`) coupled to this cellular experience
    pub vocal_motor_w: [f32; 8],
}

impl EpigeneticCell {
    pub fn memory_bytes(&self) -> usize {
        156 + (self.receptor_w.capacity()
            + self.context_w.capacity()
            + self.axon_out_w.capacity()
            + self.world_delta_w.capacity())
            * 4
    }
}

/// Unbroken Multi-Timescale Experience Chain + Sparse Epigenetic Cells + 5-Tissue Whole-Body Memory:
#[derive(Clone, Debug)]
pub struct DeliberativeMind {
    pub state_dim: usize,
    pub comparator: CorticalComparator,
    pub chain_fast: Vec<f32>,
    pub chain_slow: Vec<f32>,
    pub chain_deep: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub w_flow: Vec<f32>,
    pub w_valence: Vec<f32>,
    pub imagined_flow: Vec<f32>,
    pub chain_steps: u64,
    pub novelty_emergence: f32,
    pub cells: Vec<EpigeneticCell>,
    pub prev_cell_state: Vec<f32>,
    pub current_scene_state: Vec<f32>,
    pub transition_context_state: Vec<f32>,
    pub active_ensemble: Vec<usize>,
    /// Efference copies of the organism's limb action & vocalization on the previous tick (`t-1`)
    pub prev_action_drive: [f32; 4],
    pub prev_vocal_tract: [f32; 8],
    /// Multi-scale global action accountability traces (`fast`, `med`, `slow`)
    pub action_trace_fast: [f32; 4],
    pub action_trace_med: [f32; 4],
    pub action_trace_slow: [f32; 4],
    /// Cumulative self-supervised agency score (how much the organism's actions meaningfully change reality)
    pub agency_confidence: f32,
    pub last_similarity_vec: Vec<f32>,
    pub last_difference_vec: Vec<f32>,
    pub last_recalled_outcome: Vec<f32>,
    pub last_recalled_feeling: f32,
    pub last_recalled_action: [f32; 4],
    pub last_recalled_vocal: [f32; 8],
    /// 5 Peripheral Somatic Cellular Tissues:
    pub retina_cells: [f32; 8],
    pub cochlea_cells: [f32; 8],
    pub limb_cells: [f32; 4],
    pub organ_cells: [f32; 8],
    pub organ_erk: [f32; 8],
    pub muscle_cells: [f32; 8],
    pub habituation_trace: Vec<f32>,
    pub gaze_trail: [[f32; 3]; 4],
    pub bio_seed: u64,
}

impl DeliberativeMind {
    pub fn new(state_dim: usize, _unused_actions: usize, _unused_cap: usize) -> Self {
        let n_cells = 256usize;
        let mut cells = Vec::with_capacity(n_cells);
        for c in 0..n_cells {
            let mut receptor_w = vec![0.0f32; state_dim];
            let mut context_w = vec![0.0f32; state_dim];
            for d in 0..state_dim {
                let p1 = (((c * 37 + d * 19 + 11) % 103) as f32 / 51.0) - 1.0;
                let p2 = (((c * 61 + d * 23 + 17) % 107) as f32 / 53.0) - 1.0;
                receptor_w[d] = p1;
                context_w[d] = p2;
            }
            normalize_l2_in_place(&mut receptor_w);
            normalize_l2_in_place(&mut context_w);
            let creb_init = ((c * 29 + 5) % 97) as f32 / 97.0;
            let mut motor_init = [0.20f32; 4];
            motor_init[c % 4] = 0.70;
            cells.push(EpigeneticCell {
                creb_excitability: creb_init,
                chromatin_locked: 0.0,
                synaptic_tag: 0.0,
                tag_med: 0.0,
                tag_slow: 0.0,
                erk_nuclear: 0.0,
                vesicle_ready: 1.0,
                receptor_w,
                context_w,
                axon_out_w: vec![0.0; state_dim],
                world_delta_w: vec![0.0; state_dim],
                somatic_feeling: 0.0,
                elig_action: [motor_init; 3],
                action_pending: 0.0,
                motor_action_w: motor_init,
                failed_actions: [0.0; 4],
                vocal_motor_w: [0.5; 8],
            });
        }

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
            cells,
            prev_cell_state: vec![0.0f32; state_dim],
            current_scene_state: vec![0.0f32; state_dim],
            transition_context_state: vec![0.0f32; state_dim],
            active_ensemble: Vec::with_capacity(8),
            prev_action_drive: [0.25; 4],
            prev_vocal_tract: [0.5; 8],
            action_trace_fast: [0.25; 4],
            action_trace_med: [0.25; 4],
            action_trace_slow: [0.25; 4],
            agency_confidence: 0.0,
            last_similarity_vec: vec![0.0f32; state_dim],
            last_difference_vec: vec![0.0f32; state_dim],
            last_recalled_outcome: vec![0.0f32; state_dim],
            last_recalled_feeling: 0.0,
            last_recalled_action: [0.25; 4],
            last_recalled_vocal: [0.5; 8],
            retina_cells: [1.0; 8],
            cochlea_cells: [1.0; 8],
            limb_cells: [0.25; 4],
            organ_cells: [0.0; 8],
            organ_erk: [0.0; 8],
            muscle_cells: [0.5; 8],
            habituation_trace: vec![0.0f32; state_dim],
            gaze_trail: [[0.5, 0.5, 0.0]; 4],
            bio_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let cell_bytes: usize = self.cells.iter().map(|c| c.memory_bytes()).sum();
        self.comparator.memory_bytes()
            + cell_bytes
            + (self.chain_fast.capacity()
                + self.chain_slow.capacity()
                + self.chain_deep.capacity()
                + self.eligibility.capacity()
                + self.w_flow.capacity()
                + self.w_valence.capacity()
                + self.imagined_flow.capacity()
                + self.prev_cell_state.capacity()
                + self.current_scene_state.capacity()
                + self.transition_context_state.capacity()
                + self.last_similarity_vec.capacity()
                + self.last_difference_vec.capacity()
                + self.last_recalled_outcome.capacity()
                + self.habituation_trace.capacity())
                * 4
    }

    pub fn next_ion_fluctuation(&mut self, perturbation: f32) -> f32 {
        let bits = (perturbation.to_bits() as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        self.bio_seed = self.bio_seed.wrapping_add(0x9E37_79B9_7F4A_7C15 ^ bits);
        let mut z = self.bio_seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / ((1u64 << 23) as f32)) - 1.0
    }

    pub fn encode_multimodal_state(&self, s: &[f32], c: &[f32]) -> Vec<f32> {
        let half = (self.state_dim / 2).max(1);
        let mut out = vec![0.0f32; self.state_dim];
        if !s.is_empty() {
            let s_mean: f32 = s.iter().copied().sum::<f32>() / (s.len() as f32);
            for (idx, &val) in s.iter().enumerate() {
                let b = idx % half;
                let sign = if (idx / half) % 2 == 0 { 1.0 } else { -1.0 };
                let ret_gain = self.retina_cells[b % 4];
                out[b] += sign * (val - s_mean) * ret_gain;
            }
            normalize_l2_in_place(&mut out[..half]);
        }
        if !c.is_empty() {
            let c_mean: f32 = c.iter().copied().sum::<f32>() / (c.len() as f32);
            let rem = (self.state_dim - half).max(1);
            for (idx, &val) in c.iter().enumerate() {
                let b = half + (idx % rem);
                let sign = if (idx / rem) % 2 == 0 { 1.0 } else { -1.0 };
                let ret_gain = self.retina_cells[4 + (b % 4)];
                out[b] += sign * (val - c_mean) * ret_gain;
            }
            normalize_l2_in_place(&mut out[half..]);
        }
        normalize_l2_in_place(&mut out);
        out
    }

    pub fn encode_auditory_state(&self, a_heard: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.state_dim];
        if !a_heard.is_empty() && l2_norm(a_heard) > 1e-6 {
            let a_mean: f32 = a_heard.iter().copied().sum::<f32>() / (a_heard.len() as f32);
            for (idx, &val) in a_heard.iter().enumerate() {
                let b = idx % self.state_dim;
                let sign = if (idx / self.state_dim) % 2 == 0 { 1.0 } else { -1.0 };
                let coch_gain = self.cochlea_cells[b % 8];
                out[b] += sign * (val - a_mean) * coch_gain;
            }
            normalize_l2_in_place(&mut out);
        }
        out
    }

    /// Step the population of 256 living epigenetic cells with Multi-Scale Action Accountability:
    /// 1. Compute self-supervised world change `delta_s = s_in - prev_cell_state`.
    /// 2. Retroactive STC + Multi-Scale D1/D2 Action Accountability across `active_ensemble` AND
    ///    any cells holding medium/slow eligibility tags (`tag_med`, `tag_slow`) when a delayed
    ///    consequence arrives!
    /// 3. Homeostatic TET chromatin demethylation keeps ~25-30% of cells open for lifelong learning.
    pub fn step_epigenetic_cells(&mut self, s_in: &[f32], valence: f32, arousal: f32) {
        let d = self.state_dim;
        let salience = (valence.abs() + 0.5 * arousal).clamp(0.0, 2.0);
        let n_cells = self.cells.len();

        // Update distinct-scene transition context whenever the sensory room/object changes
        if l2_norm(&self.current_scene_state) > 0.1 {
            if dot(s_in, &self.current_scene_state) < 0.82 {
                self.transition_context_state.copy_from_slice(&self.current_scene_state);
                self.current_scene_state.copy_from_slice(s_in);
            }
        } else if l2_norm(s_in) > 0.1 {
            self.current_scene_state.copy_from_slice(s_in);
        }

        // Compute self-supervised state change caused by the organism's actions: `delta_s = s(t) - s(t-1)`
        let mut delta_s = vec![0.0f32; d];
        let has_prev = l2_norm(&self.prev_cell_state) > 0.1;
        if has_prev {
            for i in 0..d {
                delta_s[i] = s_in[i] - self.prev_cell_state[i];
            }
        }
        let delta_mag = l2_norm(&delta_s);
        self.agency_confidence = 0.95 * self.agency_confidence + 0.05 * (delta_mag * 0.7).clamp(0.0, 1.0);

        let prev_act = self.prev_action_drive;
        let prev_voc = self.prev_vocal_tract;
        let step_u = self.chain_steps as usize;

        // 1A. Immediate (`t-1 -> t`) Retroactive STC & Self-Supervised Agency update on `active_ensemble`
        for &c_idx in &self.active_ensemble {
            let cell = &mut self.cells[c_idx];
            let shield = if salience > 0.45 {
                1.0 - 0.45 * cell.chromatin_locked
            } else {
                (1.0 - 0.98 * cell.chromatin_locked).max(0.0)
            };
            let lr_out = (0.45 * (0.3 + salience) * shield).clamp(0.0, 0.85);
            if lr_out > 1e-4 {
                for i in 0..d {
                    cell.axon_out_w[i] = (1.0 - lr_out) * cell.axon_out_w[i] + lr_out * s_in[i];
                    cell.world_delta_w[i] = (1.0 - lr_out) * cell.world_delta_w[i] + lr_out * delta_s[i];
                }
                cell.somatic_feeling = (1.0 - lr_out) * cell.somatic_feeling + lr_out * valence;
            }

            // Record or inhibit the limb action `prev_act` based on outcome valence (consuming `action_pending`)
            let mut chosen_a = 0usize;
            for a in 1..4 {
                if prev_act[a] > prev_act[chosen_a] {
                    chosen_a = a;
                }
            }
            if valence >= 0.12 && cell.action_pending > 0.5 {
                // D1 pathway: strong positive outcome locks in `chosen_a` (or Approach `0` on passive comfort)
                if chosen_a == 1 && cell.failed_actions.iter().copied().sum::<f32>() < 0.1 {
                    chosen_a = 0;
                }
                cell.failed_actions[chosen_a] = 0.0;
                for a in 0..4 {
                    let target = if a == chosen_a { 0.96 } else { 0.04 };
                    cell.motor_action_w[a] = 0.05 * cell.motor_action_w[a] + 0.95 * target;
                }
                for v in 0..8 {
                    cell.vocal_motor_w[v] = 0.35 * cell.vocal_motor_w[v] + 0.65 * prev_voc[v];
                }
                cell.chromatin_locked = (cell.chromatin_locked + 0.65).clamp(0.0, 0.99);
                cell.action_pending = 0.0;
            } else if valence <= -0.08 && cell.action_pending > 0.5 {
                // D2 pathway: mark `chosen_a` as failed in this cell and shift `motor_action_w` to an untried action!
                cell.failed_actions[chosen_a] = (cell.failed_actions[chosen_a] + 0.95).clamp(0.0, 1.0);
                let mut next_best_a = (chosen_a + 1) % 4;
                for cand in 0..4 {
                    let a_try = (chosen_a + 1 + cand) % 4;
                    if cell.failed_actions[a_try] < cell.failed_actions[next_best_a] {
                        next_best_a = a_try;
                    }
                }
                for a in 0..4 {
                    let target = if a == next_best_a {
                        0.94
                    } else if cell.failed_actions[a] > 0.5 {
                        0.02
                    } else {
                        0.18
                    };
                    cell.motor_action_w[a] = 0.05 * cell.motor_action_w[a] + 0.95 * target;
                }
                cell.chromatin_locked = (cell.chromatin_locked + 0.38).clamp(0.0, 0.85);
                cell.action_pending = 0.0;
            } else {
                for v in 0..8 {
                    cell.vocal_motor_w[v] = 0.70 * cell.vocal_motor_w[v] + 0.30 * prev_voc[v];
                }
            }

            let kukushkin_creb = cell.vesicle_ready * cell.erk_nuclear;
            if salience > 0.45 || kukushkin_creb > 0.18 {
                let boost = if salience > 0.45 {
                    0.55 + 0.35 * salience
                } else {
                    0.85 * kukushkin_creb
                };
                cell.chromatin_locked = (cell.chromatin_locked + boost).clamp(0.0, 0.99);
            }
            cell.vesicle_ready *= 0.22;
        }

        // Align any other unspecialized perceptual cells matching `s_in` (`max_mw < 0.85`)
        // when a clear motor outcome (`valence >= 0.12` or `valence <= -0.08`) is discovered on `s_in`
        if (valence >= 0.12 || valence <= -0.08) && !self.active_ensemble.is_empty() {
            let ref_idx = self.active_ensemble[0];
            let ref_mw = self.cells[ref_idx].motor_action_w;
            let ref_fa = self.cells[ref_idx].failed_actions;
            for c_idx in 0..n_cells {
                if self.active_ensemble.iter().any(|&x| x == c_idx) {
                    continue;
                }
                let cell = &mut self.cells[c_idx];
                let max_mw = cell.motor_action_w.iter().copied().fold(0.0f32, f32::max);
                if max_mw < 0.85 && dot(&cell.receptor_w, s_in) > 0.80 {
                    cell.motor_action_w = ref_mw;
                    cell.failed_actions = ref_fa;
                }
            }
        }

        // 1B. Multi-Scale Time-Horizon Accountability (`tag_med` & `tag_slow`):
        //     When an outcome (`valence >= 0.12` or `valence <= -0.08`) arrives at tick `t`, credit
        //     propagates backward across cells that were active 2..15 ticks ago (`tag_med`, `tag_slow`)!
        if valence >= 0.12 || valence <= -0.08 {
            for c_idx in 0..n_cells {
                let is_imm = self.active_ensemble.iter().any(|&x| x == c_idx);
                if is_imm {
                    continue;
                }
                let cell = &mut self.cells[c_idx];
                let horizon_elig = 0.55 * cell.tag_med + 0.45 * cell.tag_slow;
                if horizon_elig > 0.15 && cell.action_pending > 0.5 && dot(&cell.receptor_w, s_in) > 0.65 {
                    let lr_h = (0.55 * horizon_elig * (0.4 + valence.abs())).clamp(0.10, 0.90);
                    cell.somatic_feeling = (1.0 - lr_h) * cell.somatic_feeling + lr_h * valence;
                    let mut cred_a = 0usize;
                    let mut best_ca = -1.0f32;
                    for a in 0..4 {
                        let ca = 0.55 * cell.elig_action[1][a] + 0.45 * cell.elig_action[2][a];
                        if ca > best_ca {
                            best_ca = ca;
                            cred_a = a;
                        }
                    }
                    if valence >= 0.12 {
                        cell.failed_actions[cred_a] = 0.0;
                        for a in 0..4 {
                            let target = if a == cred_a { 0.96 } else { 0.04 };
                            cell.motor_action_w[a] = 0.05 * cell.motor_action_w[a] + 0.95 * target;
                        }
                        cell.chromatin_locked = (cell.chromatin_locked + 0.55 * horizon_elig).clamp(0.0, 0.99);
                        cell.action_pending = 0.0;
                    } else if valence <= -0.08 {
                        cell.failed_actions[cred_a] = (cell.failed_actions[cred_a] + 0.95).clamp(0.0, 1.0);
                        let mut next_best_a = (cred_a + 1) % 4;
                        for cand in 0..4 {
                            let a_try = (cred_a + 1 + cand) % 4;
                            if cell.failed_actions[a_try] < cell.failed_actions[next_best_a] {
                                next_best_a = a_try;
                            }
                        }
                        for a in 0..4 {
                            let target = if a == next_best_a {
                                0.94
                            } else if cell.failed_actions[a] > 0.5 {
                                0.02
                            } else {
                                0.18
                            };
                            cell.motor_action_w[a] = 0.05 * cell.motor_action_w[a] + 0.95 * target;
                        }
                        cell.chromatin_locked = (cell.chromatin_locked + 0.38 * horizon_elig).clamp(0.0, 0.85);
                        cell.action_pending = 0.0;
                    }
                }
            }
        }

        // 2. Multiplicative NMDA context-gated ensemble allocation for `s_in`
        let ctx_ref = if l2_norm(&self.transition_context_state) > 0.1 {
            &self.transition_context_state
        } else {
            &self.prev_cell_state
        };
        let has_ctx = l2_norm(ctx_ref) > 0.1;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_cells);
        for (c_idx, cell) in self.cells.iter().enumerate() {
            let match_s = dot(&cell.receptor_w, s_in);
            let match_ctx = if has_ctx {
                dot(&cell.context_w, ctx_ref)
            } else {
                0.5
            };
            let basal = (match_s + 1.0).max(0.0);
            let apical = (match_ctx + 1.0).max(0.0);
            let nmda_coincidence = basal * apical;

            let allocation_drive = if match_s > 0.72 && (!has_ctx || match_ctx > 0.82) {
                nmda_coincidence * 1.55
            } else if cell.chromatin_locked > 0.20 && has_ctx && match_ctx < 0.80 {
                // Protect context-specialized cells (`Key_A -> Lock` vs `Key_B -> Lock`) from overwriting each other!
                nmda_coincidence * 0.02
            } else {
                (1.0 - 0.85 * cell.chromatin_locked) * (nmda_coincidence + 0.35 * cell.creb_excitability)
            };
            scores.push((c_idx, allocation_drive));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = 6usize.min(n_cells);
        self.active_ensemble.clear();
        for &(c_idx, _) in scores.iter().take(k) {
            self.active_ensemble.push(c_idx);
            let cell = &mut self.cells[c_idx];
            let lr_in = 0.88 * (1.0 - 0.92 * cell.chromatin_locked);
            if lr_in > 1e-3 {
                for i in 0..d {
                    cell.receptor_w[i] += lr_in * (s_in[i] - cell.receptor_w[i]);
                    if has_ctx {
                        cell.context_w[i] += lr_in * (ctx_ref[i] - cell.context_w[i]);
                    }
                }
                normalize_l2_in_place(&mut cell.receptor_w);
                normalize_l2_in_place(&mut cell.context_w);
            }
            cell.synaptic_tag = (cell.synaptic_tag + (0.45 + 0.35 * salience) * cell.vesicle_ready).clamp(0.0, 1.0);
            cell.tag_med = (cell.tag_med + 0.45).clamp(0.0, 1.0);
            cell.tag_slow = (cell.tag_slow + 0.35).clamp(0.0, 1.0);
            cell.creb_excitability *= 0.10;
        }

        // Now read out cellular experience, similarity/difference, and motor/vocal predictions AFTER update!
        let (sim_v, diff_v, rec_out, rec_feel) = self.resonate_and_explain(s_in);
        self.last_similarity_vec = sim_v;
        self.last_difference_vec = diff_v;
        self.last_recalled_outcome = rec_out;
        self.last_recalled_feeling = rec_feel;

        let ctx_query = if l2_norm(&self.transition_context_state) > 0.1 {
            Some(self.transition_context_state.as_slice())
        } else {
            Some(self.prev_cell_state.as_slice())
        };
        let (_, _, rec_act, rec_voc) = self.extract_cellular_action_and_vocal(s_in, ctx_query);
        self.last_recalled_action = rec_act;
        self.last_recalled_vocal = rec_voc;

        // Count how many cells currently have locked chromatin for homeostatic TET demethylation
        let locked_count = self.cells.iter().filter(|c| c.chromatin_locked > 0.5).count();
        let tet_demethyl_active = locked_count > (n_cells * 185) / 256;

        for (c_idx, cell) in self.cells.iter_mut().enumerate() {
            let is_active = self.active_ensemble.iter().any(|&x| x == c_idx);
            if !is_active {
                let wave = (((step_u * 17 + c_idx * 11) % 97) as f32) / 97.0;
                cell.creb_excitability = (0.90 * cell.creb_excitability + 0.10 * wave).clamp(0.0, 1.0);
                cell.erk_nuclear = (0.96 * cell.erk_nuclear + 0.08 * cell.synaptic_tag).clamp(0.0, 1.0);
                cell.vesicle_ready = (cell.vesicle_ready + 0.08 * (1.0 - cell.vesicle_ready)).clamp(0.0, 1.0);
                cell.synaptic_tag *= 0.94;
                cell.tag_med *= 0.78;
                cell.tag_slow *= 0.93;

                // Homeostatic TET enzyme demethylation: gently recycle cells with weak emotional anchors
                // when total locked chromatin exceeds ~72% of the pool, keeping ~28% plastic cells open!
                let max_mw = cell.motor_action_w.iter().copied().fold(0.0f32, f32::max);
                if tet_demethyl_active && cell.somatic_feeling.abs() < 0.22 && cell.synaptic_tag < 0.15 && max_mw < 0.85 {
                    cell.chromatin_locked *= 0.96;
                }

                let drift_decay = 0.012 * (1.0 - cell.chromatin_locked);
                for i in 0..d {
                    cell.axon_out_w[i] *= 1.0 - drift_decay;
                }
            }
        }

        self.prev_cell_state.copy_from_slice(s_in);
    }

    pub fn extract_cellular_experience(&self, s: &[f32], ctx: Option<&[f32]>) -> (Vec<f32>, f32) {
        let (out, feeling, _, _) = self.extract_cellular_action_and_vocal(s, ctx);
        (out, feeling)
    }

    /// Extract stored cellular experience including learned motor action (`[f32; 4]`)
    /// and vocal articulation (`[f32; 8]`). Supports both exact and partial-subspace analogy resonance!
    pub fn extract_cellular_action_and_vocal(
        &self,
        s: &[f32],
        ctx: Option<&[f32]>,
    ) -> (Vec<f32>, f32, [f32; 4], [f32; 8]) {
        let d = self.state_dim;
        let half = (d / 2).max(1);

        // First check whether `s` matches any known cell directly (`max_full >= 0.78`)
        // and whether `s` matches any motor-specialized cell directly (`max_motor_full >= 0.78`)
        let mut max_full = 0.0f32;
        let mut max_motor_full = 0.0f32;
        for cell in &self.cells {
            if l2_norm(&cell.axon_out_w) < 1e-4 && cell.chromatin_locked < 0.05 {
                continue;
            }
            let mf = dot(&cell.receptor_w, s);
            if mf > max_full {
                max_full = mf;
            }
            let max_mw = cell.motor_action_w.iter().copied().fold(0.0f32, f32::max);
            if max_mw >= 0.85 && mf > max_motor_full {
                max_motor_full = mf;
            }
        }

        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.cells.len());
        let mut motor_scores: Vec<(usize, f32)> = Vec::with_capacity(self.cells.len());
        for (c_idx, cell) in self.cells.iter().enumerate() {
            if l2_norm(&cell.axon_out_w) < 1e-4 && cell.chromatin_locked < 0.05 {
                continue;
            }
            let match_full = dot(&cell.receptor_w, s);
            let match_ctx = if let Some(c_vec) = ctx {
                if l2_norm(c_vec) > 0.1 {
                    dot(&cell.context_w, c_vec)
                } else {
                    0.5
                }
            } else {
                0.5
            };
            let ctx_gain = 0.15 + 0.85 * ((match_ctx + 1.0) * 0.5).clamp(0.0, 1.0).powi(6);

            if max_full >= 0.78 {
                if match_full >= 0.75 {
                    let res = (match_full.max(0.0)).powi(4)
                        * ctx_gain
                        * (0.3 + 0.7 * cell.chromatin_locked);
                    scores.push((c_idx, res));
                }
            } else if cell.chromatin_locked > 0.35 {
                let n_shp = (l2_norm(&cell.receptor_w[0..half]) * l2_norm(&s[0..half])) + 1e-8;
                let n_col = (l2_norm(&cell.receptor_w[half..d]) * l2_norm(&s[half..d])) + 1e-8;
                let match_sub = (dot(&cell.receptor_w[0..half], &s[0..half]) / n_shp)
                    .max(dot(&cell.receptor_w[half..d], &s[half..d]) / n_col);
                if match_sub >= 0.55 {
                    let res = (match_sub.max(0.0)).powi(6)
                        * cell.chromatin_locked
                        * (0.25 + cell.somatic_feeling.abs());
                    scores.push((c_idx, res));
                }
            }

            let max_mw = cell.motor_action_w.iter().copied().fold(0.0f32, f32::max);
            if max_motor_full >= 0.78 {
                if max_mw >= 0.85 && match_full >= 0.75 {
                    let m_res = (match_full.max(0.0)).powi(4)
                        * ctx_gain
                        * (0.3 + 0.7 * cell.chromatin_locked);
                    motor_scores.push((c_idx, m_res));
                }
            } else if max_mw >= 0.95 {
                // Zero-shot analogical motor transfer from solved puzzle/tool cells (`max_mw >= 0.95`)!
                let n_shp = (l2_norm(&cell.receptor_w[0..half]) * l2_norm(&s[0..half])) + 1e-8;
                let n_col = (l2_norm(&cell.receptor_w[half..d]) * l2_norm(&s[half..d])) + 1e-8;
                let match_sub = (dot(&cell.receptor_w[0..half], &s[0..half]) / n_shp)
                    .max(dot(&cell.receptor_w[half..d], &s[half..d]) / n_col);
                if match_sub >= 0.60 {
                    let m_res = (match_sub.max(0.0)).powi(8) * cell.chromatin_locked;
                    motor_scores.push((c_idx, m_res));
                }
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        motor_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = vec![0.0f32; d];
        let mut feeling = 0.0f32;
        let mut act = [0.25f32; 4];
        let mut voc = [0.50f32; 8];
        let mut w_sum = 0.0f32;
        let best_sc = scores.first().map(|x| x.1).unwrap_or(0.0);
        if best_sc > 1e-6 {
            act = [0.0; 4];
            voc = [0.0; 8];
        }
        for &(c_idx, sc) in scores.iter().take(6) {
            if sc < best_sc * 0.85 {
                continue;
            }
            let w = (sc / (best_sc + 1e-8)).powi(4);
            let cell = &self.cells[c_idx];
            for i in 0..d {
                out[i] += w * cell.axon_out_w[i];
            }
            for a in 0..4 {
                act[a] += w * cell.motor_action_w[a];
            }
            for v in 0..8 {
                voc[v] += w * cell.vocal_motor_w[v];
            }
            feeling += w * cell.somatic_feeling;
            w_sum += w;
        }
        if w_sum > 1e-6 {
            feeling /= w_sum;
            for a in 0..4 {
                act[a] /= w_sum;
            }
            for v in 0..8 {
                voc[v] /= w_sum;
            }
        }

        let best_msc = motor_scores.first().map(|x| x.1).unwrap_or(0.0);
        if best_msc > 1e-6 {
            act = [0.0; 4];
            let mut mw_sum = 0.0f32;
            for &(c_idx, msc) in motor_scores.iter().take(6) {
                if msc < best_msc * 0.85 {
                    continue;
                }
                let w = (msc / (best_msc + 1e-8)).powi(4);
                let cell = &self.cells[c_idx];
                for a in 0..4 {
                    act[a] += w * cell.motor_action_w[a];
                }
                mw_sum += w;
            }
            if mw_sum > 1e-6 {
                for a in 0..4 {
                    act[a] /= mw_sum;
                }
            }
        }
        (out, feeling, act, voc)
    }

    pub fn resonate_and_explain(&self, s_novel: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let d = self.state_dim;
        let half = (d / 2).max(1);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (c_idx, cell) in self.cells.iter().enumerate() {
            if cell.chromatin_locked < 0.35 || l2_norm(&cell.axon_out_w) < 1e-4 {
                continue;
            }
            let n_shp = (l2_norm(&cell.receptor_w[0..half]) * l2_norm(&s_novel[0..half])) + 1e-8;
            let n_col = (l2_norm(&cell.receptor_w[half..d]) * l2_norm(&s_novel[half..d])) + 1e-8;
            let res_shp = dot(&cell.receptor_w[0..half], &s_novel[0..half]) / n_shp;
            let res_col = dot(&cell.receptor_w[half..d], &s_novel[half..d]) / n_col;
            let res = (res_shp.max(res_col) + 0.20 * res_shp.min(res_col))
                * cell.chromatin_locked
                * (0.2 + cell.somatic_feeling.abs());
            if res > 0.05 {
                candidates.push((c_idx, res));
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut recalled_in = vec![0.0f32; d];
        let mut predicted_out = vec![0.0f32; d];
        let mut predicted_feeling = 0.0f32;
        let mut w_sum = 0.0f32;
        for &(c_idx, sc) in candidates.iter().take(6) {
            let w = sc.max(0.01);
            let cell = &self.cells[c_idx];
            for i in 0..d {
                recalled_in[i] += w * cell.receptor_w[i];
                predicted_out[i] += w * cell.axon_out_w[i];
            }
            predicted_feeling += w * cell.somatic_feeling;
            w_sum += w;
        }
        if w_sum > 1e-6 {
            for i in 0..d {
                recalled_in[i] /= w_sum;
                predicted_out[i] /= w_sum;
            }
            predicted_feeling /= w_sum;
        }

        let mut similarity_vec = vec![0.0f32; d];
        let mut difference_vec = vec![0.0f32; d];
        for i in 0..d {
            if s_novel[i] * recalled_in[i] > 0.0 {
                similarity_vec[i] = 0.5 * (s_novel[i] + recalled_in[i]);
            } else {
                difference_vec[i] = s_novel[i] - recalled_in[i];
            }
        }
        (similarity_vec, difference_vec, predicted_out, predicted_feeling)
    }

    pub fn step_whole_body_cells(
        &mut self,
        s_in: &[f32],
        a_in: &[f32],
        valence: f32,
        arousal: f32,
        action_drive: &[f32; 4],
        vocal_tract: &[f32; 8],
    ) {
        let sal_gain = (0.015 * (1.0 + valence.abs() + 0.5 * arousal)).clamp(0.005, 0.06);
        for i in 0..8 {
            let v_act = s_in.get(i * (self.state_dim / 8).max(1)).copied().unwrap_or(0.0).abs();
            self.retina_cells[i] = (self.retina_cells[i] + sal_gain * (0.85 + 0.45 * v_act - self.retina_cells[i]))
                .clamp(0.65, 1.45);

            let c_act = a_in.get(i * (a_in.len().max(8) / 8).max(1)).copied().unwrap_or(0.0).abs();
            self.cochlea_cells[i] = (self.cochlea_cells[i] + sal_gain * (0.85 + 0.45 * c_act - self.cochlea_cells[i]))
                .clamp(0.65, 1.45);

            let stress_drive = (arousal - valence).max(0.0) * (0.5 + v_act);
            self.organ_erk[i] = (0.96 * self.organ_erk[i] + 0.12 * stress_drive).clamp(0.0, 1.0);
            self.organ_cells[i] = (0.985 * self.organ_cells[i]
                + 0.04 * self.organ_erk[i] * (stress_drive - self.organ_cells[i]))
                .clamp(0.0, 1.0);

            self.muscle_cells[i] = (0.98 * self.muscle_cells[i] + 0.02 * vocal_tract[i]).clamp(0.0, 1.0);
        }

        let limb_lr = if valence >= 0.0 { 0.04 * (1.0 + valence) } else { 0.01 };
        for a in 0..4 {
            self.limb_cells[a] = (1.0 - limb_lr) * self.limb_cells[a] + limb_lr * action_drive[a];
            self.action_trace_fast[a] = 0.30 * self.action_trace_fast[a] + 0.70 * action_drive[a];
            self.action_trace_med[a] = 0.78 * self.action_trace_med[a] + 0.22 * action_drive[a];
            self.action_trace_slow[a] = 0.93 * self.action_trace_slow[a] + 0.07 * action_drive[a];
        }

        // Stamp the action decided on this tick (`t`) directly into the active ensemble's
        // multi-horizon eligibility traces (`elig_action`), so immediate (`t+1`) or delayed (`t+5`)
        // outcomes credit the exact action taken in this state!
        if valence.abs() < 0.08 {
            for &c_idx in &self.active_ensemble {
                let cell = &mut self.cells[c_idx];
                cell.elig_action[0] = *action_drive;
                for a in 0..4 {
                    cell.elig_action[1][a] = 0.45 * cell.elig_action[1][a] + 0.55 * action_drive[a];
                    cell.elig_action[2][a] = 0.55 * cell.elig_action[2][a] + 0.45 * action_drive[a];
                }
                cell.action_pending = 1.0;
            }
            self.prev_action_drive = *action_drive;
        }
        self.prev_vocal_tract = *vocal_tract;
    }

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

        self.step_epigenetic_cells(&s_in, valence, arousal);

        for i in 0..d {
            self.habituation_trace[i] = 0.82 * self.habituation_trace[i] + 0.18 * s_in[i];
        }

        let (_, diff_surprise) = self.comparator.compare(&s_in);

        let prev_fast = self.chain_fast.clone();
        let prev_norm = l2_norm(&prev_fast);

        let mut flow_pred = vec![0.0f32; d];
        if prev_norm > 1e-6 {
            for i in 0..d {
                let row = &self.w_flow[i * d..(i + 1) * d];
                flow_pred[i] = dot(row, &prev_fast);
            }
        }

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

        let mut wave = self.propagate_wave(&self.chain_fast, 2);
        if l2_norm(&self.last_recalled_outcome) > 1e-4 {
            for i in 0..d {
                wave[i] = 0.60 * wave[i] + 0.40 * self.last_recalled_outcome[i];
            }
            normalize_l2_in_place(&mut wave);
        }
        self.imagined_flow = wave;
        &self.imagined_flow
    }

    pub fn spontaneous_replay_step(&mut self, eta: f32) -> Vec<f32> {
        let d = self.state_dim;
        self.chain_steps += 1;
        let mut best_idx = 0usize;
        let mut best_drive = f32::NEG_INFINITY;
        let n_cells = self.cells.len();
        for c_idx in 0..n_cells {
            if l2_norm(&self.cells[c_idx].axon_out_w) < 1e-4 {
                continue;
            }
            let st_tag = self.cells[c_idx].synaptic_tag;
            let ch_lock = self.cells[c_idx].chromatin_locked;
            let creb = self.cells[c_idx].creb_excitability;
            let noise = 0.15 * self.next_ion_fluctuation(st_tag + (c_idx as f32));
            let drive = 0.55 * st_tag + 0.35 * ch_lock + 0.25 * creb + noise;
            if drive > best_drive {
                best_drive = drive;
                best_idx = c_idx;
            }
        }
        let s_pre = self.cells[best_idx].receptor_w.clone();
        let mut s_post = self.cells[best_idx].axon_out_w.clone();
        normalize_l2_in_place(&mut s_post);

        for i in 0..d {
            let row_w = &mut self.w_flow[i * d..(i + 1) * d];
            let pred_i = dot(row_w, &s_pre);
            let err_i = s_post[i] - pred_i;
            for j in 0..d {
                row_w[j] = (row_w[j] + eta * err_i * s_pre[j]).clamp(-2.5, 2.5);
            }
        }
        self.cells[best_idx].chromatin_locked = (self.cells[best_idx].chromatin_locked + 0.08).clamp(0.0, 0.99);
        self.cells[best_idx].creb_excitability *= 0.25;
        self.imagined_flow = s_post.clone();
        s_post
    }

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
        let _ = self.spontaneous_replay_step(eta * 0.5);
    }

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
