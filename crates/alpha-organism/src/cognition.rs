//! Unbroken Biological Experience Chain & Whole-Body Cellular Epigenetic Memory (`cognition.rs`).
//!
//! Core Biological Principles:
//! 1. **Continuous River of Consciousness (`chain_fast -> chain_slow -> chain_deep` + `w_flow`)**:
//!    Every sensation and inner feeling perturbs a multi-timescale experience chain and carves
//!    recurrent valleys into `w_flow` via decaying eligibility traces (`eligibility`).
//! 2. **Sparse Cellular Epigenetic Engrams (`cells: Vec<EpigeneticCell>`)**:
//!    To prevent the continuous river from washing rare childhood experiences into blurry soup
//!    or confusing overlapping crossroads (`A -> B -> C` vs `D -> B -> E`), experience is physically
//!    etched into a population of 256 living cells:
//!    - **Intrinsic CREB Excitability Waves**: Only a sparse ~2-4% subset of cells with high
//!      intracellular CREB allocate to any new moment.
//!    - **Retroactive Synaptic Tagging & Capture (Frey & Morris, 1997)**: Moment `t` sets a
//!      temporary phosphorylation tag (`synaptic_tag`); when a strong bodily consequence arrives
//!      at `t+1`, retrograde signaling locks the cell's chromatin (`chromatin_locked -> 0.99`),
//!      permanently shielding that cellular memory from mundane overwriting.
//!    - **Kukushkin (2024) Spaced Nuclear ERK -> CREB Coincidence**: Crammed (massed) stimulation
//!      depletes `vesicle_ready`, whereas spaced experiences allow `vesicle_ready` to recover while
//!      delayed `erk_nuclear` is primed in the nucleus, locking chromatin naturally.
//!    - **Multiplicative Apical-Basal NMDA Context Gating**: Somatic firing requires both basal
//!      sensory input (`receptor_w`) and apical context (`context_w`), separating `A -> B -> C`
//!      from `D -> B -> E`.
//! 3. **Whole-Body Non-Neural Cellular Memory (`organ_cells`, `muscle_cells`)**:
//!    Visceral organ cells learn anticipatory allostasis, and vocal muscle cells adapt their
//!    resting myofibril coordination alongside the brain.
//! 4. **Constrained Stochasticity & Saccadic Habituation (`habituation_trace`, `bio_seed`)**:
//!    Endogenous ion-channel fluctuations and Inhibition-of-Return prevent robotic repetition
//!    while homeostatic and epigenetic constraints guide intelligent behavior.
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

/// Single Living Cell with Intracellular Molecular & Epigenetic Cascade:
#[derive(Clone, Debug)]
pub struct EpigeneticCell {
    pub creb_excitability: f32,
    pub chromatin_locked: f32,
    pub synaptic_tag: f32,
    pub erk_nuclear: f32,
    pub vesicle_ready: f32,
    pub receptor_w: Vec<f32>,
    pub context_w: Vec<f32>,
    pub axon_out_w: Vec<f32>,
    pub somatic_feeling: f32,
}

impl EpigeneticCell {
    pub fn memory_bytes(&self) -> usize {
        36 + (self.receptor_w.capacity() + self.context_w.capacity() + self.axon_out_w.capacity()) * 4
    }
}

/// Unbroken Multi-Timescale Experience Chain + Sparse Cellular Epigenetic Engrams + Whole-Body Memory:
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
    /// Population of 256 living cells with intracellular CREB, ERK, and chromatin state
    pub cells: Vec<EpigeneticCell>,
    pub prev_cell_state: Vec<f32>,
    pub active_ensemble: Vec<usize>,
    /// Most recently extracted similarity features (`present * recalled > 0`)
    pub last_similarity_vec: Vec<f32>,
    /// Most recently extracted novel difference features (`present - recalled`)
    pub last_difference_vec: Vec<f32>,
    /// Most recently recalled cellular outcome & bodily feeling
    pub last_recalled_outcome: Vec<f32>,
    pub last_recalled_feeling: f32,
    /// Non-neural visceral organ cell molecular memory (anticipatory allostasis, Kukushkin 2024)
    pub organ_cells: [f32; 8],
    pub organ_erk: [f32; 8],
    /// Vocal & motor muscle cell myofibrillar baseline adaptation
    pub muscle_cells: [f32; 8],
    /// Saccadic & sensory habituation trace (Inhibition of Return to prevent robotic staring/loops)
    pub habituation_trace: Vec<f32>,
    pub gaze_trail: [[f32; 3]; 4],
    /// Endogenous ion-channel thermal fluctuation state
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
            cells.push(EpigeneticCell {
                creb_excitability: creb_init,
                chromatin_locked: 0.0,
                synaptic_tag: 0.0,
                erk_nuclear: 0.0,
                vesicle_ready: 1.0,
                receptor_w,
                context_w,
                axon_out_w: vec![0.0; state_dim],
                somatic_feeling: 0.0,
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
            active_ensemble: Vec::with_capacity(8),
            last_similarity_vec: vec![0.0f32; state_dim],
            last_difference_vec: vec![0.0f32; state_dim],
            last_recalled_outcome: vec![0.0f32; state_dim],
            last_recalled_feeling: 0.0,
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
                + self.last_similarity_vec.capacity()
                + self.last_difference_vec.capacity()
                + self.last_recalled_outcome.capacity()
                + self.habituation_trace.capacity())
                * 4
    }

    /// Endogenous ion-channel membrane fluctuation (`[-1.0, 1.0]`):
    /// Perturbed continuously by the organism's live sensory/visceral state so its trajectory
    /// is non-repetitive and genuinely alive while remaining bounded by cellular attractors.
    pub fn next_ion_fluctuation(&mut self, perturbation: f32) -> f32 {
        let bits = (perturbation.to_bits() as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        self.bio_seed = self.bio_seed.wrapping_add(0x9E37_79B9_7F4A_7C15 ^ bits);
        let mut z = self.bio_seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / ((1u64 << 23) as f32)) - 1.0
    }

    /// Pool full-dimensional IT shape (`s`, e.g. 256D) and C4 colour (`c`, e.g. 64D) assemblies
    /// into a zero-mean Excitatory-Inhibitory (E-I) balanced cortical vector of dimension `self.state_dim`.
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

    /// Step the population of 256 living epigenetic cells:
    /// 1. Retroactive Synaptic Tagging & Capture (Frey & Morris): cells in `active_ensemble` (from `t-1`)
    ///    wire their `axon_out_w` to `s_in(t)` and lock their chromatin if `salience` or spaced
    ///    `vesicle_ready * erk_nuclear` (Kukushkin 2024) is high.
    /// 2. Multiplicative Apical-Basal NMDA Context Gating: recruits `k = 6` cells matching both
    ///    current `s_in` and preceding context `prev_cell_state`.
    pub fn step_epigenetic_cells(&mut self, s_in: &[f32], valence: f32, arousal: f32) {
        let d = self.state_dim;
        let salience = (valence.abs() + 0.5 * arousal).clamp(0.0, 2.0);
        let n_cells = self.cells.len();

        // First extract similarity/difference & recalled outcome BEFORE modifying the cells with `s_in`
        let (sim_v, diff_v, rec_out, rec_feel) = self.resonate_and_explain(s_in);
        self.last_similarity_vec = sim_v;
        self.last_difference_vec = diff_v;
        self.last_recalled_outcome = rec_out;
        self.last_recalled_feeling = rec_feel;

        // 1. Retroactive STC update on the ensemble that fired at `t-1`
        for &c_idx in &self.active_ensemble {
            let cell = &mut self.cells[c_idx];
            let shield = if salience > 0.50 {
                1.0 - 0.50 * cell.chromatin_locked
            } else {
                (1.0 - 0.98 * cell.chromatin_locked).max(0.0)
            };
            let lr_out = (0.45 * (0.3 + salience) * shield).clamp(0.0, 0.85);
            if lr_out > 1e-4 {
                for i in 0..d {
                    cell.axon_out_w[i] = (1.0 - lr_out) * cell.axon_out_w[i] + lr_out * s_in[i];
                }
                cell.somatic_feeling = (1.0 - lr_out) * cell.somatic_feeling + lr_out * valence;
            }
            let kukushkin_creb = cell.vesicle_ready * cell.erk_nuclear;
            if salience > 0.50 || kukushkin_creb > 0.18 {
                let boost = if salience > 0.50 {
                    0.55 + 0.35 * salience
                } else {
                    0.85 * kukushkin_creb
                };
                cell.chromatin_locked = (cell.chromatin_locked + boost).clamp(0.0, 0.99);
            }
            cell.vesicle_ready *= 0.22;
        }

        // 2. Multiplicative NMDA context-gated ensemble allocation for `s_in`
        let has_ctx = l2_norm(&self.prev_cell_state) > 0.1;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_cells);
        for (c_idx, cell) in self.cells.iter().enumerate() {
            let match_s = dot(&cell.receptor_w, s_in);
            let match_ctx = if has_ctx {
                dot(&cell.context_w, &self.prev_cell_state)
            } else {
                0.5
            };
            let basal = (match_s + 1.0).max(0.0);
            let apical = (match_ctx + 1.0).max(0.0);
            let nmda_coincidence = basal * apical;

            let allocation_drive = if match_s > 0.72 && (!has_ctx || match_ctx > 0.25) {
                nmda_coincidence * 1.15
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
            let lr_in = 0.45 * (1.0 - 0.95 * cell.chromatin_locked);
            if lr_in > 1e-3 {
                for i in 0..d {
                    cell.receptor_w[i] += lr_in * (s_in[i] - cell.receptor_w[i]);
                    if has_ctx {
                        cell.context_w[i] += lr_in * (self.prev_cell_state[i] - cell.context_w[i]);
                    }
                }
                normalize_l2_in_place(&mut cell.receptor_w);
                normalize_l2_in_place(&mut cell.context_w);
            }
            cell.synaptic_tag = (cell.synaptic_tag + (0.45 + 0.35 * salience) * cell.vesicle_ready).clamp(0.0, 1.0);
            cell.creb_excitability *= 0.10;
        }

        let step_u = self.chain_steps as usize;
        for (c_idx, cell) in self.cells.iter_mut().enumerate() {
            let is_active = self.active_ensemble.iter().any(|&x| x == c_idx);
            if !is_active {
                let wave = (((step_u * 17 + c_idx * 11) % 97) as f32) / 97.0;
                cell.creb_excitability = (0.90 * cell.creb_excitability + 0.10 * wave).clamp(0.0, 1.0);
                cell.erk_nuclear = (0.96 * cell.erk_nuclear + 0.08 * cell.synaptic_tag).clamp(0.0, 1.0);
                cell.vesicle_ready = (cell.vesicle_ready + 0.08 * (1.0 - cell.vesicle_ready)).clamp(0.0, 1.0);
                cell.synaptic_tag *= 0.95;
                let drift_decay = 0.012 * (1.0 - cell.chromatin_locked);
                for i in 0..d {
                    cell.axon_out_w[i] *= 1.0 - drift_decay;
                }
            }
        }

        self.prev_cell_state.copy_from_slice(s_in);
    }

    /// Reactivate / extract stored cellular experience from a cue `s` (and optional preceding context `ctx`).
    pub fn extract_cellular_experience(&self, s: &[f32], ctx: Option<&[f32]>) -> (Vec<f32>, f32) {
        let d = self.state_dim;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.cells.len());
        for (c_idx, cell) in self.cells.iter().enumerate() {
            if l2_norm(&cell.axon_out_w) < 1e-4 {
                continue;
            }
            let match_s = dot(&cell.receptor_w, s);
            if match_s < 0.45 {
                continue;
            }
            let match_ctx = if let Some(c_vec) = ctx {
                dot(&cell.context_w, c_vec)
            } else {
                0.5
            };
            let res = (match_s.max(0.0)) * ((match_ctx + 1.0).max(0.1)) * (0.5 + 0.5 * cell.chromatin_locked);
            scores.push((c_idx, res));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = vec![0.0f32; d];
        let mut feeling = 0.0f32;
        let mut w_sum = 0.0f32;
        for &(c_idx, sc) in scores.iter().take(6) {
            let w = sc.powi(2);
            let cell = &self.cells[c_idx];
            for i in 0..d {
                out[i] += w * cell.axon_out_w[i];
            }
            feeling += w * cell.somatic_feeling;
            w_sum += w;
        }
        if w_sum > 1e-6 {
            feeling /= w_sum;
        }
        (out, feeling)
    }

    /// Given a novel or familiar experience `s_novel`, resonate with chromatin-locked cellular engrams
    /// by partial subspace similarity (shape `0..half` or colour `half..dim`), and decompose into:
    /// - `similarity_vec`: where present perception agrees with recalled cellular engram (shared principle)
    /// - `difference_vec`: where present perception contrasts with recalled engram (novel aspect)
    /// - `predicted_out`: expected outcome by analogy
    /// - `predicted_feeling`: expected bodily valence by analogy
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

    /// Update whole-body non-neural organ cells and vocal muscle cells (Kukushkin 2024):
    pub fn step_whole_body_cells(&mut self, s_in: &[f32], valence: f32, arousal: f32, vocal_tract: &[f32; 8]) {
        for i in 0..8 {
            let s_val = s_in.get(i * (self.state_dim / 8).max(1)).copied().unwrap_or(0.0);
            let stress_drive = (arousal - valence).max(0.0) * (0.5 + s_val.abs());
            self.organ_erk[i] = (0.96 * self.organ_erk[i] + 0.12 * stress_drive).clamp(0.0, 1.0);
            self.organ_cells[i] = (0.985 * self.organ_cells[i]
                + 0.04 * self.organ_erk[i] * (stress_drive - self.organ_cells[i]))
                .clamp(0.0, 1.0);
            self.muscle_cells[i] = (0.98 * self.muscle_cells[i] + 0.02 * vocal_tract[i]).clamp(0.0, 1.0);
        }
    }

    /// Advance both the continuous Experience Chain (`chain_fast -> chain_slow -> chain_deep` + `w_flow`)
    /// AND the 256 living Epigenetic Cells (`step_epigenetic_cells`).
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

        // Step the sparse cellular epigenetic population & extract similarity/difference
        self.step_epigenetic_cells(&s_in, valence, arousal);

        // Update saccadic/feature habituation trace (fatigues repeated states so the infant explores!)
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

        // Blend recurrent wave propagation with any recalled cellular outcome
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

    /// Spontaneous quiet-rest / sleep cellular replay (Sharp-Wave Ripple analogue):
    /// When the external room is dark/silent, high-tag or chromatin-locked cells spontaneously
    /// ignite and replay their `receptor_w -> axon_out_w` transition into `w_flow` and `imagined_flow`.
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

        // Consolidate replayed cellular transition into the cortical flow matrix `w_flow`
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
