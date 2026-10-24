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

pub const CELL_TISSUE_RETINA_MEANING: u8 = 0;
pub const CELL_TISSUE_COCHLEA_WORD: u8 = 1;
pub const CELL_TISSUE_VISCERAL_DRIVE: u8 = 2;
pub const CELL_TISSUE_LIMB_MOTOR: u8 = 3;
pub const CELL_TISSUE_CONCEPT_CLUSTER: u8 = 4;

/// Channel 3 (Brain -> Cells): Chemical / Neuromodulatory Broadcast across all living cells.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModulationBroadcast {
    pub surprise: f32,
    pub reward: f32,
    pub danger: f32,
    pub rest: f32,
}

/// Channel 4 (Cells -> Brain): Short status report from the living cells to the Waking Brain.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellReport {
    pub matched: bool,
    pub unknown: bool,
    pub conflict: bool,
    pub emotional: bool,
    pub confidence: f32,
    pub active_count: usize,
    pub brain_woke_up: bool,
}

/// Channel 2 (Brain -> Cells Readout): Result of sending a cue into the living cell population.
#[derive(Clone, Debug)]
pub struct CellReadout {
    pub recalled_meaning: Vec<f32>,
    pub recalled_word_pat: Vec<f32>,
    pub recalled_action: [f32; 4],
    pub recalled_vocal: [f32; 8],
    pub second_word_vocal: Option<[f32; 8]>,
    pub recalled_valence: f32,
    pub active_cell_ids: Vec<u64>,
    pub report: CellReport,
}

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

/// Homoiconic Epigenetic Regulatory S-Expression (`RegGeneExpr` — Code-as-Data inside every Cell):
/// Compact 7-node prefix tree (`(op_root (op_left a b) (op_right c d))`) over biophysical signals:
/// Terminals `0..5`: `[basal_conductance, hunger_drive, energy_atp, vesicle_rrp, somatic_valence, surprise_ca2]`
/// Combinators `6..9`: `[6=avg_sum, 7=coincidence_mul, 8=hill_saturate, 9=homeostatic_diff]`
#[derive(Clone, Copy, Debug)]
pub struct RegGeneExpr {
    pub ops: [u8; 7],
}

impl Default for RegGeneExpr {
    fn default() -> Self {
        // Canonical biophysical regulatory tree: avg(mul(basal, hunger), hill(energy, vesicle))
        Self {
            ops: [6, 7, 8, 0, 1, 2, 3],
        }
    }
}

impl RegGeneExpr {
    #[inline]
    fn eval_leaf(op: u8, signals: &[f32; 6]) -> f32 {
        signals[(op as usize) % 6].clamp(0.0, 2.0)
    }

    #[inline]
    fn eval_bin(op: u8, x: f32, y: f32) -> f32 {
        match op.saturating_sub(6) % 4 {
            0 => 0.5 * (x + y),
            1 => (x * y).clamp(0.0, 2.0),
            2 => {
                let num = x * x + 0.35 * y;
                (num / (0.25 + num)).clamp(0.0, 1.5)
            }
            _ => (x * (0.65 + 0.35 * (1.0 - (x - y).abs().min(1.0)))).clamp(0.0, 2.0),
        }
    }

    /// Evaluate the cell's homoiconic regulatory program in O(1) with zero heap allocations:
    #[inline]
    pub fn eval_biophysical(&self, signals: &[f32; 6]) -> f32 {
        let l0 = Self::eval_leaf(self.ops[3], signals);
        let l1 = Self::eval_leaf(self.ops[4], signals);
        let r0 = Self::eval_leaf(self.ops[5], signals);
        let r1 = Self::eval_leaf(self.ops[6], signals);
        let left = Self::eval_bin(self.ops[1], l0, l1);
        let right = Self::eval_bin(self.ops[2], r0, r1);
        let out = Self::eval_bin(self.ops[0], left, right);
        if out.is_finite() {
            (0.72 + 0.32 * out).clamp(0.55, 1.30)
        } else {
            1.0
        }
    }

    /// Homoiconic subform point-mutation (`patch_subform`, inspired by Lisp `(patch name old new)`):
    /// Mutates exactly one operator or terminal node while preserving valid S-expression arity.
    #[inline]
    pub fn patch_subform(&self, locus_seed: u64) -> Self {
        let mut child = *self;
        let slot = (locus_seed as usize) % 7;
        if slot < 3 {
            child.ops[slot] = 6 + (((locus_seed >> 3) as u8) % 4);
        } else {
            child.ops[slot] = ((locus_seed >> 5) as u8) % 6;
        }
        child
    }
}

/// Single Living Cell with Multi-Scale Action Eligibility, Self-Supervised Delta, Epigenetics,
/// Homoiconic `RegGeneExpr` + Provisional Escrow Self-Healing, and Full Ecological Lifecycle:
#[derive(Clone, Debug)]
pub struct EpigeneticCell {
    pub id: u64,
    pub tissue_kind: u8,
    pub layer: u8,
    pub energy: f32,
    pub fullness: f32,
    pub variance_w: Vec<f32>,
    pub variety_score: f32,
    pub similarity_limit: f32,
    pub decay_rate: f32,
    pub fed_count: u32,
    pub pos: [f32; 3],
    pub co_bonds: Vec<(u64, f32)>,
    pub time_bonds: Vec<(u64, f32)>,
    pub parent_id: Option<u64>,
    pub child_ids: Vec<u64>,
    pub creb_excitability: f32,
    pub chromatin_locked: f32,
    /// Multi-scale intracellular eligibility tags (`fast` ~1-2 ticks, `med` ~4-8 ticks, `slow` ~12-25 ticks)
    pub synaptic_tag: f32,
    pub tag_med: f32,
    pub tag_slow: f32,
    pub erk_nuclear: f32,
    pub vesicle_ready: f32,
    /// Homoiconic regulatory S-expression + Escrowed parent backup (`provisional_hold`: 0=sealed, 1=held)
    pub gene_expr: RegGeneExpr,
    pub escrow_gene: RegGeneExpr,
    pub provisional_hold: u8,
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
        236 + (self.receptor_w.capacity()
            + self.context_w.capacity()
            + self.axon_out_w.capacity()
            + self.world_delta_w.capacity()
            + self.variance_w.capacity())
            * 4
            + (self.co_bonds.capacity() + self.time_bonds.capacity()) * 12
            + self.child_ids.capacity() * 8
    }

    /// Biophysical Dendritic Ionotropic (AMPA) + Voltage-Unblocked NMDA Coincidence Conductance
    /// normalized by Carandini-Heeger Divisive Pool Energy (replaces unbiological raw cosine lookup):
    #[inline]
    pub fn ionotropic_conductance(&self, afferent_cue: &[f32]) -> f32 {
        let len = self.receptor_w.len().min(afferent_cue.len());
        if len == 0 {
            return 0.0;
        }
        let mut g_ampa = 0.0f32;
        let mut r_energy = 0.0f32;
        let mut c_energy = 0.0f32;
        for i in 0..len {
            let rw = self.receptor_w[i];
            let cv = afferent_cue[i];
            g_ampa += rw * cv;
            r_energy += rw * rw;
            c_energy += cv * cv;
        }
        // Carandini-Heeger divisive normalization across dendritic spine pool (sigma^2 = 1e-6)
        let denom = ((1e-6 + r_energy) * (1e-6 + c_energy)).sqrt();
        let norm_g = (g_ampa / denom).clamp(-1.0, 1.0);
        // Presynaptic Readily-Releasable Vesicle Pool (RRP) + PSD-95 chromatin stabilization gain
        let rrp_gain = (0.95 + 0.05 * self.vesicle_ready).clamp(0.92, 1.02);
        (norm_g * rrp_gain).clamp(-1.0, 1.0)
    }

    pub fn add_or_strengthen_co_bond(&mut self, other_id: u64, delta: f32) {
        if other_id == self.id {
            return;
        }
        for (id, w) in &mut self.co_bonds {
            if *id == other_id {
                *w = (*w + delta).clamp(0.0, 2.5);
                return;
            }
        }
        self.co_bonds.push((other_id, delta.clamp(0.0, 2.5)));
    }

    pub fn add_or_strengthen_time_bond(&mut self, next_id: u64, delta: f32) {
        if next_id == self.id {
            return;
        }
        for (id, w) in &mut self.time_bonds {
            if *id == next_id {
                *w = (*w + delta).clamp(0.0, 2.5);
                return;
            }
        }
        self.time_bonds.push((next_id, delta.clamp(0.0, 2.5)));
    }
}

/// Unbroken Multi-Timescale Experience Chain + Living Epigenetic Cell Population + 5-Tissue Whole-Body Memory:
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
    pub basal_columns: usize,
    pub next_cell_id: u64,
    pub max_cells: usize,
    pub total_energy_budget: f32,
    pub variety_split_limit: f32,
    pub merge_similarity_limit: f32,
    pub prev_active_cell_ids: Vec<u64>,
    pub last_cell_report: CellReport,
    pub cells_born: u64,
    pub cells_split: u64,
    pub cells_merged: u64,
    pub cells_died_handoff: u64,
    pub dream_bridges_formed: u64,
    pub cheap_cell_steps: u64,
    pub waking_brain_steps: u64,
    /// Locked survival floor (`basal_columns` = `kernel`) vs mutable `evolved` cells + provisional lineage:
    pub floor_locked: bool,
    pub provisional_commits: u64,
    pub provisional_reverts: u64,
    pub sealed_generation: u64,
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
        let basal_columns = 256usize;
        let mut cells = Vec::with_capacity(basal_columns + 64);
        let mut next_cell_id = 1u64;
        for c in 0..basal_columns {
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
            let cid = next_cell_id;
            next_cell_id += 1;
            cells.push(EpigeneticCell {
                id: cid,
                tissue_kind: CELL_TISSUE_RETINA_MEANING,
                layer: 0,
                energy: 0.55,
                fullness: 0.0,
                variance_w: vec![0.0; state_dim],
                variety_score: 0.0,
                similarity_limit: 0.75,
                decay_rate: 0.008,
                fed_count: 0,
                pos: [
                    ((c % 16) as f32) / 8.0 - 1.0,
                    ((c / 16) as f32) / 8.0 - 1.0,
                    0.0,
                ],
                co_bonds: Vec::new(),
                time_bonds: Vec::new(),
                parent_id: None,
                child_ids: Vec::new(),
                creb_excitability: creb_init,
                chromatin_locked: 0.0,
                synaptic_tag: 0.0,
                tag_med: 0.0,
                tag_slow: 0.0,
                erk_nuclear: 0.0,
                vesicle_ready: 1.0,
                gene_expr: RegGeneExpr::default(),
                escrow_gene: RegGeneExpr::default(),
                provisional_hold: 0,
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

        // Seed 12 embryonic dynamic tissue cells (4 Meaning/Retina, 4 Word/Cochlea, 4 Drive/Visceral)
        for e in 0..12usize {
            let kind = match e % 3 {
                0 => CELL_TISSUE_RETINA_MEANING,
                1 => CELL_TISSUE_COCHLEA_WORD,
                _ => CELL_TISSUE_VISCERAL_DRIVE,
            };
            let mut receptor_w = vec![0.0f32; state_dim];
            for d in 0..state_dim {
                receptor_w[d] = (((e * 43 + d * 29 + 7) % 101) as f32 / 50.0) - 1.0;
            }
            normalize_l2_in_place(&mut receptor_w);
            let cid = next_cell_id;
            next_cell_id += 1;
            cells.push(EpigeneticCell {
                id: cid,
                tissue_kind: kind,
                layer: 0,
                energy: 0.45,
                fullness: 0.0,
                variance_w: vec![0.0; state_dim],
                variety_score: 0.0,
                similarity_limit: if kind == CELL_TISSUE_VISCERAL_DRIVE { 0.65 } else { 0.76 },
                decay_rate: 0.012,
                fed_count: 1,
                pos: [
                    ((e % 4) as f32) * 0.4 - 0.6,
                    ((e / 4) as f32) * 0.4 - 0.4,
                    (kind as f32) * 0.3,
                ],
                co_bonds: Vec::new(),
                time_bonds: Vec::new(),
                parent_id: None,
                child_ids: Vec::new(),
                creb_excitability: 0.5,
                chromatin_locked: 0.0,
                synaptic_tag: 0.0,
                tag_med: 0.0,
                tag_slow: 0.0,
                erk_nuclear: 0.0,
                vesicle_ready: 1.0,
                gene_expr: RegGeneExpr::default(),
                escrow_gene: RegGeneExpr::default(),
                provisional_hold: 0,
                receptor_w: receptor_w.clone(),
                context_w: receptor_w,
                axon_out_w: vec![0.0; state_dim],
                world_delta_w: vec![0.0; state_dim],
                somatic_feeling: 0.0,
                elig_action: [[0.25; 4]; 3],
                action_pending: 0.0,
                motor_action_w: [0.25; 4],
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
            basal_columns,
            next_cell_id,
            max_cells: 448,
            total_energy_budget: 140.0,
            variety_split_limit: 0.145,
            merge_similarity_limit: 0.965,
            prev_active_cell_ids: Vec::new(),
            last_cell_report: CellReport::default(),
            cells_born: 12,
            cells_split: 0,
            cells_merged: 0,
            cells_died_handoff: 0,
            dream_bridges_formed: 0,
            cheap_cell_steps: 0,
            waking_brain_steps: 0,
            floor_locked: true,
            provisional_commits: 0,
            provisional_reverts: 0,
            sealed_generation: 0,
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
        let n_cells = self.basal_columns.min(self.cells.len());

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
        for (c_idx, cell) in self.cells.iter().take(n_cells).enumerate() {
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
            cell.energy = (cell.energy + 0.08).clamp(0.10, 1.50);
            cell.fullness = (cell.fullness + 0.25).clamp(0.0, 1.0);
            cell.fed_count = cell.fed_count.saturating_add(1);
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
        let locked_count = self
            .cells
            .iter()
            .take(n_cells)
            .filter(|c| c.chromatin_locked > 0.5)
            .count();
        let tet_demethyl_active = locked_count > (n_cells * 185) / 256;

        for (c_idx, cell) in self.cells.iter_mut().take(n_cells).enumerate() {
            let is_active = self.active_ensemble.iter().any(|&x| x == c_idx);
            if !is_active {
                let wave = (((step_u * 17 + c_idx * 11) % 97) as f32) / 97.0;
                cell.creb_excitability = (0.90 * cell.creb_excitability + 0.10 * wave).clamp(0.0, 1.0);
                cell.erk_nuclear = (0.96 * cell.erk_nuclear + 0.08 * cell.synaptic_tag).clamp(0.0, 1.0);
                cell.vesicle_ready = (cell.vesicle_ready + 0.08 * (1.0 - cell.vesicle_ready)).clamp(0.0, 1.0);
                cell.synaptic_tag *= 0.94;
                cell.tag_med *= 0.78;
                cell.tag_slow *= 0.93;
                cell.fullness = (cell.fullness * 0.65 - 0.04).clamp(0.0, 1.0);

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
        let n_basal = self.basal_columns.min(self.cells.len());

        // First check whether `s` matches any known cell directly (`max_full >= 0.78`)
        // and whether `s` matches any motor-specialized cell directly (`max_motor_full >= 0.78`)
        let mut max_full = 0.0f32;
        let mut max_motor_full = 0.0f32;
        for cell in self.cells.iter().take(n_basal) {
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

        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_basal);
        let mut motor_scores: Vec<(usize, f32)> = Vec::with_capacity(n_basal);
        for (c_idx, cell) in self.cells.iter().take(n_basal).enumerate() {
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
        let n_basal = self.basal_columns.min(self.cells.len());
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (c_idx, cell) in self.cells.iter().take(n_basal).enumerate() {
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
        let _ = self.dream_living_cells(3);
        let mut best_idx = 0usize;
        let mut best_drive = f32::NEG_INFINITY;
        let n_cells = self.basal_columns.min(self.cells.len());
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

    #[inline]
    pub fn find_cell_idx(&self, cell_id: u64) -> Option<usize> {
        self.cells.iter().position(|c| c.id == cell_id)
    }

    /// Channel 2 (Brain -> Cells Readout) + Channel 4 (Cells -> Brain Report):
    /// Sends sensory/word/drive cues into the living `EpigeneticCell` population (`self.cells`),
    /// runs within-modality lateral inhibition and associative spreading along `co_bonds`,
    /// orders multi-word sequences via `time_bonds` + `fullness`, and reports whether the
    /// Waking Brain needs to wake up (`report.brain_woke_up`).
    pub fn read_living_cells(
        &mut self,
        meaning_cue: Option<&[f32]>,
        word_cue: Option<&[f32]>,
        drive_cue: Option<&[f32]>,
        broadcast: ModulationBroadcast,
    ) -> CellReadout {
        let start = self.basal_columns.min(self.cells.len());
        let n = self.cells.len();
        let mut raw_excite = vec![0.0f32; n];
        let mut best_meaning_sim = 0.0f32;
        let mut best_word_sim = 0.0f32;
        let mut has_input_cue = false;

        for idx in start..n {
            let c = &self.cells[idx];
            let hunger_gate = (1.0 - 0.55 * c.fullness).clamp(0.15, 1.0);
            let mut sim = 0.0f32;
            match c.tissue_kind {
                CELL_TISSUE_RETINA_MEANING | CELL_TISSUE_CONCEPT_CLUSTER => {
                    if let Some(m_vec) = meaning_cue {
                        if l2_norm(m_vec) > 0.05 {
                            has_input_cue = true;
                            let s = c.ionotropic_conductance(m_vec);
                            if c.tissue_kind == CELL_TISSUE_RETINA_MEANING && s > best_meaning_sim {
                                best_meaning_sim = s;
                            }
                            if s >= c.similarity_limit - 0.06 {
                                sim = s;
                            }
                        }
                    }
                }
                CELL_TISSUE_COCHLEA_WORD => {
                    if let Some(w_vec) = word_cue {
                        if l2_norm(w_vec) > 0.05 {
                            has_input_cue = true;
                            let s = c.ionotropic_conductance(w_vec);
                            if s > best_word_sim {
                                best_word_sim = s;
                            }
                            if s >= c.similarity_limit - 0.06 {
                                sim = s;
                            }
                        }
                    }
                }
                CELL_TISSUE_VISCERAL_DRIVE => {
                    if let Some(d_vec) = drive_cue {
                        if l2_norm(d_vec) > 0.05 {
                            has_input_cue = true;
                            let s = c.ionotropic_conductance(d_vec);
                            if s >= c.similarity_limit - 0.15 {
                                sim = s;
                            }
                        }
                    }
                }
                _ => {}
            }
            if sim > 0.0 {
                let gene_gain = c.gene_expr.eval_biophysical(&[
                    sim,
                    hunger_gate,
                    c.energy,
                    c.vesicle_ready,
                    c.somatic_feeling.abs(),
                    broadcast.surprise,
                ]);
                raw_excite[idx] = sim * hunger_gate * (0.45 + 0.55 * c.energy) * gene_gain;
            }
        }

        // Associative spreading across `co_bonds` (steered by active Meaning and Drive clusters):
        // Note: `time_bonds` ONLY boost a target word cell if that word cell ALREADY has
        // `co_bond` support from an active meaning/drive cluster in the room!
        let mut spread_co = vec![0.0f32; n];
        for idx in start..n {
            let a_i = raw_excite[idx];
            if a_i <= 0.05 {
                continue;
            }
            let co_list = self.cells[idx].co_bonds.clone();
            for (nid, w_co) in co_list {
                if let Some(n_idx) = self.find_cell_idx(nid) {
                    if n_idx >= start {
                        let h_gate = (1.0 - 0.50 * self.cells[n_idx].fullness).clamp(0.15, 1.0);
                        spread_co[n_idx] += 0.62 * a_i * w_co * h_gate;
                    }
                }
            }
        }
        for idx in start..n {
            raw_excite[idx] = (raw_excite[idx] + spread_co[idx]).clamp(0.0, 3.0);
        }

        // Second wave: time-ordered bonds (`time_bonds`) boost word cells ONLY if they are
        // already supported by the active meaning/drive cluster (`raw_excite[n_idx] > 0.08`)
        let mut spread_time = vec![0.0f32; n];
        for idx in start..n {
            let a_i = raw_excite[idx];
            if a_i <= 0.05 {
                continue;
            }
            let t_list = self.cells[idx].time_bonds.clone();
            for (nid, w_t) in t_list {
                if let Some(n_idx) = self.find_cell_idx(nid) {
                    if n_idx >= start && raw_excite[n_idx] > 0.08 {
                        let h_gate = (1.0 - 0.65 * self.cells[n_idx].fullness).clamp(0.10, 1.0);
                        spread_time[n_idx] += 0.35 * a_i * w_t * h_gate;
                    }
                }
            }
        }
        for idx in start..n {
            raw_excite[idx] = (raw_excite[idx] + spread_time[idx]).clamp(0.0, 3.0);
        }

        // Global lateral inhibition WITHIN each tissue role so high-amplitude visual cells never
        // silence associatively-excited word cells in a silent room
        let mut ranked: Vec<(usize, f32)> = Vec::new();
        for kind_filter in [
            CELL_TISSUE_RETINA_MEANING,
            CELL_TISSUE_COCHLEA_WORD,
            CELL_TISSUE_VISCERAL_DRIVE,
            CELL_TISSUE_CONCEPT_CLUSTER,
        ] {
            let mut group: Vec<(usize, f32)> = (start..n)
                .filter(|&i| self.cells[i].tissue_kind == kind_filter && raw_excite[i] > 0.06)
                .map(|i| (i, raw_excite[i]))
                .collect();
            group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let g_max = group.first().map(|x| x.1).unwrap_or(0.0);
            for &(idx, sc) in group.iter().take(4) {
                if sc >= g_max * 0.18 {
                    ranked.push((idx, sc));
                }
            }
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut recalled_meaning = vec![0.0f32; self.state_dim];
        let mut recalled_word_pat = vec![0.0f32; self.state_dim];
        let mut recalled_action = [0.25f32; 4];
        let mut recalled_vocal = [0.50f32; 8];
        let mut second_word_vocal: Option<[f32; 8]> = None;
        let mut recalled_valence = 0.0f32;
        let mut active_cell_ids = Vec::new();

        let mut m_w_sum = 0.0f32;
        let mut act_w_sum = 0.0f32;
        let mut word_candidates: Vec<(usize, f32)> = Vec::new();

        for &(idx, sc) in &ranked {
            let c = &self.cells[idx];
            active_cell_ids.push(c.id);
            match c.tissue_kind {
                CELL_TISSUE_RETINA_MEANING | CELL_TISSUE_CONCEPT_CLUSTER => {
                    for d in 0..self.state_dim {
                        recalled_meaning[d] += sc * c.receptor_w[d];
                    }
                    m_w_sum += sc;
                    if act_w_sum == 0.0 {
                        recalled_action = [0.0; 4];
                    }
                    for a in 0..4 {
                        recalled_action[a] += sc * c.motor_action_w[a];
                    }
                    recalled_valence += sc * c.somatic_feeling;
                    act_w_sum += sc;
                }
                CELL_TISSUE_COCHLEA_WORD => {
                    word_candidates.push((idx, sc));
                }
                _ => {}
            }
        }

        if m_w_sum > 1e-6 {
            normalize_l2_in_place(&mut recalled_meaning);
        }
        if act_w_sum > 1e-6 {
            for a in 0..4 {
                recalled_action[a] /= act_w_sum;
            }
            recalled_valence /= act_w_sum;
        }

        if !word_candidates.is_empty() {
            let first_pos = if word_candidates.len() >= 2 {
                let mut best_first_pos = 0usize;
                let mut best_order_score = f32::NEG_INFINITY;
                for (pos_i, &(w_idx, w_sc)) in word_candidates.iter().enumerate() {
                    let order_sc = self.compute_sequence_order_score(w_idx, w_sc, &word_candidates);
                    if order_sc > best_order_score {
                        best_order_score = order_sc;
                        best_first_pos = pos_i;
                    }
                }
                best_first_pos
            } else {
                0usize
            };

            let (w1_idx, _) = word_candidates[first_pos];
            recalled_word_pat.copy_from_slice(&self.cells[w1_idx].receptor_w);
            recalled_vocal = self.cells[w1_idx].vocal_motor_w;
            self.cells[w1_idx].fullness = (self.cells[w1_idx].fullness + 0.35).clamp(0.0, 1.0);

            // Second word candidate in a two-word phrase (`"want cat"`):
            // Must be a distinct word pattern (`dot < 0.82`) excited by the active meaning cluster
            for (pos_j, &(w2_idx, w2_sc)) in word_candidates.iter().enumerate() {
                if pos_j == first_pos {
                    continue;
                }
                let pat_sim = dot(&self.cells[w1_idx].receptor_w, &self.cells[w2_idx].receptor_w);
                if pat_sim < 0.82 && w2_sc > 0.05 {
                    second_word_vocal = Some(self.cells[w2_idx].vocal_motor_w);
                    self.cells[w2_idx].fullness = (self.cells[w2_idx].fullness + 0.25).clamp(0.0, 1.0);
                    break;
                }
            }
        }

        let best_cue_sim = best_meaning_sim.max(best_word_sim);
        let matched = !active_cell_ids.is_empty() && best_cue_sim >= 0.72;
        let unknown = has_input_cue && !matched;
        let conflict = ranked.len() >= 2
            && self.cells[ranked[0].0].tissue_kind == self.cells[ranked[1].0].tissue_kind
            && (ranked[0].1 - ranked[1].1).abs() < 0.06
            && dot(&self.cells[ranked[0].0].receptor_w, &self.cells[ranked[1].0].receptor_w) < 0.50;
        let emotional = recalled_valence.abs() > 0.25
            || broadcast.reward > 0.20
            || broadcast.danger > 0.20;
        let brain_woke_up = unknown || conflict || broadcast.surprise > 0.45;
        if brain_woke_up {
            self.waking_brain_steps += 1;
        } else if matched {
            self.cheap_cell_steps += 1;
        }

        let report = CellReport {
            matched,
            unknown,
            conflict,
            emotional,
            confidence: best_cue_sim.clamp(0.0, 1.0),
            active_count: active_cell_ids.len(),
            brain_woke_up,
        };
        self.last_cell_report = report;

        CellReadout {
            recalled_meaning,
            recalled_word_pat,
            recalled_action,
            recalled_vocal,
            second_word_vocal,
            recalled_valence,
            active_cell_ids,
            report,
        }
    }

    fn compute_sequence_order_score(
        &self,
        w_idx: usize,
        w_sc: f32,
        candidates: &[(usize, f32)],
    ) -> f32 {
        let c = &self.cells[w_idx];
        let mut score = 0.25 * w_sc * (1.0 - 0.5 * c.fullness);
        // Outgoing time_bonds (`A -> B`) indicate an initiator/prefix word (like `"want"`)
        let out_time_sum: f32 = c.time_bonds.iter().map(|(_, w)| *w).sum();
        score += 0.45 * out_time_sum.min(2.5);

        // Check if this word cell is bonded to an active Drive cell (`Drive("want") <-> Word("want")`),
        // where Drive cells naturally initiate request sequences before object-meaning words
        for &(co_id, w_co) in &c.co_bonds {
            if let Some(co_idx) = self.find_cell_idx(co_id) {
                if self.cells[co_idx].tissue_kind == CELL_TISSUE_VISCERAL_DRIVE {
                    score += 0.85 * w_co;
                }
            }
        }

        for &(other_idx, _) in candidates {
            if other_idx == w_idx {
                continue;
            }
            let other_id = self.cells[other_idx].id;
            for &(tid, tw) in &c.time_bonds {
                if tid == other_id {
                    score += 1.20 * tw;
                }
            }
            for &(tid, tw) in &self.cells[other_idx].time_bonds {
                if tid == c.id {
                    score -= 1.20 * tw;
                }
            }
        }
        score
    }

    /// Channel 1 (Brain -> Cells Write) + Full Ecological Lifecycle Step on `self.cells`:
    /// 1. Decay `fullness`, `energy`, `co_bonds`, and `time_bonds`.
    /// 2. Feed matching cells or birth new `EpigeneticCell`s when input fits no existing cell (`unknown`).
    /// 3. Consolidate repeatedly fed STM cells (`layer 0 -> 1`).
    /// 4. Strengthen `co_bonds` and directed `time_bonds` (`prev_active_cell_ids -> current_active_ids`)
    ///    and move bonded cells toward each other in 3D space (`pos`).
    /// 5. Variety-limit Mitosis (`variety_score > variety_split_limit` -> Parent Concept + 2 Daughter Cells).
    /// 6. Elevate spatial neighborhood clusters into `CELL_TISSUE_CONCEPT_CLUSTER` cells (`layer = 2`).
    /// 7. Merge near-duplicate cells (`cos > merge_similarity_limit`) and reap starved cells with
    ///    full Neighbor Handoff (`cells_died_handoff`).
    pub fn step_living_cells(
        &mut self,
        meaning_vec: Option<&[f32]>,
        word_vec: Option<&[f32]>,
        drive_vec: Option<&[f32]>,
        motor_act: &[f32; 4],
        vocal_act: &[f32; 8],
        valence: f32,
        broadcast: ModulationBroadcast,
    ) -> CellReport {
        let start = self.basal_columns.min(self.cells.len());
        for c in self.cells.iter_mut().skip(start) {
            c.fullness = (c.fullness * 0.62 - 0.04).clamp(0.0, 1.0);
            let eff_decay = c.decay_rate * (1.0 - 0.45 * broadcast.rest);
            c.energy = (c.energy - eff_decay).clamp(0.0, 2.0);
            c.vesicle_ready = (c.vesicle_ready + 0.07 + 0.10 * broadcast.rest).clamp(0.25, 1.0);
            for (_, w) in &mut c.co_bonds {
                *w *= 0.995;
            }
            c.co_bonds.retain(|(_, w)| *w > 0.015);
            for (_, w) in &mut c.time_bonds {
                *w *= 0.995;
            }
            c.time_bonds.retain(|(_, w)| *w > 0.015);
        }

        let mut current_active_ids: Vec<u64> = Vec::new();
        let mut any_unknown = false;
        let mut any_matched = false;

        if let Some(m_in) = meaning_vec {
            if l2_norm(m_in) > 0.08 {
                let (cid, was_born) = self.feed_or_birth_cell(
                    CELL_TISSUE_RETINA_MEANING,
                    m_in,
                    motor_act,
                    vocal_act,
                    valence,
                );
                current_active_ids.push(cid);
                if was_born {
                    any_unknown = true;
                } else {
                    any_matched = true;
                }
            }
        }

        if let Some(w_in) = word_vec {
            if l2_norm(w_in) > 0.08 {
                let (cid, was_born) = self.feed_or_birth_cell(
                    CELL_TISSUE_COCHLEA_WORD,
                    w_in,
                    motor_act,
                    vocal_act,
                    valence,
                );
                current_active_ids.push(cid);
                if was_born {
                    any_unknown = true;
                } else {
                    any_matched = true;
                }
            }
        }

        if let Some(d_in) = drive_vec {
            if l2_norm(d_in) > 0.08 {
                let (cid, _) = self.feed_or_birth_cell(
                    CELL_TISSUE_VISCERAL_DRIVE,
                    d_in,
                    motor_act,
                    vocal_act,
                    valence,
                );
                current_active_ids.push(cid);
            }
        }

        let bond_gain = (0.28 + 0.35 * (broadcast.reward + broadcast.danger + broadcast.surprise))
            .clamp(0.18, 0.85);
        for i in 0..current_active_ids.len() {
            for j in (i + 1)..current_active_ids.len() {
                let id_a = current_active_ids[i];
                let id_b = current_active_ids[j];
                if let Some(idx_a) = self.find_cell_idx(id_a) {
                    self.cells[idx_a].add_or_strengthen_co_bond(id_b, bond_gain);
                }
                if let Some(idx_b) = self.find_cell_idx(id_b) {
                    self.cells[idx_b].add_or_strengthen_co_bond(id_a, bond_gain);
                }
                self.pull_cells_together(id_a, id_b, 0.08);
            }
        }

        if !self.prev_active_cell_ids.is_empty() && !current_active_ids.is_empty() {
            let prev_ids = self.prev_active_cell_ids.clone();
            for &prev_id in &prev_ids {
                for &curr_id in &current_active_ids {
                    if prev_id != curr_id {
                        if let Some(p_idx) = self.find_cell_idx(prev_id) {
                            self.cells[p_idx].add_or_strengthen_time_bond(curr_id, bond_gain * 0.90);
                        }
                    }
                }
            }
        }
        if !current_active_ids.is_empty() {
            self.prev_active_cell_ids = current_active_ids.clone();
        }

        self.check_and_split_cells();
        self.elevate_neighborhood_clusters();
        self.merge_near_duplicate_cells();
        self.reap_starved_cells_with_handoff();

        let brain_woke_up = any_unknown || broadcast.surprise > 0.45;
        if brain_woke_up {
            self.waking_brain_steps += 1;
        } else if any_matched {
            self.cheap_cell_steps += 1;
        }
        let report = CellReport {
            matched: any_matched,
            unknown: any_unknown,
            conflict: false,
            emotional: valence.abs() > 0.20,
            confidence: if any_matched { 0.88 } else { 0.30 },
            active_count: current_active_ids.len(),
            brain_woke_up,
        };
        self.last_cell_report = report;
        report
    }

    fn feed_or_birth_cell(
        &mut self,
        tissue_kind: u8,
        pat_in: &[f32],
        motor_act: &[f32; 4],
        vocal_act: &[f32; 8],
        valence: f32,
    ) -> (u64, bool) {
        let mut pat = vec![0.0f32; self.state_dim];
        let n_copy = self.state_dim.min(pat_in.len());
        pat[..n_copy].copy_from_slice(&pat_in[..n_copy]);
        normalize_l2_in_place(&mut pat);

        let start = self.basal_columns.min(self.cells.len());
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for idx in start..self.cells.len() {
            let c = &self.cells[idx];
            if c.tissue_kind != tissue_kind {
                continue;
            }
            let sim = c.ionotropic_conductance(&pat);
            // Strict receptor conductance limit so distinct acoustic/visual patterns
            // each birth their own living cell instead of swallowing each other!
            if sim >= c.similarity_limit {
                let hunger = (1.0 - 0.45 * c.fullness).clamp(0.25, 1.0);
                let sc = sim * hunger;
                if sc > best_score {
                    best_score = sc;
                    best_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = best_idx {
            let mut commit_delta = false;
            let mut revert_delta = false;
            let cell_id = {
                let c = &mut self.cells[idx];
                let gene_gain = c.gene_expr.eval_biophysical(&[
                    1.0,
                    (1.0 - 0.45 * c.fullness).clamp(0.25, 1.0),
                    c.energy,
                    c.vesicle_ready,
                    valence.abs(),
                    0.5,
                ]);
                let base_lr = if c.layer == 0 {
                    (0.26 / (1.0 + 0.12 * (c.fed_count as f32))).clamp(0.05, 0.26)
                } else {
                    0.04
                };
                let lr = (base_lr * gene_gain).clamp(0.03, 0.30);
                let mut var_sum = 0.0f32;
                for d in 0..self.state_dim {
                    let diff = pat[d] - c.receptor_w[d];
                    c.variance_w[d] = 0.82 * c.variance_w[d] + 0.18 * (diff * diff);
                    var_sum += c.variance_w[d];
                    c.receptor_w[d] += lr * diff;
                }
                normalize_l2_in_place(&mut c.receptor_w);
                c.variety_score = var_sum.sqrt();
                c.energy = (c.energy + 0.24).clamp(0.10, 1.85);
                c.fullness = (c.fullness + 0.35).clamp(0.0, 1.0);
                c.vesicle_ready = (c.vesicle_ready - 0.10).clamp(0.30, 1.0);
                c.fed_count = c.fed_count.saturating_add(1);
                c.somatic_feeling = 0.80 * c.somatic_feeling + 0.20 * valence;

                // Provisional-Hold First-Traversal Proof vs Instant Escrow Self-Healing:
                if c.provisional_hold == 1 && c.fed_count >= 3 {
                    if valence >= -0.25 && c.energy > 0.15 {
                        c.provisional_hold = 0;
                        c.escrow_gene = c.gene_expr;
                        c.chromatin_locked = (c.chromatin_locked + 0.12).clamp(0.0, 1.0);
                        commit_delta = true;
                    } else {
                        c.gene_expr = c.escrow_gene;
                        c.provisional_hold = 0;
                        revert_delta = true;
                    }
                }

                if valence >= 0.08 {
                    for a in 0..4 {
                        c.motor_action_w[a] = 0.25 * c.motor_action_w[a] + 0.75 * motor_act[a];
                    }
                }
                if tissue_kind == CELL_TISSUE_COCHLEA_WORD {
                    for v in 0..8 {
                        c.vocal_motor_w[v] = 0.25 * c.vocal_motor_w[v] + 0.75 * vocal_act[v];
                    }
                }

                if c.layer == 0 && c.fed_count >= 5 && c.energy >= 0.90 {
                    c.layer = 1;
                    c.decay_rate = 0.0020;
                    c.similarity_limit = (c.similarity_limit + 0.02).clamp(0.68, 0.90);
                }
                c.id
            };
            if commit_delta {
                self.provisional_commits += 1;
                self.sealed_generation += 1;
            }
            if revert_delta {
                self.provisional_reverts += 1;
            }
            return (cell_id, false);
        }

        // Birth (Neurogenesis) of a new living `EpigeneticCell`
        let new_id = self.next_cell_id;
        self.next_cell_id += 1;
        self.cells_born += 1;
        let p0 = self.next_ion_fluctuation(new_id as f32) * 0.8;
        let p1 = self.next_ion_fluctuation((new_id * 3) as f32) * 0.8;
        let p2 = (tissue_kind as f32) * 0.35;
        let sim_lim = match tissue_kind {
            CELL_TISSUE_COCHLEA_WORD => 0.84,
            CELL_TISSUE_RETINA_MEANING => 0.85,
            CELL_TISSUE_VISCERAL_DRIVE => 0.68,
            _ => 0.78,
        };
        let new_cell = EpigeneticCell {
            id: new_id,
            tissue_kind,
            layer: 0,
            energy: 0.82,
            fullness: 0.30,
            variance_w: vec![0.0; self.state_dim],
            variety_score: 0.0,
            similarity_limit: sim_lim,
            decay_rate: 0.009,
            fed_count: 1,
            pos: [p0, p1, p2],
            co_bonds: Vec::new(),
            time_bonds: Vec::new(),
            parent_id: None,
            child_ids: Vec::new(),
            creb_excitability: 0.6,
            chromatin_locked: 0.2,
            synaptic_tag: 0.5,
            tag_med: 0.3,
            tag_slow: 0.2,
            erk_nuclear: 0.2,
            vesicle_ready: 1.0,
            gene_expr: RegGeneExpr::default(),
            escrow_gene: RegGeneExpr::default(),
            provisional_hold: 0,
            receptor_w: pat.clone(),
            context_w: pat,
            axon_out_w: vec![0.0; self.state_dim],
            world_delta_w: vec![0.0; self.state_dim],
            somatic_feeling: valence,
            elig_action: [*motor_act; 3],
            action_pending: 0.0,
            motor_action_w: *motor_act,
            failed_actions: [0.0; 4],
            vocal_motor_w: *vocal_act,
        };
        self.cells.push(new_cell);
        (new_id, true)
    }

    fn pull_cells_together(&mut self, id_a: u64, id_b: u64, step: f32) {
        let (Some(ia), Some(ib)) = (self.find_cell_idx(id_a), self.find_cell_idx(id_b)) else {
            return;
        };
        let pos_a = self.cells[ia].pos;
        let pos_b = self.cells[ib].pos;
        for k in 0..3 {
            let diff = pos_b[k] - pos_a[k];
            self.cells[ia].pos[k] = (self.cells[ia].pos[k] + step * diff).clamp(-2.0, 2.0);
            self.cells[ib].pos[k] = (self.cells[ib].pos[k] - step * diff).clamp(-2.0, 2.0);
        }
    }

    fn check_and_split_cells(&mut self) {
        if self.cells.len() + 2 > self.max_cells {
            return;
        }
        let start = self.basal_columns.min(self.cells.len());
        let mut to_split: Option<usize> = None;
        for idx in start..self.cells.len() {
            let c = &self.cells[idx];
            if c.child_ids.is_empty()
                && c.fed_count >= 4
                && c.variety_score > self.variety_split_limit
                && c.layer < 2
            {
                to_split = Some(idx);
                break;
            }
        }
        let Some(p_idx) = to_split else {
            return;
        };

        let id_a = self.next_cell_id;
        let id_b = self.next_cell_id + 1;
        self.next_cell_id += 2;
        self.cells_split += 1;

        let parent = self.cells[p_idx].clone();
        let standing_on_stack = self.prev_active_cell_ids.iter().any(|&id| id == parent.id);
        let mut pat_a = parent.receptor_w.clone();
        let mut pat_b = parent.receptor_w.clone();
        for d in 0..self.state_dim {
            let offset = parent.variance_w[d].sqrt().max(0.14) * 0.70;
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            pat_a[d] += sign * offset;
            pat_b[d] -= sign * offset;
        }
        normalize_l2_in_place(&mut pat_a);
        normalize_l2_in_place(&mut pat_b);

        self.cells[p_idx].layer = (parent.layer + 1).min(3);
        self.cells[p_idx].tissue_kind = CELL_TISSUE_CONCEPT_CLUSTER;
        self.cells[p_idx].variety_score *= 0.35;
        for v in &mut self.cells[p_idx].variance_w {
            *v *= 0.25;
        }
        self.cells[p_idx].decay_rate = 0.0018;
        self.cells[p_idx].child_ids.push(id_a);
        self.cells[p_idx].child_ids.push(id_b);
        self.cells[p_idx].add_or_strengthen_co_bond(id_a, 0.95);
        self.cells[p_idx].add_or_strengthen_co_bond(id_b, 0.95);
        if standing_on_stack {
            self.cells[p_idx].escrow_gene = parent.gene_expr;
            self.cells[p_idx].provisional_hold = 1;
        }

        let gene_a = parent.gene_expr.patch_subform(id_a.wrapping_mul(0x9E37_79B9));
        let gene_b = parent.gene_expr.patch_subform(id_b.wrapping_mul(0xBF58_476D));

        let mut child_a = EpigeneticCell {
            id: id_a,
            tissue_kind: parent.tissue_kind,
            layer: 0,
            energy: (parent.energy * 0.65).max(0.55),
            fullness: 0.15,
            variance_w: vec![0.0; self.state_dim],
            variety_score: 0.0,
            similarity_limit: (parent.similarity_limit + 0.05).clamp(0.72, 0.88),
            decay_rate: 0.006,
            fed_count: 2,
            pos: [parent.pos[0] - 0.08, parent.pos[1] + 0.06, parent.pos[2]],
            co_bonds: parent.co_bonds.clone(),
            time_bonds: parent.time_bonds.clone(),
            parent_id: Some(parent.id),
            child_ids: Vec::new(),
            creb_excitability: 0.5,
            chromatin_locked: parent.chromatin_locked,
            synaptic_tag: 0.3,
            tag_med: 0.2,
            tag_slow: 0.2,
            erk_nuclear: 0.2,
            vesicle_ready: 1.0,
            gene_expr: gene_a,
            escrow_gene: parent.gene_expr,
            provisional_hold: 1,
            receptor_w: pat_a.clone(),
            context_w: pat_a,
            axon_out_w: parent.axon_out_w.clone(),
            world_delta_w: parent.world_delta_w.clone(),
            somatic_feeling: parent.somatic_feeling,
            elig_action: parent.elig_action,
            action_pending: 0.0,
            motor_action_w: parent.motor_action_w,
            failed_actions: parent.failed_actions,
            vocal_motor_w: parent.vocal_motor_w,
        };
        child_a.add_or_strengthen_co_bond(parent.id, 0.95);
        child_a.add_or_strengthen_co_bond(id_b, 0.50);

        let mut child_b = child_a.clone();
        child_b.id = id_b;
        child_b.gene_expr = gene_b;
        child_b.escrow_gene = parent.gene_expr;
        child_b.provisional_hold = 1;
        child_b.receptor_w = pat_b.clone();
        child_b.context_w = pat_b;
        child_b.pos = [parent.pos[0] + 0.08, parent.pos[1] - 0.06, parent.pos[2]];
        child_b.co_bonds = parent.co_bonds.clone();
        child_b.add_or_strengthen_co_bond(parent.id, 0.95);
        child_b.add_or_strengthen_co_bond(id_a, 0.50);

        self.cells.push(child_a);
        self.cells.push(child_b);
    }

    fn elevate_neighborhood_clusters(&mut self) {
        if self.cells.len() >= self.max_cells {
            return;
        }
        let start = self.basal_columns.min(self.cells.len());
        let n = self.cells.len();
        for i in start..n {
            if self.cells[i].layer >= 2 || self.cells[i].parent_id.is_some() {
                continue;
            }
            let mut cluster_members = vec![i];
            for &(nid, w_co) in &self.cells[i].co_bonds {
                if w_co < 0.55 {
                    continue;
                }
                if let Some(j) = self.find_cell_idx(nid) {
                    if j >= start && self.cells[j].layer < 2 && self.cells[j].parent_id.is_none() {
                        let d2 = (self.cells[i].pos[0] - self.cells[j].pos[0]).powi(2)
                            + (self.cells[i].pos[1] - self.cells[j].pos[1]).powi(2)
                            + (self.cells[i].pos[2] - self.cells[j].pos[2]).powi(2);
                        if d2 < 0.30 {
                            cluster_members.push(j);
                        }
                    }
                }
            }
            if cluster_members.len() >= 3 {
                let concept_id = self.next_cell_id;
                self.next_cell_id += 1;
                self.cells_born += 1;

                let mut centroid_pat = vec![0.0f32; self.state_dim];
                let mut centroid_pos = [0.0f32; 3];
                let mut child_ids = Vec::new();
                let mut co_bonds = Vec::new();
                for &m_idx in &cluster_members {
                    for d in 0..self.state_dim {
                        centroid_pat[d] += self.cells[m_idx].receptor_w[d];
                    }
                    for k in 0..3 {
                        centroid_pos[k] += self.cells[m_idx].pos[k];
                    }
                    let mid = self.cells[m_idx].id;
                    child_ids.push(mid);
                    co_bonds.push((mid, 0.90));
                }
                normalize_l2_in_place(&mut centroid_pat);
                let inv = 1.0 / (cluster_members.len() as f32);
                centroid_pos[0] *= inv;
                centroid_pos[1] *= inv;
                centroid_pos[2] = centroid_pos[2] * inv + 0.25;

                for &m_idx in &cluster_members {
                    self.cells[m_idx].parent_id = Some(concept_id);
                    self.cells[m_idx].add_or_strengthen_co_bond(concept_id, 0.90);
                }

                let concept_cell = EpigeneticCell {
                    id: concept_id,
                    tissue_kind: CELL_TISSUE_CONCEPT_CLUSTER,
                    layer: 2,
                    energy: 1.05,
                    fullness: 0.0,
                    variance_w: vec![0.0; self.state_dim],
                    variety_score: 0.0,
                    similarity_limit: 0.62,
                    decay_rate: 0.0015,
                    fed_count: 3,
                    pos: centroid_pos,
                    co_bonds,
                    time_bonds: Vec::new(),
                    parent_id: None,
                    child_ids,
                    creb_excitability: 0.5,
                    chromatin_locked: 0.6,
                    synaptic_tag: 0.3,
                    tag_med: 0.3,
                    tag_slow: 0.3,
                    erk_nuclear: 0.3,
                    vesicle_ready: 1.0,
                    gene_expr: RegGeneExpr::default(),
                    escrow_gene: RegGeneExpr::default(),
                    provisional_hold: 0,
                    receptor_w: centroid_pat.clone(),
                    context_w: centroid_pat,
                    axon_out_w: vec![0.0; self.state_dim],
                    world_delta_w: vec![0.0; self.state_dim],
                    somatic_feeling: 0.0,
                    elig_action: [[0.25; 4]; 3],
                    action_pending: 0.0,
                    motor_action_w: [0.25; 4],
                    failed_actions: [0.0; 4],
                    vocal_motor_w: [0.5; 8],
                };
                self.cells.push(concept_cell);
                break;
            }
        }
    }

    fn merge_near_duplicate_cells(&mut self) {
        let start = self.basal_columns.min(self.cells.len());
        let n = self.cells.len();
        let mut merge_pair: Option<(usize, usize)> = None;
        'outer: for i in start..n {
            for j in (i + 1)..n {
                let not_siblings = self.cells[i].parent_id.is_none()
                    || self.cells[i].parent_id != self.cells[j].parent_id;
                if not_siblings
                    && self.cells[i].tissue_kind == self.cells[j].tissue_kind
                    && self.cells[i].layer == self.cells[j].layer
                    && dot(&self.cells[i].receptor_w, &self.cells[j].receptor_w) > self.merge_similarity_limit
                {
                    merge_pair = Some((i, j));
                    break 'outer;
                }
            }
        }
        if let Some((keep_idx, drop_idx)) = merge_pair {
            let keep_id = self.cells[keep_idx].id;
            let drop_cell = self.cells.remove(drop_idx);
            self.cells_merged += 1;
            if let Some(k_idx) = self.find_cell_idx(keep_id) {
                for d in 0..self.state_dim {
                    self.cells[k_idx].receptor_w[d] =
                        0.65 * self.cells[k_idx].receptor_w[d] + 0.35 * drop_cell.receptor_w[d];
                }
                normalize_l2_in_place(&mut self.cells[k_idx].receptor_w);
                self.cells[k_idx].energy =
                    (self.cells[k_idx].energy + 0.50 * drop_cell.energy).clamp(0.1, 1.9);
                self.cells[k_idx].fed_count =
                    self.cells[k_idx].fed_count.saturating_add(drop_cell.fed_count);
                for (bid, bw) in drop_cell.co_bonds {
                    self.cells[k_idx].add_or_strengthen_co_bond(bid, bw * 0.8);
                }
                for (tid, tw) in drop_cell.time_bonds {
                    self.cells[k_idx].add_or_strengthen_time_bond(tid, tw * 0.8);
                }
            }
            self.remap_removed_cell_bonds(drop_cell.id, Some(keep_id));
        }
    }

    fn reap_starved_cells_with_handoff(&mut self) {
        let start = if self.floor_locked {
            self.basal_columns.min(self.cells.len())
        } else {
            0
        };
        if self.cells.len() <= start + 6 {
            return;
        }
        let mut starved_idx: Option<usize> = None;
        for (idx, c) in self.cells.iter().enumerate().skip(start) {
            if c.energy <= 0.08 {
                starved_idx = Some(idx);
                break;
            }
        }
        let total_e: f32 = self.cells.iter().skip(start).map(|c| c.energy).sum();
        if starved_idx.is_none() && (total_e > self.total_energy_budget || self.cells.len() > self.max_cells) {
            let mut min_e = f32::INFINITY;
            for (idx, c) in self.cells.iter().enumerate().skip(start) {
                if c.layer == 0 && c.energy < min_e {
                    min_e = c.energy;
                    starved_idx = Some(idx);
                }
            }
        }
        let Some(d_idx) = starved_idx else {
            return;
        };
        let dying = self.cells.remove(d_idx);
        self.cells_died_handoff += 1;
        if dying.provisional_hold == 1 {
            self.provisional_reverts += 1;
        }

        let mut best_neighbor_id: Option<u64> = dying
            .co_bonds
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|x| x.0);
        if best_neighbor_id.is_none() {
            let mut best_sim = f32::NEG_INFINITY;
            for c in self.cells.iter().skip(start) {
                if c.tissue_kind == dying.tissue_kind {
                    let s = dot(&c.receptor_w, &dying.receptor_w);
                    if s > best_sim {
                        best_sim = s;
                        best_neighbor_id = Some(c.id);
                    }
                }
            }
        }

        if let Some(n_id) = best_neighbor_id {
            if let Some(n_idx) = self.find_cell_idx(n_id) {
                for d in 0..self.state_dim {
                    self.cells[n_idx].receptor_w[d] =
                        0.85 * self.cells[n_idx].receptor_w[d] + 0.15 * dying.receptor_w[d];
                }
                normalize_l2_in_place(&mut self.cells[n_idx].receptor_w);
                for (bid, bw) in &dying.co_bonds {
                    if *bid != n_id {
                        self.cells[n_idx].add_or_strengthen_co_bond(*bid, *bw * 0.70);
                    }
                }
                for (tid, tw) in &dying.time_bonds {
                    if *tid != n_id {
                        self.cells[n_idx].add_or_strengthen_time_bond(*tid, *tw * 0.70);
                    }
                }
            }
        }
        self.remap_removed_cell_bonds(dying.id, best_neighbor_id);
    }

    fn remap_removed_cell_bonds(&mut self, old_id: u64, replacement_id: Option<u64>) {
        let start = self.basal_columns.min(self.cells.len());
        for c in self.cells.iter_mut().skip(start) {
            if let Some(rep) = replacement_id {
                for (id, _) in &mut c.co_bonds {
                    if *id == old_id {
                        *id = rep;
                    }
                }
                for (id, _) in &mut c.time_bonds {
                    if *id == old_id {
                        *id = rep;
                    }
                }
            }
            c.co_bonds.retain(|(id, _)| *id != old_id && *id != c.id);
            c.time_bonds.retain(|(id, _)| *id != old_id && *id != c.id);
        }
        self.prev_active_cell_ids.retain(|&id| id != old_id);
    }

    /// Offline Dreaming across the Living `EpigeneticCell` population:
    /// When there is no input, hungry cells poke their bonded neighbors, strengthen useful bonds,
    /// prune weak ones, and form new insight bridges between 2-hop distant clusters.
    pub fn dream_living_cells(&mut self, rounds: usize) -> usize {
        let start = self.basal_columns.min(self.cells.len());
        let mut new_bridges = 0usize;
        for r in 0..rounds {
            if self.cells.len() <= start + 2 {
                break;
            }
            let mut best_idx = start;
            let mut best_drive = f32::NEG_INFINITY;
            let n_total = self.cells.len();
            for idx in start..n_total {
                if self.cells[idx].co_bonds.is_empty() {
                    continue;
                }
                let fullness = self.cells[idx].fullness;
                let energy = self.cells[idx].energy;
                let n_bonds = self.cells[idx].co_bonds.len() as f32;
                let noise = 0.20 * self.next_ion_fluctuation((idx + r * 13) as f32);
                let drive = (1.0 - fullness) * energy + 0.15 * n_bonds + noise;
                if drive > best_drive {
                    best_drive = drive;
                    best_idx = idx;
                }
            }

            let poker_id = self.cells[best_idx].id;
            let neighbors = self.cells[best_idx].co_bonds.clone();
            for &(n1_id, w1) in neighbors.iter().take(4) {
                self.cells[best_idx].add_or_strengthen_co_bond(n1_id, 0.04);
                let Some(n1_idx) = self.find_cell_idx(n1_id) else {
                    continue;
                };
                let n2_list = self.cells[n1_idx].co_bonds.clone();
                for &(n2_id, w2) in n2_list.iter().take(3) {
                    if n2_id == poker_id {
                        continue;
                    }
                    let already_bonded = self.cells[best_idx].co_bonds.iter().any(|(id, _)| *id == n2_id);
                    if !already_bonded && w1 * w2 > 0.12 {
                        let bridge_w = (0.18 * (w1 * w2).sqrt()).clamp(0.05, 0.45);
                        self.cells[best_idx].add_or_strengthen_co_bond(n2_id, bridge_w);
                        if let Some(n2_idx) = self.find_cell_idx(n2_id) {
                            self.cells[n2_idx].add_or_strengthen_co_bond(poker_id, bridge_w);
                        }
                        self.dream_bridges_formed += 1;
                        new_bridges += 1;
                    }
                }
            }

            for c in self.cells.iter_mut().skip(start) {
                c.co_bonds.retain(|(_, w)| *w >= 0.03);
                c.time_bonds.retain(|(_, w)| *w >= 0.03);
            }
        }
        new_bridges
    }

    pub fn living_type_counts(&self) -> (usize, usize, usize, usize, usize) {
        let start = self.basal_columns.min(self.cells.len());
        let mut stm = 0usize;
        let mut ltm = 0usize;
        let mut concepts = 0usize;
        let mut words = 0usize;
        let mut meanings = 0usize;
        for c in self.cells.iter().skip(start) {
            if c.layer == 0 {
                stm += 1;
            } else if c.layer == 1 {
                ltm += 1;
            } else {
                concepts += 1;
            }
            if c.tissue_kind == CELL_TISSUE_COCHLEA_WORD {
                words += 1;
            } else if c.tissue_kind == CELL_TISSUE_RETINA_MEANING {
                meanings += 1;
            }
        }
        (stm, ltm, concepts, words, meanings)
    }

    pub fn total_living_bonds(&self) -> (usize, usize) {
        let start = self.basal_columns.min(self.cells.len());
        let co: usize = self.cells.iter().skip(start).map(|c| c.co_bonds.len()).sum();
        let tm: usize = self.cells.iter().skip(start).map(|c| c.time_bonds.len()).sum();
        (co, tm)
    }

    pub fn provisional_stats(&self) -> (usize, u64, u64, u64) {
        let start = self.basal_columns.min(self.cells.len());
        let held = self
            .cells
            .iter()
            .skip(start)
            .filter(|c| c.provisional_hold == 1)
            .count();
        (
            held,
            self.provisional_commits,
            self.provisional_reverts,
            self.sealed_generation,
        )
    }
}
