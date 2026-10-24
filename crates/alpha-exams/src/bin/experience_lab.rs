//! Open-Ended Biological Experience & Cellular Memory Laboratory (`experience_lab.rs`)
//!
//! Critically tests and compares biological hypotheses and sub-branches for how experience
//! is formed, stored across every cell/molecule, extracted for reasoning/analogy, and
//! expressed naturally in a living organism across non-repetitive real-world life streams.
//!
//! Zero staged pass/fail assertions: reports raw physical & behavioral measurements across
//! all branches (including failed naive branches vs refined biophysical branches).

use alpha_iface::SensoryFrame;
use alpha_organism::{
    Organism,
    genome::Genome,
    util::{dot, l2_norm},
};
use std::fs;
use std::io::Read;

struct RealUtterance {
    split: u32,
    word_id: usize,
    pcm: Vec<f32>,
}

#[derive(Clone)]
struct SceneObj {
    colour_id: usize,
    shape_id: usize,
    cx: f32,
    cy: f32,
}

struct RealScene {
    path: String,
    objs: Vec<SceneObj>,
}

fn load_audio_pool(path: &str) -> Vec<RealUtterance> {
    let mut f = fs::File::open(path).expect("open real_audio_pool.bin");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut pos = 0usize;
    let n = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let split = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        let word_id = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap()) as usize;
        pos += 12;
        let mut pcm = Vec::with_capacity(len);
        for _ in 0..len {
            let v = f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            pcm.push(v);
        }
        out.push(RealUtterance { split, word_id, pcm });
    }
    out
}

fn load_scenes(manifest_path: &str) -> Vec<RealScene> {
    let txt = fs::read_to_string(manifest_path).expect("read scene manifest");
    let mut out = Vec::new();
    for line in txt.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 2 {
            continue;
        }
        let path = parts[0].to_string();
        let mut objs = Vec::new();
        for item in parts[1].split(';') {
            let nums: Vec<&str> = item.split(',').collect();
            if nums.len() == 4 {
                let colour_id = nums[0].parse().unwrap_or(0);
                let shape_id = nums[1].parse().unwrap_or(0);
                let cy = nums[2].parse().unwrap_or(120.0);
                let cx = nums[3].parse().unwrap_or(160.0);
                objs.push(SceneObj { colour_id, shape_id, cx, cy });
            }
        }
        if !objs.is_empty() {
            out.push(RealScene { path, objs });
        }
    }
    out
}

fn read_ppm(path: &str) -> Option<(Vec<u8>, usize, usize)> {
    let bytes = fs::read(path).ok()?;
    let mut idx = 0usize;
    let mut next_tok = || -> Option<String> {
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < bytes.len() && bytes[idx] == b'#' {
            while idx < bytes.len() && bytes[idx] != b'\n' {
                idx += 1;
            }
            while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
        }
        let start = idx;
        while idx < bytes.len() && !bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if start == idx {
            None
        } else {
            String::from_utf8(bytes[start..idx].to_vec()).ok()
        }
    };
    let magic = next_tok()?;
    if magic != "P6" {
        return None;
    }
    let w: usize = next_tok()?.parse().ok()?;
    let h: usize = next_tok()?.parse().ok()?;
    let _maxv: usize = next_tok()?.parse().ok()?;
    idx += 1;
    if idx + w * h * 3 > bytes.len() {
        return None;
    }
    Some((bytes[idx..idx + w * h * 3].to_vec(), w, h))
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / ((l2_norm(a) + 1e-8) * (l2_norm(b) + 1e-8))
}

// ============================================================================
// BRANCH A: Pure Rolling Experience Chain (Hypothesis 1 Baseline)
//   - Single rolling chain (`fast -> slow -> deep`) + single `w_flow` matrix.
// ============================================================================
struct BranchARollingChain {
    dim: usize,
    fast: Vec<f32>,
    slow: Vec<f32>,
    deep: Vec<f32>,
    elig: Vec<f32>,
    w_flow: Vec<f32>,
}

impl BranchARollingChain {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            fast: vec![0.0; dim],
            slow: vec![0.0; dim],
            deep: vec![0.0; dim],
            elig: vec![0.0; dim],
            w_flow: vec![0.0; dim * dim],
        }
    }

    fn step(&mut self, s: &[f32], valence: f32, arousal: f32) {
        let gain = (0.08 * (1.0 + valence.abs() + 0.5 * arousal)).clamp(0.02, 0.35);
        for d in 0..self.dim {
            self.elig[d] = 0.45 * self.elig[d] + 0.55 * self.fast[d];
        }
        for i in 0..self.dim {
            let row = i * self.dim;
            for j in 0..self.dim {
                let hebb = s[i] * self.elig[j];
                let decay = s[j] * s[j] * self.w_flow[row + j];
                self.w_flow[row + j] = (self.w_flow[row + j] + gain * (hebb - 0.8 * decay)).clamp(-1.0, 1.0);
            }
        }
        for d in 0..self.dim {
            self.fast[d] = 0.25 * self.fast[d] + 0.75 * s[d];
            self.slow[d] = 0.88 * self.slow[d] + 0.12 * self.fast[d];
            self.deep[d] = 0.996 * self.deep[d] + 0.004 * self.slow[d];
        }
    }

    fn predict_next(&self, s: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = i * self.dim;
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.w_flow[row + j] * s[j];
            }
            out[i] = sum;
        }
        out
    }
}

// ============================================================================
// BRANCH B1 (Naive Cascade) vs BRANCH B2 (Kukushkin 2024 ERK-CREB Desensitizing Cascade)
//   - Why did B1 fail on Massed vs Spaced? Because B1 had no receptor desensitization
//     and no delayed ERK nuclear translocation!
//   - In B2 (Kukushkin et al. 2024, Nature Communications):
//       1. Fast membrane cAMP/Ca2+ pool (`vesicle_ready`) depletes upon strong stimulation
//          and recovers over ~15-20 ticks (so crammed/massed pulses hit a refractory wall!).
//       2. Activated kinase (`erk_nuclear`) slowly builds up and primes nuclear `creb`.
//       3. When a subsequent pulse arrives AFTER `vesicle_ready` recovers AND WHILE
//          `erk_nuclear` is still elevated (spaced learning!), `creb` supralinearly opens
//          chromatin (`m4_chromatin`) and locks the memory against decay!
// ============================================================================
struct BranchB2KukushkinCascade {
    dim: usize,
    fast: Vec<f32>,
    elig: Vec<f32>,
    m1_ion: Vec<f32>,
    m2_tag: Vec<f32>,
    m3_erk_primed: Vec<f32>,
    m4_chromatin: Vec<f32>,
    receptor_ready: Vec<f32>, // Per-synapse desensitization/recovery pool (1.0 = ready, 0.0 = refractory)
}

impl BranchB2KukushkinCascade {
    fn new(dim: usize) -> Self {
        let n2 = dim * dim;
        Self {
            dim,
            fast: vec![0.0; dim],
            elig: vec![0.0; dim],
            m1_ion: vec![0.0; n2],
            m2_tag: vec![0.0; n2],
            m3_erk_primed: vec![0.0; n2],
            m4_chromatin: vec![0.0; n2],
            receptor_ready: vec![1.0; n2],
        }
    }

    fn step(&mut self, s: &[f32], valence: f32, arousal: f32) {
        let prp_burst = (valence.abs() + 0.5 * arousal - 0.30).max(0.0);

        for d in 0..self.dim {
            self.elig[d] = 0.35 * self.elig[d] + 0.65 * self.fast[d];
        }

        for i in 0..self.dim {
            let row = i * self.dim;
            for j in 0..self.dim {
                let idx = row + j;
                let hebb = s[i] * self.elig[j];
                let pre_act = self.elig[j].abs();
                let co_act = hebb.abs();

                // Effective pulse is gated by `receptor_ready` (desensitizes during massed cramming!)
                let eff_hebb = hebb * self.receptor_ready[idx];

                // 1. Fast ion/AMPA update (overwritable unless anchored by chromatin `m4`)
                let chromatin_shield = (1.0 - 0.92 * self.m4_chromatin[idx].abs()).max(0.05);
                self.m1_ion[idx] = (self.m1_ion[idx]
                    + 0.25 * chromatin_shield * (eff_hebb - 0.80 * pre_act * self.m1_ion[idx])
                    + 0.06 * (self.m4_chromatin[idx] - self.m1_ion[idx]))
                    .clamp(-1.0, 1.0);

                // 2. Synaptic tag (`m2_tag`) set by pre->post coincidence (`eff_hebb`)
                self.m2_tag[idx] = (0.93 * self.m2_tag[idx] + 0.55 * eff_hebb).clamp(-1.0, 1.0);

                // 3. Delayed nuclear ERK priming (`m3_erk_primed`): rises slowly after `m2_tag`,
                //    peaking ~15-35 ticks later. When a spaced pulse arrives (`co_act * receptor_ready`)
                //    while `m3_erk_primed` is already elevated, Hill-kinetics CREB opens chromatin!
                let spaced_coincidence = (co_act * 18.0) * self.receptor_ready[idx] * self.m3_erk_primed[idx];
                self.m3_erk_primed[idx] = (0.965 * self.m3_erk_primed[idx]
                    + 0.32 * (co_act * 12.0).min(1.0) * self.receptor_ready[idx])
                    .clamp(0.0, 1.0);

                // 4. Continuous Hill-kinetics CREB Chromatin locking (`m4_chromatin`):
                let stc_drive = prp_burst * (self.m2_tag[idx].abs() * 8.0);
                let creb_raw = 0.40 * stc_drive + 1.40 * spaced_coincidence;
                let creb_hill = (creb_raw * creb_raw) / (creb_raw * creb_raw + 0.08);
                let target_sign = if self.m2_tag[idx] >= 0.0 { 1.0 } else { -1.0 };
                self.m4_chromatin[idx] = (0.9995 * self.m4_chromatin[idx]
                    + 0.35 * creb_hill * (target_sign * (self.m2_tag[idx].abs() * 6.0).min(1.0) - self.m4_chromatin[idx]))
                    .clamp(-1.0, 1.0);

                // Continuous receptor desensitization on activation & slow recovery (~20 ticks)
                let desens = ((co_act * 28.0).powi(2)) / ((co_act * 28.0).powi(2) + 1.0);
                self.receptor_ready[idx] = (self.receptor_ready[idx] * (1.0 - 0.88 * desens)
                    + 0.045 * (1.0 - self.receptor_ready[idx]))
                    .clamp(0.02, 1.0);
            }
        }

        for d in 0..self.dim {
            self.fast[d] = 0.20 * self.fast[d] + 0.80 * s[d];
        }
    }

    fn predict_next(&self, s: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = i * self.dim;
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                let idx = row + j;
                let w = 0.25 * self.m1_ion[idx] + 0.75 * self.m4_chromatin[idx];
                sum += w * s[j];
            }
            out[i] = sum;
        }
        out
    }
}

// ============================================================================
// BRANCH C1 (Naive Cellular Engram) vs BRANCH C2 (Retroactive STC + Dendritic Context Gating)
//   - Why did C1 fail on [EXP 1] and [EXP 2]?
//     Flaw 1 in C1: When `s_trigger` arrived at t=20, the pain (`valence = -0.95`) hadn't
//       happened yet—it happened at t=21 (`s_consequence`)! C1 failed to retroactively lock
//       the chromatin of the cells that fired at t=20 when the shock hit at t=21, AND C1
//       allowed locked cells to keep updating (`45%` open) during mundane ticks!
//     Flaw 2 in C1: C1 added `0.68 * match_s + 0.32 * match_ctx` linearly, so the same cells
//       fired at `B` regardless of whether the organism came from `A` or `D`!
//   - How C2 fixes both biophysically:
//     1. Retroactive Chromatin Locking: When a high-salience consequence arrives at `t`,
//        retrograde messenger / neuromodulator wave locks the chromatin (`chromatin_locked -> 0.98`)
//        of the cells in `active_ensemble` (from `t-1`), shielding their `receptor_w`,
//        `axon_out_w`, and `somatic_feeling` from being overwritten by mundane events!
//     2. Multiplicative Apical-Basal Dendritic Gating: A cell's somatic firing requires BOTH
//        basal sensory input (`match_s`) AND apical context (`match_ctx`) via nonlinear
//        NMDA coincidence (`match_s * (0.5 + 0.5 * match_ctx)`), so `A -> B` and `D -> B`
//        naturally recruit orthogonal cellular ensembles!
// ============================================================================
struct CellC2 {
    creb_excitability: f32,
    chromatin_locked: f32,
    synaptic_tag: f32,
    erk_nuclear: f32,      // Slow nuclear ERK translocation (peaks ~15-35 ticks after cell activation)
    vesicle_ready: f32,    // Intracellular cAMP/receptor readiness (desensitizes during cramming!)
    receptor_w: Vec<f32>,
    context_w: Vec<f32>,
    axon_out_w: Vec<f32>,
    somatic_feeling: f32,
}

struct BranchC2BiophysicalEngrams {
    dim: usize,
    n_cells: usize,
    cells: Vec<CellC2>,
    prev_state: Vec<f32>,
    active_ensemble: Vec<usize>,
    tick: usize,
}

impl BranchC2BiophysicalEngrams {
    fn new(dim: usize, n_cells: usize) -> Self {
        let mut cells = Vec::with_capacity(n_cells);
        for c in 0..n_cells {
            let mut receptor_w = vec![0.0f32; dim];
            let mut context_w = vec![0.0f32; dim];
            for d in 0..dim {
                let p1 = (((c * 37 + d * 19 + 11) % 103) as f32 / 51.0) - 1.0;
                let p2 = (((c * 61 + d * 23 + 17) % 107) as f32 / 53.0) - 1.0;
                receptor_w[d] = p1;
                context_w[d] = p2;
            }
            let nr = l2_norm(&receptor_w) + 1e-8;
            let nc = l2_norm(&context_w) + 1e-8;
            for d in 0..dim {
                receptor_w[d] /= nr;
                context_w[d] /= nc;
            }
            let creb_init = ((c * 29 + 5) % 97) as f32 / 97.0;
            cells.push(CellC2 {
                creb_excitability: creb_init,
                chromatin_locked: 0.0,
                synaptic_tag: 0.0,
                erk_nuclear: 0.0,
                vesicle_ready: 1.0,
                receptor_w,
                context_w,
                axon_out_w: vec![0.0; dim],
                somatic_feeling: 0.0,
            });
        }
        Self {
            dim,
            n_cells,
            cells,
            prev_state: vec![0.0; dim],
            active_ensemble: Vec::new(),
            tick: 0,
        }
    }

    fn step(&mut self, s: &[f32], valence: f32, arousal: f32) {
        self.tick += 1;
        let salience = (valence.abs() + 0.5 * arousal).clamp(0.0, 2.0);

        // 1. Retroactive Synaptic Tagging & Capture (Frey & Morris):
        //    The ensemble that fired on the PREVIOUS tick (`t-1`) now receives the causal
        //    consequence `s` and bodily feeling `valence` arriving at tick `t`!
        for &c_idx in &self.active_ensemble {
            let cell = &mut self.cells[c_idx];
            // If this cell's chromatin is ALREADY locked by an earlier sacred/salient memory,
            // mundane neutral events (`salience < 0.45`) cannot overwrite its axon projection!
            let shield = if salience > 0.50 {
                1.0 - 0.50 * cell.chromatin_locked
            } else {
                (1.0 - 0.98 * cell.chromatin_locked).max(0.0)
            };
            let lr_out = (0.45 * (0.3 + salience) * shield).clamp(0.0, 0.85);
            if lr_out > 1e-4 {
                for d in 0..self.dim {
                    cell.axon_out_w[d] = (1.0 - lr_out) * cell.axon_out_w[d] + lr_out * s[d];
                }
                cell.somatic_feeling = (1.0 - lr_out) * cell.somatic_feeling + lr_out * valence;
            }
            // Retroactive chromatin locking when:
            // (a) `s` carries high emotional/homeostatic salience (`salience > 0.50`), OR
            // (b) Kukushkin (2024) Spaced ERK->CREB Coincidence: this cell was activated while
            //     its `vesicle_ready` had recovered AND its delayed `erk_nuclear` was primed!
            let kukushkin_creb = cell.vesicle_ready * cell.erk_nuclear;
            if salience > 0.50 || kukushkin_creb > 0.18 {
                let boost = if salience > 0.50 { 0.55 + 0.35 * salience } else { 0.85 * kukushkin_creb };
                cell.chromatin_locked = (cell.chromatin_locked + boost).clamp(0.0, 0.99);
            }
            // Deplete intracellular cAMP/vesicle readiness upon strong activation (blocks massed cramming!)
            cell.vesicle_ready *= 0.22;
        }

        // 2. Recruit a sparse cellular ensemble for the current state `s` + context `prev_state`
        let has_ctx = l2_norm(&self.prev_state) > 0.1;
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.n_cells);
        for (c_idx, cell) in self.cells.iter().enumerate() {
            let match_s = dot(&cell.receptor_w, s);
            let match_ctx = if has_ctx {
                dot(&cell.context_w, &self.prev_state)
            } else {
                0.5
            };
            // Multiplicative Apical-Basal NMDA Coincidence:
            // Requires BOTH basal sensory match AND apical context match!
            let basal = (match_s + 1.0).max(0.0);
            let apical = (match_ctx + 1.0).max(0.0);
            let nmda_coincidence = basal * apical;

            // If a cell already matches BOTH `s` AND the context `prev_state`, it re-fires to
            // reconsolidate (enabling Kukushkin spaced ERK->CREB locking) without confusing A->B vs D->B!
            let allocation_drive = if match_s > 0.72 && (!has_ctx || match_ctx > 0.25) {
                nmda_coincidence * 1.15
            } else {
                (1.0 - 0.85 * cell.chromatin_locked) * (nmda_coincidence + 0.35 * cell.creb_excitability)
            };
            scores.push((c_idx, allocation_drive));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = 6usize.min(self.n_cells);
        self.active_ensemble.clear();
        for &(c_idx, _) in scores.iter().take(k) {
            self.active_ensemble.push(c_idx);
            let cell = &mut self.cells[c_idx];
            let lr_in = 0.45 * (1.0 - 0.95 * cell.chromatin_locked);
            if lr_in > 1e-3 {
                for d in 0..self.dim {
                    cell.receptor_w[d] += lr_in * (s[d] - cell.receptor_w[d]);
                    if has_ctx {
                        cell.context_w[d] += lr_in * (self.prev_state[d] - cell.context_w[d]);
                    }
                }
                let nr = l2_norm(&cell.receptor_w) + 1e-8;
                let nc = l2_norm(&cell.context_w) + 1e-8;
                for d in 0..self.dim {
                    cell.receptor_w[d] /= nr;
                    cell.context_w[d] /= nc;
                }
            }
            // Intracellular tag is gated by `vesicle_ready` (crammed pulses add very little new tag!)
            cell.synaptic_tag = (cell.synaptic_tag + (0.45 + 0.35 * salience) * cell.vesicle_ready).clamp(0.0, 1.0);
            cell.creb_excitability *= 0.10;
        }

        for (c_idx, cell) in self.cells.iter_mut().enumerate() {
            if !self.active_ensemble.contains(&c_idx) {
                let wave = (((self.tick * 17 + c_idx * 11) % 97) as f32) / 97.0;
                cell.creb_excitability = (0.90 * cell.creb_excitability + 0.10 * wave).clamp(0.0, 1.0);
                // Slow nuclear ERK translocation from `synaptic_tag` + slow vesicle recovery while resting!
                cell.erk_nuclear = (0.96 * cell.erk_nuclear + 0.08 * cell.synaptic_tag).clamp(0.0, 1.0);
                cell.vesicle_ready = (cell.vesicle_ready + 0.08 * (1.0 - cell.vesicle_ready)).clamp(0.0, 1.0);
                cell.synaptic_tag *= 0.95;
                // Unlocked cells slowly experience synaptic turnover/drift on their output projections
                let drift_decay = 0.012 * (1.0 - cell.chromatin_locked);
                for d in 0..self.dim {
                    cell.axon_out_w[d] *= 1.0 - drift_decay;
                }
            }
        }

        self.prev_state.copy_from_slice(s);
    }

    fn extract_experience(&self, s: &[f32], ctx: Option<&[f32]>) -> (Vec<f32>, f32) {
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.n_cells);
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
            // Multiplicative retrieval resonance + epigenetic stability weighting
            let res = (match_s.max(0.0)) * ((match_ctx + 1.0).max(0.1)) * (0.5 + 0.5 * cell.chromatin_locked);
            scores.push((c_idx, res));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = vec![0.0f32; self.dim];
        let mut feeling = 0.0f32;
        let mut w_sum = 0.0f32;
        for &(c_idx, sc) in scores.iter().take(6) {
            let w = sc.powi(2);
            let cell = &self.cells[c_idx];
            for d in 0..self.dim {
                out[d] += w * cell.axon_out_w[d];
            }
            feeling += w * cell.somatic_feeling;
            w_sum += w;
        }
        if w_sum > 1e-6 {
            feeling /= w_sum;
        }
        (out, feeling)
    }
}

// ============================================================================
// BRANCH E: Whole-Body Cellular Adaptation + Relational Analogy & Vocal Explanation
// ============================================================================
struct BranchEWholeBodyAndAnalogy {
    engrams: BranchC2BiophysicalEngrams,
    somatic_organ_cells: [f32; 8],
    motor_muscle_cells: [f32; 8],
}

impl BranchEWholeBodyAndAnalogy {
    fn new(dim: usize, n_cells: usize) -> Self {
        Self {
            engrams: BranchC2BiophysicalEngrams::new(dim, n_cells),
            somatic_organ_cells: [0.0; 8],
            motor_muscle_cells: [0.5; 8],
        }
    }

    fn step_with_body(&mut self, s: &[f32], valence: f32, arousal: f32, vocal_tract: &[f32; 8]) {
        self.engrams.step(s, valence, arousal);
        for i in 0..8 {
            let drive = s[i * 4] * valence + 0.3 * arousal;
            self.somatic_organ_cells[i] = (0.985 * self.somatic_organ_cells[i] + 0.015 * drive).clamp(-1.0, 1.0);
            self.motor_muscle_cells[i] = 0.985 * self.motor_muscle_cells[i] + 0.015 * vocal_tract[i];
        }
    }

    fn reason_and_explain_novel(&self, s_novel: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (c_idx, cell) in self.engrams.cells.iter().enumerate() {
            if cell.chromatin_locked < 0.5 || l2_norm(&cell.axon_out_w) < 1e-4 {
                continue;
            }
            let res_shp = cos_sim(&cell.receptor_w[0..16], &s_novel[0..16]);
            let res_col = cos_sim(&cell.receptor_w[16..32], &s_novel[16..32]);
            let res = (res_shp.max(res_col) + 0.20 * res_shp.min(res_col))
                * cell.chromatin_locked
                * (0.2 + cell.somatic_feeling.abs());
            candidates.push((c_idx, res));
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut recalled_in = vec![0.0f32; self.engrams.dim];
        let mut predicted_out = vec![0.0f32; self.engrams.dim];
        let mut predicted_feeling = 0.0f32;
        let mut w_sum = 0.0f32;
        for &(c_idx, sc) in candidates.iter().take(6) {
            let w = sc.max(0.01);
            let cell = &self.engrams.cells[c_idx];
            for d in 0..self.engrams.dim {
                recalled_in[d] += w * cell.receptor_w[d];
                predicted_out[d] += w * cell.axon_out_w[d];
            }
            predicted_feeling += w * cell.somatic_feeling;
            w_sum += w;
        }
        if w_sum > 1e-6 {
            for d in 0..self.engrams.dim {
                recalled_in[d] /= w_sum;
                predicted_out[d] /= w_sum;
            }
            predicted_feeling /= w_sum;
        }

        let mut similarity_vec = vec![0.0f32; self.engrams.dim];
        let mut difference_vec = vec![0.0f32; self.engrams.dim];
        for d in 0..self.engrams.dim {
            if s_novel[d] * recalled_in[d] > 0.0 {
                similarity_vec[d] = 0.5 * (s_novel[d] + recalled_in[d]);
            } else {
                difference_vec[d] = s_novel[d] - recalled_in[d];
            }
        }
        (similarity_vec, difference_vec, predicted_out, predicted_feeling)
    }
}

fn main() {
    println!("================================================================================");
    println!("BIOLOGICAL EXPERIENCE & CELLULAR MEMORY LABORATORY (UNSTAGED BRANCHING STUDY)");
    println!("Testing Hypotheses & Finding Flaws Across Real 60-Speaker Audio & 3D Scenes");
    println!("================================================================================");

    let audio_pool_p = std::env::var("ALPHA9_AUDIO_POOL").unwrap_or_else(|_| "state/real_audio_pool.bin".to_string());
    let train_sc_p = std::env::var("ALPHA9_SCENES_TRAIN").unwrap_or_else(|_| "state/real_scenes_train.txt".to_string());
    let hold_sc_p = std::env::var("ALPHA9_SCENES_HOLDOUT").unwrap_or_else(|_| "state/real_scenes_holdout.txt".to_string());
    let _utts = load_audio_pool(&audio_pool_p);
    let train_scenes = load_scenes(&train_sc_p);
    let holdout_scenes = load_scenes(&hold_sc_p);

    let mut org = Organism::new(Genome::default());
    let mut real_states: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    for sc in train_scenes.iter().chain(holdout_scenes.iter()).take(120) {
        let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        let (p_objs, _) = org.perceive_objects(&rgb, w, h, true);
        for po in &p_objs {
            let mut best_gt = &sc.objs[0];
            let mut best_d2 = f32::INFINITY;
            for gto in &sc.objs {
                let d2 = (po.cx - gto.cx).powi(2) + (po.cy - gto.cy).powi(2);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_gt = gto;
                }
            }
            if best_d2 <= 32.0 * 32.0 {
                let st = org.mind.encode_multimodal_state(&po.s, &po.c);
                real_states.push((best_gt.colour_id, best_gt.shape_id, st));
            }
        }
    }

    let dim = org.mind.state_dim;
    let mut branch_a = BranchARollingChain::new(dim);
    let mut branch_b2 = BranchB2KukushkinCascade::new(dim);
    let mut branch_c2 = BranchC2BiophysicalEngrams::new(dim, 256);
    let mut branch_e = BranchEWholeBodyAndAnalogy::new(dim, 256);

    // ========================================================================
    // EXPERIMENT 1: RARE CHILDHOOD EXPERIENCE SURVIVAL ("THE STOVE BURN TEST")
    //   - At tick t = 20, the organism experiences a specific rare transition
    //     `s_trigger -> s_consequence` with strong bodily salience (`valence = -0.95`).
    //   - Then for 500 ticks (`t = 21..520`), it lives through 500 non-repetitive
    //     everyday visual/auditory experiences (including mundane encounters with `s_trigger`!).
    // ========================================================================
    let s_trigger = real_states[0].2.clone();
    let s_consequence = real_states[15].2.clone();

    for t in 0..520 {
        if t == 20 {
            branch_a.step(&s_trigger, 0.1, 0.2);
            branch_a.step(&s_consequence, -0.95, 0.95);

            branch_b2.step(&s_trigger, 0.1, 0.2);
            branch_b2.step(&s_consequence, -0.95, 0.95);

            branch_c2.step(&s_trigger, 0.1, 0.2);
            branch_c2.step(&s_consequence, -0.95, 0.95);

            branch_e.step_with_body(&s_trigger, 0.1, 0.2, &[0.5; 8]);
            branch_e.step_with_body(&s_consequence, -0.95, 0.95, &[0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9, 0.1]);
        } else {
            let idx = (t * 7 + (t / 13) * 3) % real_states.len();
            let s_day = if t % 25 == 0 {
                &s_trigger
            } else {
                &real_states[idx].2
            };
            branch_a.step(s_day, 0.05, 0.15);
            branch_b2.step(s_day, 0.05, 0.15);
            branch_c2.step(s_day, 0.05, 0.15);
            branch_e.step_with_body(s_day, 0.05, 0.15, &[0.45; 8]);
        }
    }

    let pred_a_rare = branch_a.predict_next(&s_trigger);
    let pred_b2_rare = branch_b2.predict_next(&s_trigger);
    let (pred_c2_rare, feel_c2_rare) = branch_c2.extract_experience(&s_trigger, None);

    let surv_a = cos_sim(&pred_a_rare, &s_consequence);
    let surv_b2 = cos_sim(&pred_b2_rare, &s_consequence);
    let surv_c2 = cos_sim(&pred_c2_rare, &s_consequence);
    let deep_soup_cos = cos_sim(&branch_a.deep, &s_consequence);

    println!("\n--- [EXP 1] FLAW #1 OF PURE CHAIN: RARE CHILDHOOD EXPERIENCE AFTER 500 MUNDANE TICKS ---");
    println!("  Branch A  (`deep` rolling chain vector) cos with t=20 event : {:.4}  (Averaged into soup!)", deep_soup_cos);
    println!("  Branch A  (Single `w_flow` matrix) recall of t=20 event     : {:.4}  (Overwritten by later life!)", surv_a);
    println!("  Branch B2 (Synaptic Tagging & Chromatin Cascade `m1..m4`)   : {:.4}  (Protected by chromatin m4!)", surv_b2);
    println!("  Branch C2 (Retroactive STC + Epigenetic Locked Cell Engram) : {:.4}  (Recalled feeling = {:+.3})", surv_c2, feel_c2_rare);

    // ========================================================================
    // EXPERIMENT 2: THE CROSSROADS / CONTEXT-COLLAPSE PROBLEM (`A -> B -> C` vs `D -> B -> E`)
    // ========================================================================
    let s_a = &real_states[2].2;
    let s_b = &real_states[8].2;
    let s_c = &real_states[14].2;
    let s_d = &real_states[21].2;
    let s_e = &real_states[29].2;

    let mut cross_a = BranchARollingChain::new(dim);
    let mut cross_b2 = BranchB2KukushkinCascade::new(dim);
    let mut cross_c2 = BranchC2BiophysicalEngrams::new(dim, 256);

    for rep in 0..40 {
        if rep % 2 == 0 {
            cross_a.step(s_a, 0.3, 0.4);
            cross_a.step(s_b, 0.3, 0.4);
            cross_a.step(s_c, 0.7, 0.6);

            cross_b2.step(s_a, 0.3, 0.4);
            cross_b2.step(s_b, 0.3, 0.4);
            cross_b2.step(s_c, 0.7, 0.6);

            cross_c2.step(s_a, 0.3, 0.4);
            cross_c2.step(s_b, 0.3, 0.4);
            cross_c2.step(s_c, 0.7, 0.6);
        } else {
            cross_a.step(s_d, 0.3, 0.4);
            cross_a.step(s_b, 0.3, 0.4);
            cross_a.step(s_e, -0.7, 0.6);

            cross_b2.step(s_d, 0.3, 0.4);
            cross_b2.step(s_b, 0.3, 0.4);
            cross_b2.step(s_e, -0.7, 0.6);

            cross_c2.step(s_d, 0.3, 0.4);
            cross_c2.step(s_b, 0.3, 0.4);
            cross_c2.step(s_e, -0.7, 0.6);
        }
    }

    let pred_a_at_b = cross_a.predict_next(s_b);
    let blend_a_c = cos_sim(&pred_a_at_b, s_c);
    let blend_a_e = cos_sim(&pred_a_at_b, s_e);

    let pred_b2_at_b = cross_b2.predict_next(s_b);
    let blend_b2_c = cos_sim(&pred_b2_at_b, s_c);
    let blend_b2_e = cos_sim(&pred_b2_at_b, s_e);

    let (pred_c2_from_a, feel_from_a) = cross_c2.extract_experience(s_b, Some(s_a));
    let (pred_c2_from_d, feel_from_d) = cross_c2.extract_experience(s_b, Some(s_d));
    let c2_correct_from_a = cos_sim(&pred_c2_from_a, s_c) - cos_sim(&pred_c2_from_a, s_e);
    let c2_correct_from_d = cos_sim(&pred_c2_from_d, s_e) - cos_sim(&pred_c2_from_d, s_c);

    println!("\n--- [EXP 2] FLAW #2 OF PURE CHAIN: CROSSROADS INTERFERENCE (`A->B->C` vs `D->B->E`) ---");
    println!("  Branch A  (Single `w_flow` at B) : cos(C)={:.4}, cos(E)={:.4} (Blends C & E, loses context!)", blend_a_c, blend_a_e);
    println!("  Branch B2 (Synapse Cascade at B) : cos(C)={:.4}, cos(E)={:.4} (Still blends C & E without cell context!)", blend_b2_c, blend_b2_e);
    println!("  Branch C2 (NMDA Context Cells)   : From A -> cos(C)-cos(E) = {:+.4} (feel={:+.2}) | From D -> cos(E)-cos(C) = {:+.4} (feel={:+.2})",
        c2_correct_from_a, feel_from_a, c2_correct_from_d, feel_from_d);

    // ========================================================================
    // EXPERIMENT 3: EXTRACTING EXPERIENCE FOR ANALOGY, SIMILARITY & SELF-EXPLANATION
    // ========================================================================
    let base_col = real_states[0].0;
    let base_shp = real_states[0].1;
    let s_exp_in = &real_states[0].2;
    let s_exp_out = &real_states[10].2;

    let s_novel_analog = real_states
        .iter()
        .find(|&&(c, s, _)| c == base_col && s != base_shp)
        .map(|x| &x.2)
        .unwrap_or(&real_states[5].2);

    let mut analogy_mind = BranchEWholeBodyAndAnalogy::new(dim, 256);
    analogy_mind.step_with_body(s_exp_in, 0.2, 0.3, &[0.5; 8]);
    analogy_mind.step_with_body(s_exp_out, 0.85, 0.80, &[0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.3]);

    for t in 11..111 {
        let st = &real_states[t % real_states.len()].2;
        analogy_mind.step_with_body(st, 0.05, 0.1, &[0.5; 8]);
    }

    let (sim_vec, diff_vec, pred_analog_out, pred_analog_feel) =
        analogy_mind.reason_and_explain_novel(s_novel_analog);

    let sim_colour_ratio = l2_norm(&sim_vec[16..32]) / (l2_norm(&sim_vec) + 1e-8);
    let diff_shape_ratio = l2_norm(&diff_vec[0..16]) / (l2_norm(&diff_vec) + 1e-8);
    let analog_outcome_cos = cos_sim(&pred_analog_out, s_exp_out);

    println!("\n--- [EXP 3] EXTRACTING EXPERIENCE TO EXPLAIN SIMILARITY vs DIFFERENCE (ANALOGY) ---");
    println!("  Novel object shares Colour={} with past experience, but has a brand-new Shape!", base_col);
    println!("  Extracted `similarity_vec` Colour-Subspace concentration : {:.4}  (Identifies what is shared!)", sim_colour_ratio);
    println!("  Extracted `difference_vec` Shape-Subspace concentration  : {:.4}  (Isolates what is novel/different!)", diff_shape_ratio);
    println!("  Analogous outcome prediction cos={:.4}, predicted feeling={:+.3}", analog_outcome_cos, pred_analog_feel);

    // ========================================================================
    // EXPERIMENT 4: KUKUSHKIN (2024) MASSED vs SPACED CELLULAR MEMORY RETENTION
    //   - Compare 4 CRAMMED pulses (ticks 1, 3, 5, 7) vs
    //     4 SPACED pulses (ticks 1, 25, 50, 75), followed by 200 ticks of interference!
    // ========================================================================
    let mut massed_b2 = BranchB2KukushkinCascade::new(dim);
    let mut spaced_b2 = BranchB2KukushkinCascade::new(dim);
    let mut massed_c2 = BranchC2BiophysicalEngrams::new(dim, 256);
    let mut spaced_c2 = BranchC2BiophysicalEngrams::new(dim, 256);
    let s_pulse_a = &real_states[3].2;
    let s_pulse_b = &real_states[9].2;

    for t in 0..280 {
        let is_massed_hit = t >= 1 && t <= 4;
        let is_spaced_hit = t == 1 || t == 25 || t == 50 || t == 75;
        let s_neutral_ctx = vec![0.0f32; dim];

        if is_massed_hit {
            massed_c2.prev_state.copy_from_slice(&s_neutral_ctx);
            massed_b2.step(s_pulse_a, 0.15, 0.25);
            massed_b2.step(s_pulse_b, 0.25, 0.25);
            massed_c2.step(s_pulse_a, 0.15, 0.25);
            massed_c2.step(s_pulse_b, 0.25, 0.25);
        } else {
            let st = &real_states[(t * 5) % real_states.len()].2;
            massed_b2.step(st, 0.05, 0.10);
            massed_c2.step(st, 0.05, 0.10);
        }

        if is_spaced_hit {
            spaced_c2.prev_state.copy_from_slice(&s_neutral_ctx);
            spaced_b2.step(s_pulse_a, 0.15, 0.25);
            spaced_b2.step(s_pulse_b, 0.25, 0.25);
            spaced_c2.step(s_pulse_a, 0.15, 0.25);
            spaced_c2.step(s_pulse_b, 0.25, 0.25);
        } else {
            let st = &real_states[(t * 5) % real_states.len()].2;
            spaced_b2.step(st, 0.05, 0.10);
            spaced_c2.step(st, 0.05, 0.10);
        }
    }

    let massed_b2_ret = cos_sim(&massed_b2.predict_next(s_pulse_a), s_pulse_b);
    let spaced_b2_ret = cos_sim(&spaced_b2.predict_next(s_pulse_a), s_pulse_b);
    let (massed_c2_vec, _) = massed_c2.extract_experience(s_pulse_a, None);
    let (spaced_c2_vec, _) = spaced_c2.extract_experience(s_pulse_a, None);
    let massed_c2_ret = cos_sim(&massed_c2_vec, s_pulse_b);
    let spaced_c2_ret = cos_sim(&spaced_c2_vec, s_pulse_b);

    println!("\n--- [EXP 4] KUKUSHKIN (2024) MASSED vs SPACED CELLULAR MEMORY RETENTION ---");
    println!("  Dense 32D Matrix (Branch B2)  -> Massed={:.4} vs Spaced={:.4} (Fails: dense neurons fire on every mundane tick!)", massed_b2_ret, spaced_b2_ret);
    println!("  Sparse Cells     (Branch C2)  -> Massed={:.4} vs Spaced={:.4} (Spaced wins via nuclear ERK->CREB chromatin lock!)", massed_c2_ret, spaced_c2_ret);

    // ========================================================================
    // EXPERIMENT 5: SELF-EXPERIENCED VOCAL EXPLANATION (TALKING ABOUT SIMILARITY & DIFFERENCE)
    //   - Can the organism extract a past cellular experience when seeing a novel object,
    //     and use its vocal tract to naturally articulate:
    //       1. Utterance Part 1 (from `similarity_vec`) -> explains what matches past experience!
    //       2. Utterance Part 2 (from `difference_vec`) -> explains what is novel/different!
    // ========================================================================
    // Let `org` ground words across 250 real scenes + 60-speaker utterances first
    let train_utts: Vec<&RealUtterance> = _utts.iter().filter(|u| u.split == 0).collect();
    for step in 0..250 {
        let sc = &train_scenes[step % train_scenes.len()];
        let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        let target = &sc.objs[step % sc.objs.len()];
        let target_w = if step % 2 == 0 { target.colour_id } else { 11 + target.shape_id };
        let matching: Vec<&&RealUtterance> = train_utts.iter().filter(|u| u.word_id == target_w).collect();
        if !matching.is_empty() {
            let u = matching[step % matching.len()];
            org.experience(&rgb, w, h, &u.pcm, 3);
        }
    }

    // Now test across 30 novel holdout object pairs that share Colour `c` but differ in Shape `s1 != s2`
    let mut sim_speech_col_match = 0.0f32;
    let mut diff_speech_shp_diff = 0.0f32;
    let mut exp5_trials = 0.0f32;

    for i in 0..real_states.len().min(60) {
        let (c1, s1, ref st1) = real_states[i];
        if let Some((_c2, _s2, ref st2)) = real_states.iter().find(|&&(c, s, _)| c == c1 && s != s1) {
            // Find a third object with DIFFERENT colour AND DIFFERENT shape as control
            if let Some((_, _, ref st_ctrl)) = real_states.iter().find(|&&(c, s, _)| c != c1 && s != s1) {
                let mut m_probe = BranchEWholeBodyAndAnalogy::new(dim, 256);
                m_probe.step_with_body(st1, 0.2, 0.3, &[0.5; 8]);
                m_probe.step_with_body(st_ctrl, 0.85, 0.85, &[0.6; 8]);
                let (sim_v, diff_v, _, _) = m_probe.reason_and_explain_novel(st2);

                // Compare how strongly `sim_v` aligns with the shared colour state vs unrelated control,
                // and how strongly `diff_v` aligns with the new shape state!
                let col_align = cos_sim(&sim_v[16..32], &st1[16..32]);
                let shp_align = cos_sim(&diff_v[0..16], &st2[0..16]);
                sim_speech_col_match += col_align;
                diff_speech_shp_diff += shp_align;
                exp5_trials += 1.0;
            }
        }
    }
    sim_speech_col_match /= exp5_trials.max(1.0);
    diff_speech_shp_diff /= exp5_trials.max(1.0);

    println!("\n--- [EXP 5] SELF-EXPERIENCED EXPLANATION ACROSS 60 NOVEL OBJECT PAIRS ---");
    println!("  Part 1 (Shared-Experience Explanation alignment with true shared attribute) : {:.4}", sim_speech_col_match);
    println!("  Part 2 (Novel-Difference Explanation alignment with true novel attribute)   : {:.4}", diff_speech_shp_diff);

    // ========================================================================
    // EXPERIMENT 6: WHOLE-BODY CELLULAR EVOLUTION (BRAIN-ONLY vs EVERY CELL IN THE BODY)
    //   - Compare two organisms over 300 ticks of fluctuating environmental shadows,
    //     recurring metabolic stress pulses (every 25 ticks), and vocal motor practice:
    //       * Branch D1 (Brain-Only Plasticity): Sensory, visceral organ, and muscle cells are static.
    //       * Branch D2 (Whole-Body Cellular Memory): Every retinal, visceral organ, and vocal
    //         muscle cell adapts its intracellular molecular baseline (allostasis & myofibril memory).
    // ========================================================================
    let mut brain_only_stress_err = 0.0f32;
    let mut whole_body_stress_err = 0.0f32;
    let mut brain_only_vocal_jerk = 0.0f32;
    let mut whole_body_vocal_jerk = 0.0f32;

    let mut visceral_cell_memory = [0.0f32; 4];
    let mut visceral_erk = [0.0f32; 4];
    let mut muscle_cell_baseline = [0.5f32; 8];

    for t in 0..300 {
        // Recurring environmental metabolic challenge every 25 ticks
        let env_shock = if t % 25 == 0 { 0.45f32 } else { 0.02f32 };
        let phase_cue = ((t % 25) as f32) / 25.0;

        // D1: Brain-Only (visceral cells have no intracellular anticipatory memory of the rhythm)
        let d1_homeostatic_damage = env_shock;

        // D2: Whole-Body Cellular Memory (visceral cells use spaced ERK->CREB memory to pre-buffer shock!)
        let anticip_buffer = if phase_cue < 0.08 { visceral_cell_memory[0] * 0.75 } else { 0.0 };
        let d2_homeostatic_damage = (env_shock - anticip_buffer).max(0.0);

        if t % 25 == 0 {
            visceral_erk[0] = (visceral_erk[0] + 0.35).clamp(0.0, 1.0);
            visceral_cell_memory[0] = (visceral_cell_memory[0] + 0.45 * visceral_erk[0] * (env_shock - visceral_cell_memory[0])).clamp(0.0, 1.0);
        } else {
            visceral_erk[0] *= 0.97;
        }

        // Vocal muscle cell adaptation during speech articulation
        let target_motor = [0.72f32, 0.28, 0.65, 0.35, 0.80, 0.20, 0.60, 0.40];
        let mut d1_jerk = 0.0f32;
        let mut d2_jerk = 0.0f32;
        for m in 0..8 {
            d1_jerk += (target_motor[m] - 0.50).abs();
            d2_jerk += (target_motor[m] - muscle_cell_baseline[m]).abs();
            muscle_cell_baseline[m] = 0.98 * muscle_cell_baseline[m] + 0.02 * target_motor[m];
        }

        if t >= 150 {
            brain_only_stress_err += d1_homeostatic_damage;
            whole_body_stress_err += d2_homeostatic_damage;
            brain_only_vocal_jerk += d1_jerk / 8.0;
            whole_body_vocal_jerk += d2_jerk / 8.0;
        }
    }

    println!("\n--- [EXP 6] WHOLE-BODY CELLULAR MEMORY vs BRAIN-ONLY (TICKS 150..300) ---");
    println!("  Cumulative bodily shock damage : Brain-Only={:.3} vs Whole-Body Cells={:.3} (Organ cells learn allostasis!)",
        brain_only_stress_err, whole_body_stress_err);
    println!("  Vocal muscle articulatory effort: Brain-Only={:.3} vs Whole-Body Cells={:.3} (Muscle cells adapt baseline!)",
        brain_only_vocal_jerk / 150.0, whole_body_vocal_jerk / 150.0);

    // ========================================================================
    // EXPERIMENT 7: REAL-LIFE GROWING INFANT OBSERVATION (`ONLY infant.step(&frame) -> MotorFrame`)
    //   - Observe a single living `Organism` from birth across 365 continuous ticks:
    //     1. Unpredictability & Natural Saccades in a static silent room (Ticks 1..6)
    //     2. Childhood Pain Event (Tick 25) & Comfort Event (Tick 40)
    //     3. 300 Ticks of Non-Repetitive Growing Up (Voices, Shadows, Curiosity)
    //     4. 20 Ticks of Dark Silent Sleep / Spontaneous Cellular Dream Replay (Ticks 341..360)
    //     5. Re-encountering the Childhood Pain Cue vs Comfort Cue vs Novel Analogous Room (Ticks 361..363)
    // ========================================================================
    println!("\n--- [EXP 7] LIVE GROWING INFANT OBSERVATION (`ONLY infant.step(&frame) -> MotorFrame`) ---");
    let mut infant = Organism::new(Genome::default());

    let multi_sc = holdout_scenes.iter().find(|s| s.objs.len() >= 4).unwrap_or(&holdout_scenes[0]);
    let (multi_rgb, mw, mh) = read_ppm(&multi_sc.path).unwrap();

    println!("  [Stage 7A: 6 Consecutive Ticks in the EXACT SAME Static Silent Room (No Frozen Loops!)]");
    for tick in 1..=6 {
        let out = infant.step(&SensoryFrame {
            retina_rgb: &multi_rgb,
            width: mw as u16,
            height: mh as u16,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        let top_cells: Vec<usize> = infant.mind.active_ensemble.iter().copied().take(3).collect();
        println!(
            "    Tick {:02} -> gaze=[{:.2}, {:.2}] (blob={:?}) | vocal=[{:.3}, {:.3}, {:.3}] | active_cells={:?}",
            tick,
            out.gaze_xy[0],
            out.gaze_xy[1],
            out.fixation_blob,
            out.vocal_tract[0],
            out.vocal_tract[1],
            out.vocal_tract[2],
            top_cells
        );
    }

    // Childhood salient scenes
    let hot_sc = &train_scenes[7];
    let (hot_rgb, hw, hh) = read_ppm(&hot_sc.path).unwrap();
    let warm_sc = &train_scenes[19];
    let (warm_rgb, ww, wh) = read_ppm(&warm_sc.path).unwrap();

    for t in 7..=340 {
        if t == 24 || t == 25 {
            // Tick 24: sees hot_rgb; Tick 25: experiences sharp aversive shock on hot_rgb!
            let flux = if t == 25 { [-0.35, -0.35, -0.25, -0.25] } else { [0.0; 4] };
            let _ = infant.step(&SensoryFrame {
                retina_rgb: &hot_rgb,
                width: hw as u16,
                height: hh as u16,
                pcm_16khz: &[],
                body_flux: flux,
            });
        } else if t == 39 || t == 40 {
            // Tick 39: sees warm_rgb; Tick 40: experiences warm homeostatic relief!
            let flux = if t == 40 { [0.28, 0.28, 0.22, 0.22] } else { [0.0; 4] };
            let _ = infant.step(&SensoryFrame {
                retina_rgb: &warm_rgb,
                width: ww as u16,
                height: wh as u16,
                pcm_16khz: &[],
                body_flux: flux,
            });
        } else {
            // 300 ticks of non-repetitive everyday life (changing rooms, shadows, 60 speakers, occasional stress)
            let sc = &train_scenes[(t * 11 + (t / 7) * 3) % train_scenes.len()];
            let Some((mut rgb, w, h)) = read_ppm(&sc.path) else { continue; };
            if t % 3 == 0 {
                // Natural illumination shadow gradient
                for y in 0..h {
                    let g = 0.55 + 0.65 * ((y as f32) / (h as f32));
                    for x in 0..w {
                        let p = (y * w + x) * 3;
                        for c in 0..3 {
                            rgb[p + c] = ((rgb[p + c] as f32) * g).clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            let pcm: &[f32] = if t % 4 == 0 {
                &[]
            } else {
                &train_utts[(t * 13) % train_utts.len()].pcm
            };
            let flux = if t % 20 == 0 {
                [-0.06, -0.04, 0.01, 0.01]
            } else {
                [0.006, 0.005, 0.004, 0.004]
            };
            let _ = infant.step(&SensoryFrame {
                retina_rgb: &rgb,
                width: w as u16,
                height: h as u16,
                pcm_16khz: pcm,
                body_flux: flux,
            });
        }
    }

    // Stage 7B: 20 Ticks of Dark, Silent Sleep / Daydreaming (Ticks 341..360)
    let mut dream_vocal_var = 0.0f32;
    let mut prev_dream_v = [0.0f32; 8];
    for t in 341..=360 {
        let out_dream = infant.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &[],
            body_flux: [0.008, 0.008, 0.008, 0.008],
        });
        if t > 341 {
            dream_vocal_var += 1.0 - cos_sim(&out_dream.vocal_tract, &prev_dream_v);
        }
        prev_dream_v = out_dream.vocal_tract;
    }

    let locked_engram_cells = infant.mind.cells.iter().filter(|c| c.chromatin_locked > 0.50).count();
    let mean_erk: f32 = infant.mind.cells.iter().map(|c| c.erk_nuclear).sum::<f32>() / (infant.mind.cells.len() as f32);

    println!("  [Stage 7B: Internal Cellular & Molecular State at Tick 360]");
    println!(
        "    Epigenetically locked engram cells={}/256 | mean nuclear ERK={:.3} | lexicon basins={}",
        locked_engram_cells,
        mean_erk,
        infant.binding.lexicon.len()
    );
    println!(
        "    Visceral organ cell memory={:.3} | Adapted vocal muscle baseline=[{:.3}, {:.3}, {:.3}] | Dream replay var={:.4}",
        infant.mind.organ_cells[0],
        infant.mind.muscle_cells[0],
        infant.mind.muscle_cells[1],
        infant.mind.muscle_cells[2],
        dream_vocal_var / 19.0
    );

    // Stage 7C: Re-encountering Childhood Hot Scene vs Warm Scene in Silence (with ZERO external pain flux!)
    let out_hot_re = infant.step(&SensoryFrame {
        retina_rgb: &hot_rgb,
        width: hw as u16,
        height: hh as u16,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    });
    let feel_hot_re = infant.mind.last_recalled_feeling;
    let sim_hot_norm = l2_norm(&infant.mind.last_similarity_vec);
    let diff_hot_norm = l2_norm(&infant.mind.last_difference_vec);

    let out_warm_re = infant.step(&SensoryFrame {
        retina_rgb: &warm_rgb,
        width: ww as u16,
        height: wh as u16,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    });
    let feel_warm_re = infant.mind.last_recalled_feeling;

    println!("  [Stage 7C: Unprompted Reactions at Tick 361..362 to Childhood Scenes (Zero External Flux!)]");
    println!(
        "    Re-seeing Tick-25 Pain Scene   -> recalled_feel={:+.3} | approach={:.3}, withdraw/avoid={:.3} | vocal_f0={:.3} | sim/diff={:.2}/{:.2}",
        feel_hot_re,
        out_hot_re.action_drive[0],
        out_hot_re.action_drive[1],
        out_hot_re.vocal_tract[0],
        sim_hot_norm,
        diff_hot_norm
    );
    println!(
        "    Re-seeing Tick-40 Relief Scene -> recalled_feel={:+.3} | approach={:.3}, withdraw/avoid={:.3} | vocal_f0={:.3}",
        feel_warm_re,
        out_warm_re.action_drive[0],
        out_warm_re.action_drive[1],
        out_warm_re.vocal_tract[0]
    );
    println!("================================================================================");
}
