//! Alpha 9 Biophysical, Feeling-Driven Emergent Organism (`alpha-organism`).
//! Law 2: Contains ZERO string literals, ZERO text comparisons, and ZERO world dependencies.

pub mod body;
pub mod cortex;
pub mod curiosity;
pub mod forward_speech;
pub mod genome;
pub mod heatmap;
pub mod hippocampus;
pub mod sensors;
pub mod util;
pub mod vocal;

use alpha_iface::{MotorFrame, SensoryFrame};
use body::HomeostaticBody;
use cortex::{CanonicalArea, CompetitiveArea};
use curiosity::DiscoveryArchive;
use forward_speech::FeelingSpeechEngine;
use genome::Genome;
use heatmap::BiophysicalHeatMap;
use hippocampus::{CrossModalBinding, PerceivedObject};
use sensors::{Cochlea, Retina, RetinalBlob};
use util::{dot, l2_norm, normalize_l2_in_place, Rng64};
use vocal::VocalTract;

pub struct Organism {
    pub genome: Genome,
    pub rng: Rng64,
    pub body: HomeostaticBody,
    pub heatmap: BiophysicalHeatMap,
    pub discovery: DiscoveryArchive,
    pub speech_engine: FeelingSpeechEngine,
    pub cochlea: Cochlea,
    pub a1: CompetitiveArea,
    pub wa: CanonicalArea,
    pub v1: CompetitiveArea,
    pub v4: CompetitiveArea,
    pub it: CanonicalArea,
    pub c4: CanonicalArea,
    pub binding: CrossModalBinding,
    pub vocal: VocalTract,
    pub step_count: u64,
}

impl Organism {
    pub fn new(genome: Genome) -> Self {
        let mut rng = Rng64::new(genome.seed);
        let body = HomeostaticBody::default();
        let heatmap = BiophysicalHeatMap::new(genome.heatmap_dim);
        let discovery = DiscoveryArchive::new(64);
        let speech_engine = FeelingSpeechEngine::new(32, &mut rng);
        let cochlea = Cochlea::new(&genome);
        let a1 = CompetitiveArea::new(
            genome.cochlea_bands * genome.a1_ctx,
            genome.a1_units,
            genome.a1_k,
            genome.eta_comp,
            &mut rng,
        );
        let wa = CanonicalArea::new(
            3 * genome.a1_units,
            genome.wa_units,
            genome.wa_k,
            0.10,
            genome.beta_assembly,
            &mut rng,
        );
        let v1 = CompetitiveArea::new(25, genome.v1_units, genome.v1_k, genome.eta_comp, &mut rng);
        let v4 = CompetitiveArea::new(3, genome.v4_units, genome.v4_k, 0.03, &mut rng);
        let it = CanonicalArea::new(
            genome.v1_units * 16,
            genome.it_units,
            genome.it_k,
            0.10,
            genome.beta_assembly,
            &mut rng,
        );
        let c4 = CanonicalArea::new(
            genome.v4_units,
            genome.c4_units,
            genome.c4_k,
            0.30,
            genome.beta_assembly,
            &mut rng,
        );
        let binding = CrossModalBinding::new(
            genome.wa_units,
            genome.wa_k,
            genome.it_units,
            genome.it_k,
            genome.c4_units,
            genome.c4_k,
            genome.lex_overlap_thresh,
        );
        let vocal = VocalTract::new(genome.sr_hz, 3 * genome.a1_units, &mut rng);

        Self {
            genome,
            rng,
            body,
            heatmap,
            discovery,
            speech_engine,
            cochlea,
            a1,
            wa,
            v1,
            v4,
            it,
            c4,
            binding,
            vocal,
            step_count: 0,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.heatmap.memory_bytes()
            + self.discovery.memory_bytes()
            + self.speech_engine.memory_bytes()
            + self.cochlea.memory_bytes()
            + self.a1.memory_bytes()
            + self.wa.memory_bytes()
            + self.v1.memory_bytes()
            + self.v4.memory_bytes()
            + self.it.memory_bytes()
            + self.c4.memory_bytes()
            + self.binding.memory_bytes()
            + self.vocal.memory_bytes()
    }

    pub fn ram_pressure_ok(&self) -> bool {
        self.memory_bytes() <= self.genome.max_ram_bytes
    }

    pub fn a1_codes(&mut self, seg: &[Vec<f32>], learn: bool) -> Vec<Vec<f32>> {
        let ctx = self.genome.a1_ctx;
        let nb = self.genome.cochlea_bands;
        let mut padded = seg.to_vec();
        while padded.len() < ctx {
            let last = padded.last().cloned().unwrap_or_else(|| vec![0.0; nb]);
            padded.push(last);
        }
        let n_windows = padded.len() - ctx + 1;
        let mut codes = Vec::with_capacity(n_windows);
        let mut patch = Vec::with_capacity(ctx * nb);
        for t in 0..n_windows {
            patch.clear();
            for c in 0..ctx {
                patch.extend_from_slice(&padded[t + c]);
            }
            if learn {
                self.a1.learn(&patch);
            }
            let (r, _) = self.a1.respond(&patch);
            codes.push(r);
        }
        codes
    }

    pub fn word_form(&mut self, pcm: &[f32], learn_a1: bool) -> Vec<f32> {
        let seg = self.cochlea.transduce(pcm);
        let r = self.a1_codes(&seg, learn_a1);
        let n = r.len().max(1);
        let u = self.genome.a1_units;
        let mut v = vec![0.0f32; 3 * u];
        for (i, row) in r.iter().enumerate() {
            let third = ((i * 3) / n).min(2);
            let dst = &mut v[third * u..(third + 1) * u];
            for (dv, &rv) in dst.iter_mut().zip(row.iter()) {
                *dv += rv;
            }
        }
        normalize_l2_in_place(&mut v);
        v
    }

    pub fn develop_hearing(&mut self, pcm: &[f32]) {
        let seg = self.cochlea.transduce(pcm);
        let mut s = 0usize;
        while s < seg.len() {
            let end = (s + 120).min(seg.len());
            self.a1_codes(&seg[s..end], true);
            s += 120;
        }
    }

    pub fn hear_word(&mut self, pcm: &[f32], learn: bool) -> (Vec<f32>, bool) {
        let v = self.word_form(pcm, false);
        if learn {
            let a = self.wa.fire_and_learn(&v, None);
            let (_, is_novel) = self.binding.update_lexicon(&a, self.genome.wa_k as f32);
            (a, is_novel)
        } else {
            (self.wa.fire(&v), false)
        }
    }

    pub fn v1_features(&mut self, fovea: &[f32], learn: bool) -> Vec<f32> {
        let fov = self.genome.fovea_dim;
        let grid_dim = (fov - 5) / 2 + 1;
        let u = self.genome.v1_units;
        let mut responses = vec![vec![0.0f32; u]; grid_dim * grid_dim];
        let mut patch = [0.0f32; 25];

        for gy in 0..grid_dim {
            let py = gy * 2;
            for gx in 0..grid_dim {
                let px = gx * 2;
                let mut mean = 0.0f32;
                let mut idx = 0usize;
                for dy in 0..5 {
                    for dx in 0..5 {
                        let val = fovea[(py + dy) * fov + (px + dx)];
                        patch[idx] = val;
                        mean += val;
                        idx += 1;
                    }
                }
                mean /= 25.0;
                for v in &mut patch {
                    *v -= mean;
                }
                if l2_norm(&patch) > 0.05 {
                    if learn {
                        self.v1.learn(&patch);
                    }
                    let (r, _) = self.v1.respond(&patch);
                    responses[gy * grid_dim + gx] = r;
                }
            }
        }

        let mut pooled = vec![0.0f32; 16 * u];
        for by in 0..4 {
            let y0 = (by * grid_dim) / 4;
            let y1 = ((by + 1) * grid_dim) / 4;
            for bx in 0..4 {
                let x0 = (bx * grid_dim) / 4;
                let x1 = ((bx + 1) * grid_dim) / 4;
                let cell = &mut pooled[(by * 4 + bx) * u..(by * 4 + bx + 1) * u];
                for gy in y0..y1 {
                    for gx in x0..x1 {
                        let r = &responses[gy * grid_dim + gx];
                        for (cv, &rv) in cell.iter_mut().zip(r.iter()) {
                            if rv > *cv {
                                *cv = rv;
                            }
                        }
                    }
                }
            }
        }
        normalize_l2_in_place(&mut pooled);
        pooled
    }

    pub fn v4_features(&mut self, colour_opp: &[f32; 3], learn: bool) -> Vec<f32> {
        let shifted = [colour_opp[0] - 1.0, colour_opp[1], colour_opp[2]];
        if learn {
            self.v4.learn(&shifted);
        }
        let (mut r, _) = self.v4.respond(&shifted);
        normalize_l2_in_place(&mut r);
        r
    }

    pub fn develop_vision(&mut self, rgb: &[u8], width: usize, height: usize) {
        let (blobs, _) = Retina::segment(rgb, width, height);
        let fov = self.genome.fovea_dim;
        for b in &blobs {
            let patch = Retina::log_polar_form(b, fov, fov);
            self.v1_features(&patch, true);
            self.v4_features(&b.colour_opp, true);
        }
    }

    pub fn perceive_objects(&mut self, rgb: &[u8], width: usize, height: usize, learn: bool) -> (Vec<PerceivedObject>, Vec<RetinalBlob>) {
        let (blobs, _) = Retina::segment(rgb, width, height);
        let fov = self.genome.fovea_dim;
        let mut objs = Vec::with_capacity(blobs.len());
        for b in &blobs {
            let fs = self.v1_features(&Retina::log_polar_form(b, fov, fov), false);
            let fc = self.v4_features(&b.colour_opp, false);
            let s = if learn { self.it.fire_and_learn(&fs, None) } else { self.it.fire(&fs) };
            let c = if learn { self.c4.fire_and_learn(&fc, None) } else { self.c4.fire(&fc) };
            objs.push(PerceivedObject {
                cy: b.cy,
                cx: b.cx,
                s,
                c,
            });
        }
        (objs, blobs)
    }

    pub fn experience(&mut self, rgb: &[u8], width: usize, height: usize, pcm: &[f32], max_fixations: usize) {
        let (a, is_novel) = self.hear_word(pcm, true);
        let (objs, _) = self.perceive_objects(rgb, width, height, true);
        if objs.is_empty() {
            return;
        }
        let perm = self.rng.shuffle_indices(objs.len());
        let take = max_fixations.min(perm.len());
        let eff_eta = self.body.emotional_lr(self.genome.eta_bind);
        self.binding.bind_experience(&a, &objs, &perm[..take], eff_eta, is_novel);
        self.step_count += 1;
    }

    pub fn attend_to_word(&mut self, pcm: &[f32], objs: &[PerceivedObject]) -> Option<usize> {
        if objs.is_empty() {
            return None;
        }
        let (a, _) = self.hear_word(pcm, false);
        let (es, ec) = self.binding.expectations(&a);
        let ns = l2_norm(&es) + 1e-8;
        let nc = l2_norm(&ec) + 1e-8;
        let gs = ns / (ns + nc);
        let gc = nc / (ns + nc);

        let mut best_i = 0usize;
        let mut best_sc = f32::NEG_INFINITY;
        for (i, o) in objs.iter().enumerate() {
            let ds: Vec<f32> = o.s.iter().zip(self.binding.m_s.iter()).map(|(&x, &m)| x - m).collect();
            let dc: Vec<f32> = o.c.iter().zip(self.binding.m_c.iter()).map(|(&x, &m)| x - m).collect();
            let sc = gs * (dot(&es, &ds) / ns) + gc * (dot(&ec, &dc) / nc);
            if sc > best_sc {
                best_sc = sc;
                best_i = i;
            }
        }
        Some(best_i)
    }

    pub fn stream_preference(&mut self, pcm: &[f32]) -> f32 {
        let (a, _) = self.hear_word(pcm, false);
        let (es, ec) = self.binding.expectations(&a);
        let ns = l2_norm(&es);
        let nc = l2_norm(&ec);
        nc / (ns + nc + 1e-8)
    }

    pub fn babble_step(&mut self, eta: f32) {
        let mut m = [0.0f32; 8];
        for v in &mut m {
            *v = self.rng.next_f32();
        }
        let pcm = self.vocal.synthesize_pcm(&m);
        let wf = self.word_form(&pcm, false);
        self.vocal.learn_babble_pair(&wf, &m, eta);
    }

    pub fn imitate_audio(&mut self, target_pcm: &[f32]) -> ([f32; 8], Vec<f32>) {
        let wf = self.word_form(target_pcm, false);
        let m = self.vocal.infer_motor(&wf);
        let pcm_out = self.vocal.synthesize_pcm(&m);
        (m, pcm_out)
    }


    /// Autonomous offline sleep consolidation (SWR replay + Generator/Judge dream settling + metabolic pruning).
    /// Runs with ZERO external sensory input. Returns (pruned_synapses, discovery_archive_count).
    pub fn sleep_consolidate(&mut self, replay_cycles: usize, prune_thresh: f32) -> (usize, usize) {
        // 1. Restore cellular ATP and homeostatic energy during slow-wave sleep
        for a in &mut self.heatmap.atp {
            *a = (*a + 0.35).clamp(0.0, 1.0);
        }
        self.body.state[0] = (self.body.state[0] + 0.20).clamp(0.0, 1.0);

        // 2. Sharp-Wave Ripple (SWR) replay: reactivate discovered lexicon prototypes and sharpen cross-modal bindings
        let n_lex = self.binding.lexicon.len();
        if n_lex > 0 {
            for c in 0..replay_cycles {
                let idx = c % n_lex;
                let proto = self.binding.lexicon[idx].clone();
                let (mut es, mut ec) = self.binding.expectations(&proto);
                normalize_l2_in_place(&mut es);
                normalize_l2_in_place(&mut ec);
                let replay_eta = 0.015;
                for i in 0..self.genome.wa_units {
                    let pa = proto[i];
                    if pa <= 0.0 {
                        continue;
                    }
                    let row_s = &mut self.binding.bs[i * self.genome.it_units..(i + 1) * self.genome.it_units];
                    for (w, &s_val) in row_s.iter_mut().zip(es.iter()) {
                        if s_val > 0.05 {
                            *w += replay_eta * pa * s_val;
                        }
                    }
                    let row_c = &mut self.binding.bc[i * self.genome.c4_units..(i + 1) * self.genome.c4_units];
                    for (w, &c_val) in row_c.iter_mut().zip(ec.iter()) {
                        if c_val > 0.05 {
                            *w += replay_eta * pa * c_val;
                        }
                    }
                }
            }
        }

        // 3. Offline dream Generator + Judge on the biophysical cortical sheet
        let n_hm = self.heatmap.n;
        let base_cue: Vec<f32> = (0..n_hm)
            .map(|i| if self.heatmap.s[i] >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        for _ in 0..replay_cycles.min(8) {
            let _ = self.discovery.propose_and_settle(&self.heatmap, &base_cue, 0.25, -0.01, &mut self.rng);
        }

        // 4. Metabolic synaptic scaling & pruning of weak/noisy cross-modal weights
        let mut pruned = 0usize;
        for w in &mut self.binding.bs {
            if w.abs() > 0.0 && w.abs() < prune_thresh {
                *w = 0.0;
                pruned += 1;
            } else {
                *w *= 0.98;
            }
        }
        for w in &mut self.binding.bc {
            if w.abs() > 0.0 && w.abs() < prune_thresh {
                *w = 0.0;
                pruned += 1;
            } else {
                *w *= 0.98;
            }
        }
        (pruned, self.discovery.candidates.len())
    }

    pub fn step(&mut self, frame: &SensoryFrame) -> MotorFrame {
        self.step_count += 1;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let (objs, _) = if !frame.retina_rgb.is_empty() && w > 0 && h > 0 {
            self.perceive_objects(frame.retina_rgb, w, h, false)
        } else {
            (Vec::new(), Vec::new())
        };

        let mut motor = MotorFrame::default();
        if !frame.pcm_16khz.is_empty() {
            if let Some(choice) = self.attend_to_word(frame.pcm_16khz, &objs) {
                motor.fixation_blob = Some(choice);
                motor.gaze_xy = [
                    objs[choice].cx / (w.max(1) as f32),
                    objs[choice].cy / (h.max(1) as f32),
                ];
            }
            let (m, pcm_out) = self.imitate_audio(frame.pcm_16khz);
            motor.vocal_tract = m;
            motor.pcm_out = pcm_out;
        }
        let _ = self.body.step_homeostasis(&frame.body_flux, 0.04, self.wa.last_surprise);
        for d in 0..4 {
            motor.action_drive[d] = (self.body.set_point[d] - self.body.state[d]).max(0.0);
        }
        motor
    }
}
