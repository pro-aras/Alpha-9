//! Alpha 9 Biophysical, Feeling-Driven Emergent Organism (`alpha-organism`).
//! Law 2: Contains ZERO string literals, ZERO text comparisons, and ZERO world dependencies.

pub mod body;
pub mod cognition;
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
use cognition::{DeliberativeMind, ModulationBroadcast};
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
    pub mind: DeliberativeMind,
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
        let mind = DeliberativeMind::new(32, 6, 256);
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
            mind,
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
        self.mind.memory_bytes() + self.heatmap.memory_bytes()
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
        self.word_form_from_seg(&seg, learn_a1)
    }

    pub fn word_form_from_seg(&mut self, seg: &[Vec<f32>], learn_a1: bool) -> Vec<f32> {
        let r = self.a1_codes(seg, learn_a1);
        let n = r.len().max(1);
        let u = self.genome.a1_units;

        // Compute per-unit mean across the utterance & subtract common-mode activation cone
        let mut unit_mean = vec![0.0f32; u];
        for row in &r {
            for (m, &rv) in unit_mean.iter_mut().zip(row.iter()) {
                *m += rv;
            }
        }
        for m in &mut unit_mean {
            *m /= n as f32;
        }

        let mut v = vec![0.0f32; 3 * u];
        for (i, row) in r.iter().enumerate() {
            let third = ((i * 3) / n).min(2);
            let dst = &mut v[third * u..(third + 1) * u];
            for j in 0..u {
                dst[j] += (row[j] - 0.65 * unit_mean[j]).max(0.0);
            }
        }
        let v_mean: f32 = v.iter().sum::<f32>() / (v.len() as f32);
        for val in &mut v {
            *val = (*val - 0.5 * v_mean).max(0.0);
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
        let wf = self.word_form_from_seg(&seg, false);
        let _ = self.wa.fire_and_learn(&wf, None);
    }

    pub fn hear_word(&mut self, pcm: &[f32], learn: bool) -> (Vec<f32>, bool) {
        let v = self.word_form(pcm, learn);
        if learn {
            let a = self.wa.fire_and_learn(&v, None);
            let (_, is_novel) = self.binding.update_lexicon(&a, self.genome.wa_k as f32);
            (a, is_novel)
        } else {
            (self.wa.fire(&v), false)
        }
    }

    /// Process a continuous multi-word speech stream (phrase, sentence, or discourse) segmented
    /// via cochlear theta-envelope trough detection, returning the sequence of `(wa_assembly, is_novel)`.
    pub fn hear_stream(&mut self, pcm: &[f32], learn: bool) -> Vec<(Vec<f32>, bool)> {
        let segs = self.cochlea.transduce_stream(pcm);
        let mut out = Vec::with_capacity(segs.len());
        for seg in &segs {
            let v = self.word_form_from_seg(seg, learn);
            if learn {
                let a = self.wa.fire_and_learn(&v, None);
                let (_, is_novel) = self.binding.update_lexicon(&a, self.genome.wa_k as f32);
                out.push((a, is_novel));
            } else {
                out.push((self.wa.fire(&v), false));
            }
        }
        out
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
            let fs = self.v1_features(&patch, true);
            let fc = self.v4_features(&b.colour_opp, true);
            let _ = self.it.fire_and_learn(&fs, None);
            let _ = self.c4.fire_and_learn(&fc, None);
        }
    }

    pub fn perceive_objects(&mut self, rgb: &[u8], width: usize, height: usize, learn: bool) -> (Vec<PerceivedObject>, Vec<RetinalBlob>) {
        let (blobs, _) = Retina::segment(rgb, width, height);
        let fov = self.genome.fovea_dim;
        let mut objs = Vec::with_capacity(blobs.len());
        for b in &blobs {
            let fs = self.v1_features(&Retina::log_polar_form(b, fov, fov), learn);
            let fc = self.v4_features(&b.colour_opp, learn);
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
        let take = max_fixations.max(objs.len()).min(perm.len());
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
        let ns2 = ns * ns;
        let nc2 = nc * nc;
        let gs = ns2 / (ns2 + nc2 + 1e-8);
        let gc = nc2 / (ns2 + nc2 + 1e-8);

        let aud_q = self.mind.encode_auditory_state(&a);
        let q_read = self.mind.read_living_cells(
            None,
            Some(&aud_q),
            None,
            ModulationBroadcast::default(),
        );
        let has_cell_meaning = l2_norm(&q_read.recalled_meaning) > 0.10;

        let mut best_i = 0usize;
        let mut best_sc = f32::NEG_INFINITY;
        for (i, o) in objs.iter().enumerate() {
            let ds: Vec<f32> = o.s.iter().zip(self.binding.m_s.iter()).map(|(&x, &m)| x - m).collect();
            let dc: Vec<f32> = o.c.iter().zip(self.binding.m_c.iter()).map(|(&x, &m)| x - m).collect();
            let nds = l2_norm(&ds) + 1e-8;
            let ndc = l2_norm(&dc) + 1e-8;
            let cos_s = dot(&es, &ds) / (ns * nds);
            let cos_c = dot(&ec, &dc) / (nc * ndc);
            let mut sc = gs * cos_s + gc * cos_c;
            if has_cell_meaning {
                let obj_state = self.mind.encode_multimodal_state(&o.s, &o.c);
                let cell_m_cos = dot(&obj_state, &q_read.recalled_meaning);
                sc = 0.35 * sc + 0.65 * cell_m_cos;
            }
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


    /// Autonomous offline sleep consolidation (SWR replay + episodic-to-cortical replay + Generator/Judge dream settling + metabolic pruning).
    /// Runs with ZERO external sensory input. Returns (pruned_synapses, discovery_archive_count).
    pub fn sleep_consolidate(&mut self, replay_cycles: usize, prune_thresh: f32) -> (usize, usize) {
        // 1. Restore cellular ATP and homeostatic energy during slow-wave sleep
        for a in &mut self.heatmap.atp {
            *a = (*a + 0.35).clamp(0.0, 1.0);
        }
        for k in 0..4 {
            self.body.state[k] = (self.body.state[k] + 0.35).clamp(0.0, self.body.set_point[k]);
        }
        self.body.distress = HomeostaticBody::compute_distress_for(&self.body.state, &self.body.set_point);
        self.body.distress_prev = self.body.distress;

        // 2. Consolidate continuous eligibility trace into the recurrent experience chain fabric
        //    and run offline Dreaming across the Living Epigenetic Cell population
        self.mind.consolidate_chain(0.08);
        let _ = self.mind.dream_living_cells(replay_cycles.max(4));

        // 3. Sharp-Wave Ripple (SWR) replay: reactivate discovered lexicon prototypes and sharpen cross-modal bindings
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
                    if pa <= 0.05 {
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

        // 4. Offline dream Generator + Judge on the biophysical cortical sheet
        let n_hm = self.heatmap.n;
        let base_cue: Vec<f32> = (0..n_hm)
            .map(|i| if self.heatmap.s[i] >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        for _ in 0..replay_cycles.min(8) {
            let _ = self.discovery.propose_and_settle(&self.heatmap, &base_cue, 0.25, -0.01, &mut self.rng);
        }

        // 5. Metabolic synaptic scaling & pruning of weak/noisy cross-modal weights
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


    /// Convert any auditory word assembly `a_word` (`WA` cortical state) into an 8D articulatory
    /// motor trajectory via `WA` basal inversion + `Area Id4` (`self.vocal.infer_motor`).
    pub fn assembly_to_vocal_motor(&self, a_word: &[f32]) -> [f32; 8] {
        let in_dim = self.wa.n_in;
        let mut wf_est = vec![0.0f32; in_dim];
        for u in 0..self.wa.n_units.min(a_word.len()) {
            let act = a_word[u];
            if act <= 0.0 {
                continue;
            }
            let row = &self.wa.w_basal[u * in_dim..(u + 1) * in_dim];
            for (dst, &w) in wf_est.iter_mut().zip(row.iter()) {
                *dst += act * w;
            }
        }
        normalize_l2_in_place(&mut wf_est);
        self.vocal.infer_motor(&wf_est)
    }

    /// [S4] Grounded Spoken Naming: Given a perceived visual object, recall its bound word-form
    /// assembly (`stream_mode`: 0=colour, 1=shape, 2=joint), invert to articulatory motor commands,
    /// and synthesize a real 16 kHz spoken PCM waveform.
    pub fn speak_object_attribute(&self, obj: &PerceivedObject, stream_mode: u8) -> (Vec<f32>, [f32; 8], Vec<f32>) {
        let a_word = self.binding.recall_word_assembly(&obj.s, &obj.c, stream_mode);
        let motor = self.assembly_to_vocal_motor(&a_word);
        let pcm = self.vocal.synthesize_pcm(&motor);
        (a_word, motor, pcm)
    }

    /// [S5] Two-Word Compositional Speech: Look at a visual object and sequentially articulate
    /// `[colour_word_pcm, shape_word_pcm]` as a connected two-word 16 kHz audio utterance.
    pub fn speak_two_word_phrase(&self, obj: &PerceivedObject) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (a_col, _, pcm_col) = self.speak_object_attribute(obj, 0);
        let (a_shp, _, pcm_shp) = self.speak_object_attribute(obj, 1);
        let mut phrase_pcm = Vec::with_capacity(pcm_col.len() + 800 + pcm_shp.len());
        phrase_pcm.extend_from_slice(&pcm_col);
        phrase_pcm.resize(pcm_col.len() + 800, 0.0);
        phrase_pcm.extend_from_slice(&pcm_shp);
        (a_col, a_shp, phrase_pcm)
    }

    /// Continuous, open-ended developmental life cycle step:
    /// The ONLY interface between the organism and reality (`SensoryFrame -> MotorFrame`).
    /// Operates as a single coupled dynamical flow without discrete mode switches:
    /// 1. Visual, auditory, and bodily currents form a unified attention field over the visual room
    ///    (combining acoustic expectation resonance, ongoing experience-chain flow, and curiosity).
    /// 2. Vocal tract articulation emerges from the attended experience blended with the organism's
    ///    personal biographical trace (`chain_deep`) and insular feeling state, self-calibrating via
    ///    `Area Id4` auditory-motor reafference.
    /// 3. The unbroken Chain of Experience (`chain_fast -> chain_slow -> chain_deep`) advances and
    ///    sculpts the recurrent causal fabric (`w_flow`) via continuous eligibility traces.
    pub fn step(&mut self, frame: &SensoryFrame) -> MotorFrame {
        self.step_count += 1;
        let w = frame.width as usize;
        let h = frame.height as usize;
        let mut motor = MotorFrame::default();

        let (objs_now, _) = if !frame.retina_rgb.is_empty() && w > 0 && h > 0 {
            self.perceive_objects(frame.retina_rgb, w, h, false)
        } else {
            (Vec::new(), Vec::new())
        };

        // Compute internal homeostatic drive vector (`drive_vec_opt`) when bodily needs are active
        let mut max_deficit = 0.0f32;
        let mut deficits = [0.0f32; 4];
        for k in 0..4 {
            deficits[k] = (self.body.set_point[k] - self.body.state[k]).max(0.0);
            if frame.body_flux[k] < -0.05 {
                deficits[k] = (deficits[k] - frame.body_flux[k]).min(1.0);
            }
            if deficits[k] > max_deficit {
                max_deficit = deficits[k];
            }
        }
        let drive_vec_opt: Option<Vec<f32>> = if max_deficit > 0.16 {
            let mut dv = vec![0.0f32; 32];
            for d in 0..32 {
                let sign = if (d / 4) % 2 == 0 { 1.0 } else { -1.0 };
                dv[d] = deficits[d % 4] * sign * (1.0 + 0.25 * ((d % 7) as f32));
            }
            normalize_l2_in_place(&mut dv);
            Some(dv)
        } else {
            None
        };

        // Compute continuous auditory expectation field (naturally zero when room is silent)
        // plus Channel-2 Living Cell Organ word->meaning lookup when a word is heard
        let (es, ec, aud_gain, cell_word_meaning) = if !frame.pcm_16khz.is_empty() {
            let (a_now, _) = self.hear_word(frame.pcm_16khz, false);
            let (es_v, ec_v) = self.binding.expectations(&a_now);
            let ns2 = l2_norm(&es_v).powi(2);
            let nc2 = l2_norm(&ec_v).powi(2);
            let g = (ns2 + nc2) / (ns2 + nc2 + 1e-4);
            let aud_q = self.mind.encode_auditory_state(&a_now);
            let q_read = self.mind.read_living_cells(
                None,
                Some(&aud_q),
                None,
                ModulationBroadcast::default(),
            );
            (es_v, ec_v, g, q_read.recalled_meaning)
        } else {
            (
                vec![0.0; self.genome.it_units],
                vec![0.0; self.genome.c4_units],
                0.0,
                vec![0.0; 32],
            )
        };

        // Unified continuous attention pull across all objects in the room
        let mut attended_idx = 0usize;
        let mut attended_recalled_feeling = 0.0f32;
        if !objs_now.is_empty() {
            let ns = l2_norm(&es) + 1e-8;
            let nc = l2_norm(&ec) + 1e-8;
            let gs = (ns * ns) / (ns * ns + nc * nc + 1e-8);
            let gc = (nc * nc) / (ns * ns + nc * nc + 1e-8);
            let has_cell_meaning = l2_norm(&cell_word_meaning) > 0.10;
            let mut best_pull = f32::NEG_INFINITY;

            for (i, o) in objs_now.iter().enumerate() {
                let ds: Vec<f32> = o.s.iter().zip(self.binding.m_s.iter()).map(|(&x, &m)| x - m).collect();
                let dc: Vec<f32> = o.c.iter().zip(self.binding.m_c.iter()).map(|(&x, &m)| x - m).collect();
                let cos_s = dot(&es, &ds) / (ns * (l2_norm(&ds) + 1e-8));
                let cos_c = dot(&ec, &dc) / (nc * (l2_norm(&dc) + 1e-8));
                let mut aud_pull = gs * cos_s + gc * cos_c;

                let obj_state = self.mind.encode_multimodal_state(&o.s, &o.c);
                if has_cell_meaning {
                    let cell_m_cos = dot(&obj_state, &cell_word_meaning);
                    aud_pull = 0.65 * aud_pull + 0.35 * cell_m_cos;
                }
                let flow_res = dot(&obj_state, &self.mind.imagined_flow);
                let deep_res = dot(&obj_state, &self.mind.chain_deep);
                let (_, cell_feeling) = self
                    .mind
                    .extract_cellular_experience(&obj_state, Some(&self.mind.prev_cell_state));

                // Saccadic Inhibition of Return (IoR) across 4-step Superior Colliculus trail
                // (active when remaining in a familiar/static visual scene so the infant explores all objects!)
                let ox = o.cx / (w.max(1) as f32);
                let oy = o.cy / (h.max(1) as f32);
                let scene_persistence = ((self.mind.comparator.last_similarity - 0.45) * 2.0).clamp(0.0, 1.2);
                let mut spatial_ior = 0.0f32;
                for &[tx, ty, strg] in &self.mind.gaze_trail {
                    let d2 = (ox - tx).powi(2) + (oy - ty).powi(2);
                    spatial_ior += strg * (-d2 / 0.020).exp();
                }
                spatial_ior *= scene_persistence;
                let hab_fatigue = dot(&obj_state, &self.mind.habituation_trace).max(0.0) * scene_persistence;
                let ion_explore = 0.07
                    * self.mind.next_ion_fluctuation(ox + oy + self.mind.chain_deep[0] + (i as f32));

                let fam_norm = (self.binding.object_familiarity(o)
                    / (l2_norm(&ds).max(l2_norm(&dc)) + 1e-8))
                    .clamp(0.0, 1.0);
                let salience = (l2_norm(&o.c) + l2_norm(&o.s)) * 0.08;
                let spontaneous_pull = 0.48 * flow_res
                    + 0.35 * deep_res
                    + 0.35 * cell_feeling
                    + 0.10 * (1.0 - fam_norm)
                    + 0.05 * salience
                    - 0.48 * spatial_ior
                    - 0.28 * hab_fatigue
                    + ion_explore;

                let total_pull = aud_gain * aud_pull + (1.0 - aud_gain) * spontaneous_pull;
                if total_pull > best_pull {
                    best_pull = total_pull;
                    attended_idx = i;
                    attended_recalled_feeling = cell_feeling;
                }
            }
            motor.fixation_blob = Some(attended_idx);
            motor.gaze_xy = [
                objs_now[attended_idx].cx / (w.max(1) as f32),
                objs_now[attended_idx].cy / (h.max(1) as f32),
            ];
            for k in (1..4).rev() {
                self.mind.gaze_trail[k] = [
                    self.mind.gaze_trail[k - 1][0],
                    self.mind.gaze_trail[k - 1][1],
                    self.mind.gaze_trail[k - 1][2] * 0.72,
                ];
            }
            self.mind.gaze_trail[0] = [motor.gaze_xy[0], motor.gaze_xy[1], 1.0];
        }

        // Endogenous cortical development & cross-modal plasticity from current experience
        let (objs_learned, _) = if !frame.retina_rgb.is_empty() && w > 0 && h > 0 {
            self.perceive_objects(frame.retina_rgb, w, h, true)
        } else {
            (Vec::new(), Vec::new())
        };

        let mut heard_assembly: Vec<f32> = Vec::new();
        let mut stream_heard: Vec<(Vec<f32>, bool)> = Vec::new();
        if !frame.pcm_16khz.is_empty() {
            if frame.pcm_16khz.len() >= 11000 {
                stream_heard = self.hear_stream(frame.pcm_16khz, true);
            }
            if stream_heard.len() >= 2 {
                let last_w = stream_heard.last().map(|x| x.0.clone()).unwrap_or_default();
                if !objs_learned.is_empty() && !last_w.is_empty() {
                    let perm = self.rng.shuffle_indices(objs_learned.len());
                    let eff_eta = self.body.emotional_lr(self.genome.eta_bind);
                    self.binding.bind_experience(&last_w, &objs_learned, &perm, eff_eta, false);
                }
                heard_assembly = last_w;
            } else {
                let (a_heard, is_novel) = self.hear_word(frame.pcm_16khz, true);
                if !objs_learned.is_empty() {
                    let perm = self.rng.shuffle_indices(objs_learned.len());
                    let eff_eta = self.body.emotional_lr(self.genome.eta_bind);
                    self.binding.bind_experience(&a_heard, &objs_learned, &perm, eff_eta, is_novel);
                }
                heard_assembly = a_heard;
            }
        }

        let attended_vis_state: Option<Vec<f32>> = objs_now
            .get(attended_idx)
            .or_else(|| objs_now.first())
            .map(|o| self.mind.encode_multimodal_state(&o.s, &o.c));
        let single_aud_vec: Option<Vec<f32>> = if !heard_assembly.is_empty() {
            Some(self.mind.encode_auditory_state(&heard_assembly))
        } else {
            None
        };

        let mut cortical_summary = vec![0.0f32; 32];
        if !objs_learned.is_empty() {
            let primary_st = self.mind.encode_multimodal_state(&objs_learned[0].s, &objs_learned[0].c);
            let mut scene_gist = vec![0.0f32; 32];
            for o in &objs_learned {
                let st = self.mind.encode_multimodal_state(&o.s, &o.c);
                for d in 0..32 {
                    scene_gist[d] += st[d];
                }
            }
            normalize_l2_in_place(&mut scene_gist[0..16]);
            normalize_l2_in_place(&mut scene_gist[16..32]);
            for d in 0..32 {
                cortical_summary[d] = 0.76 * primary_st[d] + 0.24 * scene_gist[d];
            }
            normalize_l2_in_place(&mut cortical_summary[0..16]);
            normalize_l2_in_place(&mut cortical_summary[16..32]);
            normalize_l2_in_place(&mut cortical_summary);
            if !heard_assembly.is_empty() {
                let aud_st = self.mind.encode_auditory_state(&heard_assembly);
                for d in 0..32 {
                    cortical_summary[d] = 0.72 * cortical_summary[d] + 0.28 * aud_st[d];
                }
                normalize_l2_in_place(&mut cortical_summary);
            }
        } else if !heard_assembly.is_empty() {
            // Pure auditory conversation / spoken prompt in the absence of visual objects
            cortical_summary = self.mind.encode_auditory_state(&heard_assembly);
        } else if objs_now.is_empty() && frame.pcm_16khz.is_empty() {
            // In a dark, silent room, spontaneous cellular engram replay (dreaming/thinking) ignites!
            cortical_summary = self.mind.spontaneous_replay_step(0.08);
        }

        // Whole-body visceral organ cells buffer recurring negative metabolic shocks (allostasis)
        let mut buffered_flux = frame.body_flux;
        for d in 0..4 {
            if buffered_flux[d] < 0.0 {
                let organ_shield = 0.45 * self.mind.organ_cells[d];
                buffered_flux[d] *= (1.0 - organ_shield).clamp(0.25, 1.0);
            }
        }

        // Step cortical attractor heatmap & Quadripartite Insula homeostasis
        let spikes = self.heatmap.step_biophysics(&cortical_summary, 0.03, buffered_flux[0].max(0.001), &mut self.rng);
        let spike_frac = (spikes as f32) / (self.heatmap.n.max(1) as f32);
        let top_down_reappraisal = (self.mind.comparator.last_similarity * 0.5
            + 0.35 * attended_recalled_feeling)
            .clamp(-1.0, 1.0);
        let (valence, arousal) = self.body.step_homeostasis_with_cognition(
            &buffered_flux,
            spike_frac,
            self.wa.last_surprise,
            top_down_reappraisal,
        );

        // Advance the unbroken Chain of Experience + 256 Epigenetic Cells
        let flow = self.mind.advance_experience_chain(&cortical_summary, valence, arousal, 0.14).to_vec();

        // Channel 2 + 3 + 4: Read from the Living Cell Population Organ & check if Waking Brain is needed
        let aud_surprise_now = if !frame.pcm_16khz.is_empty() {
            self.wa.last_surprise
        } else {
            0.0
        };
        let vis_surprise_now = if !objs_now.is_empty() {
            (1.0 - self.mind.comparator.last_similarity.clamp(0.0, 1.0)) * 0.5
        } else {
            0.0
        };
        let broadcast = ModulationBroadcast {
            surprise: vis_surprise_now.max(aud_surprise_now),
            reward: valence.max(0.0),
            danger: (-valence).max(0.0),
            rest: if objs_now.is_empty() && frame.pcm_16khz.is_empty() {
                1.0
            } else {
                (self.body.extracellular_adenosine_um * 0.35).clamp(0.0, 0.75)
            },
        };
        let cell_readout = self.mind.read_living_cells(
            attended_vis_state.as_deref(),
            single_aud_vec.as_deref(),
            drive_vec_opt.as_deref(),
            broadcast,
        );
        // When the Living Cells handle a familiar input cheaply (`!brain_woke_up`), refund cortical ATP
        if !cell_readout.report.brain_woke_up {
            for a in &mut self.heatmap.atp {
                *a = (*a + 0.0015).clamp(0.0, 1.0);
            }
        }

        // Subvocal motor-auditory calibration + natural vocalization shaped by Living Epigenetic Cells,
        // object perception, conversational speech input, similarity/difference subspace, and muscle cells
        self.babble_step(0.08);
        if let Some(att_obj) = objs_now.get(attended_idx) {
            let shp_diff = l2_norm(&self.mind.last_difference_vec[0..16]);
            let col_diff = l2_norm(&self.mind.last_difference_vec[16..32]);
            let attr_kind = if shp_diff > col_diff * 1.15 { 1 } else { 0 };
            let (_, mut m_grounded, _) = self.speak_object_attribute(att_obj, attr_kind);

            if frame.pcm_16khz.is_empty()
                && drive_vec_opt.is_some()
                && self.body.state[0] < 0.40
                && cell_readout.second_word_vocal.is_some()
            {
                // Stage 4 Emergent Two-Word Combination (e.g. Drive Word -> Object Word):
                // Two word cells lit up simultaneously and were ordered by directed `time_bonds` + `fullness`!
                let m1 = cell_readout.recalled_vocal;
                let m2_cell = cell_readout.second_word_vocal.unwrap_or(m1);
                let mut m2 = [0.0f32; 8];
                for d in 0..8 {
                    m2[d] = (0.88 * m2_cell[d] + 0.12 * m_grounded[d]).clamp(0.0, 1.0);
                }
                let pcm1 = self.vocal.synthesize_pcm(&m1);
                let pcm2 = self.vocal.synthesize_pcm(&m2);
                let mut phrase_pcm = Vec::with_capacity(pcm1.len() + 800 + pcm2.len());
                phrase_pcm.extend_from_slice(&pcm1);
                phrase_pcm.resize(pcm1.len() + 800, 0.0);
                phrase_pcm.extend_from_slice(&pcm2);
                motor.vocal_tract = m1;
                motor.pcm_out = phrase_pcm;
            } else {
                let has_cell_word = frame.pcm_16khz.is_empty() && l2_norm(&cell_readout.recalled_word_pat) > 0.15;

                for d in 0..8 {
                    let mod_scale = if has_cell_word { 0.15 } else { 1.0 };
                    let base_m = if has_cell_word {
                        0.88 * cell_readout.recalled_vocal[d] + 0.12 * m_grounded[d]
                    } else {
                        m_grounded[d]
                    };
                    let chain_mod = (0.14 * self.mind.chain_deep.get(d * 2).copied().unwrap_or(0.0)
                        + 0.10 * flow.get(d * 3).copied().unwrap_or(0.0))
                        * mod_scale;
                    let feeling_mod = (if d % 2 == 0 {
                        0.10 * (valence + 0.5 * self.mind.last_recalled_feeling)
                    } else {
                        0.10 * (arousal - 0.2)
                    }) * mod_scale;
                    let muscle_tone = 0.06 * (self.mind.muscle_cells[d] - 0.5) * mod_scale;
                    let vocal_explore = 0.03
                        * self.mind.novelty_emergence.clamp(0.1, 1.5)
                        * self.mind.next_ion_fluctuation((d as f32) + valence)
                        * mod_scale;
                    m_grounded[d] =
                        (base_m + chain_mod + feeling_mod + muscle_tone + vocal_explore).clamp(0.0, 1.0);
                }
                let pcm_out = self.vocal.synthesize_pcm(&m_grounded);
                motor.vocal_tract = m_grounded;
                motor.pcm_out = pcm_out;
            }
        } else if !heard_assembly.is_empty() {
            // Conversational vocal response when spoken to without visual objects:
            // Combines auditory-motor echo (`Area Id4`) with recalled cellular vocal synergy (`last_recalled_vocal`)
            let m_echo = self.assembly_to_vocal_motor(&heard_assembly);
            let mut m_resp = [0.0f32; 8];
            for d in 0..8 {
                let flow_mod = 0.08 * flow.get(d * 3).copied().unwrap_or(0.0);
                m_resp[d] = (0.68 * m_echo[d]
                    + 0.22 * self.mind.last_recalled_vocal[d]
                    + 0.10 * self.mind.muscle_cells[d]
                    + flow_mod)
                    .clamp(0.0, 1.0);
            }
            let pcm_out = self.vocal.synthesize_pcm(&m_resp);
            motor.vocal_tract = m_resp;
            motor.pcm_out = pcm_out;
        } else {
            // Subvocal murmur / spontaneous vocalization during quiet rest or after a conversational prompt
            let mut m_dream = self.mind.last_recalled_vocal;
            for d in 0..8 {
                m_dream[d] = (0.60 * m_dream[d]
                    + 0.30 * self.mind.muscle_cells[d]
                    + 0.10 * flow.get(d * 4).copied().unwrap_or(0.0)
                    + 0.03 * self.mind.next_ion_fluctuation(d as f32))
                .clamp(0.0, 1.0);
            }
            motor.vocal_tract = m_dream;
            motor.pcm_out = self.vocal.synthesize_pcm(&m_dream);
        }

        // Continuous motor action drive emerging from:
        // 1. Multi-horizon D1/D2 recalled cellular action synergy (`last_recalled_action`, for puzzles & tools)
        // 2. Peripheral limb proprioceptive cell baseline (`limb_cells`)
        // 3. Homeostatic need + cellular feeling (`approach` vs `withdraw/avoid`) + exploratory ion noise
        let recall_feel = if self.mind.last_recalled_feeling.abs() >= attended_recalled_feeling.abs() {
            self.mind.last_recalled_feeling
        } else {
            attended_recalled_feeling
        };
        let act_max = self
            .mind
            .last_recalled_action
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let act_min = self
            .mind
            .last_recalled_action
            .iter()
            .copied()
            .fold(1.0f32, f32::min);
        let unspecialized = (1.0 - (act_max - act_min) * 1.8).clamp(0.0, 1.0);
        for d in 0..4 {
            let homeo_need = (self.body.set_point[d] - self.body.state[d]).max(0.0) * 0.20 * unspecialized;
            let wave_drive = (flow[d * 4] + flow[16 + d * 4]).max(0.0) * 0.08 * unspecialized;
            let learned_act = 0.88 * self.mind.last_recalled_action[d]
                + 0.12 * self.mind.limb_cells[d] * unspecialized;
            let cell_affect = if d == 0 {
                recall_feel.max(0.0) * 0.35 * unspecialized
            } else if d == 1 {
                let burn_reflex = if recall_feel < -0.45 { 1.0 } else { unspecialized };
                (recall_feel + 0.30).min(0.0).abs() * 0.95 * burn_reflex
            } else {
                0.0
            };
            let act_explore = 0.02 * self.mind.next_ion_fluctuation((d as f32) + arousal) * unspecialized;
            motor.action_drive[d] = (homeo_need + wave_drive + learned_act + cell_affect + act_explore).clamp(0.0, 2.0);
        }

        // Step all 5 peripheral somatic cellular tissues (`retina`, `cochlea`, `limb`, `organ`, `muscle`)
        self.mind.step_whole_body_cells(
            &cortical_summary,
            &heard_assembly,
            valence,
            arousal,
            &motor.action_drive,
            &motor.vocal_tract,
        );

        // Channel 1 + 3: Write experience & broadcast chemical signals into the Living Epigenetic Cell population
        if stream_heard.len() >= 2 {
            let aud_0 = self.mind.encode_auditory_state(&stream_heard[0].0);
            let m_echo_0 = self.assembly_to_vocal_motor(&stream_heard[0].0);
            let _ = self.mind.step_living_cells(
                None,
                Some(&aud_0),
                drive_vec_opt.as_deref(),
                &motor.action_drive,
                &m_echo_0,
                valence,
                broadcast,
            );
            let aud_1 = self.mind.encode_auditory_state(&stream_heard[1].0);
            let m_echo_1 = self.assembly_to_vocal_motor(&stream_heard[1].0);
            let _ = self.mind.step_living_cells(
                attended_vis_state.as_deref(),
                Some(&aud_1),
                None,
                &motor.action_drive,
                &m_echo_1,
                valence,
                broadcast,
            );
        } else {
            let vocal_for_cell = if !heard_assembly.is_empty() {
                self.assembly_to_vocal_motor(&heard_assembly)
            } else {
                motor.vocal_tract
            };
            let _ = self.mind.step_living_cells(
                attended_vis_state.as_deref(),
                single_aud_vec.as_deref(),
                drive_vec_opt.as_deref(),
                &motor.action_drive,
                &vocal_for_cell,
                valence,
                broadcast,
            );
        }

        if self.body.state[0] < 0.22 || self.body.extracellular_adenosine_um > 1.85 {
            let _ = self.sleep_consolidate(16, 0.005);
            self.body.extracellular_adenosine_um *= 0.65;
        }

        motor
    }
}
