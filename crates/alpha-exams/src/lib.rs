//! Law 3: Pre-Registered Held-Out Exams & Biophysical/Feeling Verification Gates (`alpha-exams`).

use std::fs;
use std::path::Path;
use alpha_organism::body::HomeostaticBody;
use alpha_organism::curiosity::StreamCuriosityTracker;
use alpha_organism::genome::Genome;
use alpha_organism::util::{dot, l2_norm, normalize_l2_in_place, Rng64};
use alpha_organism::Organism;
use alpha_world::{render_scene, synthesize_word_utterance, WorldObject, NUM_COLOURS, NUM_SHAPES, NUM_WORDS};

#[derive(Clone, Debug)]
pub struct GateReport {
    pub g0_anti_cheat_zero_strings: bool,
    pub g0_files_scanned: usize,
    pub b1_corrupted_cue_recovery: f32,
    pub b1_bound_word_recall: f32,
    pub b1_atp_energy_constrained: bool,
    pub b1_passed: bool,
    pub b2_curiosity_learnable_prog: f32,
    pub b2_curiosity_noisy_tv_prog: f32,
    pub b2_curiosity_static_prog: f32,
    pub b2_passed: bool,
    pub b3_positive_valence_on_relief: f32,
    pub b3_negative_valence_on_harm: f32,
    pub b3_emotional_memory_gain_ratio: f32,
    pub b3_passed: bool,
    pub b4_feeling_speech_need_acc: f32,
    pub b4_passed: bool,
    pub b5_posterior_pain_specificity: f32,
    pub b5_top_down_relief_delta: f32,
    pub b5_polar_action_mode_invariance: f32,
    pub b5_passed: bool,
    pub s0_v1_selectivity: f32,
    pub s0_a1_abx_acc: f32,
    pub s0_passed: bool,
    pub s1_saffran_word_score: f32,
    pub s1_saffran_partword_score: f32,
    pub s1_passed: bool,
    pub s2_c1_heldout_spk_acc: f32,
    pub s2_c1_chance: f32,
    pub s2_c2_heldout_voice_acc: f32,
    pub s2_c3_novel_combos_acc: f32,
    pub s2_c4_lesion_acc: f32,
    pub s2_c5_shuffled_acc: f32,
    pub s2_c6_colour_stream_pref: f32,
    pub s2_c6_shape_stream_pref: f32,
    pub s2_c7_lexicon_purity: f32,
    pub s2_passed: bool,
    pub s3_babble_imitation_cos: f32,
    pub s3_babble_null_cos: f32,
    pub s3_passed: bool,
    pub l1_acc_t0_naive: f32,
    pub l1_acc_t1_early: f32,
    pub l1_acc_t2_late: f32,
    pub l1_acc_t3_post_sleep: f32,
    pub l1_pruned_synapses: usize,
    pub l1_passed: bool,
    pub organism_ram_mb: f32,
    pub ram_budget_ok: bool,
}

pub fn run_g0_anti_cheat(organism_src_dir: &Path) -> (bool, usize) {
    let mut files = 0usize;
    let Ok(entries) = fs::read_dir(organism_src_dir) else {
        return (false, 0);
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        files += 1;
        let Ok(src) = fs::read_to_string(&p) else {
            return (false, files);
        };
        for line in src.lines() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains('"')
                || code.contains(".contains(")
                || code.contains(".split(")
                || code.contains("HashMap")
                || code.contains("BTreeMap<String")
            {
                return (false, files);
            }
        }
    }
    (files >= 9, files)
}

pub fn run_b1_biophysical_heatmap(org: &mut Organism) -> (f32, f32, bool, bool) {
    let mut rng = Rng64::new(101);
    let n = org.heatmap.n;

    let high_input = vec![2.0f32; n];
    let spikes_first = org.heatmap.step_biophysics(&high_input, 0.05, 0.0, &mut rng);
    let mut spikes_exhausted = 0usize;
    for _ in 0..30 {
        spikes_exhausted = org.heatmap.step_biophysics(&high_input, 0.05, 0.0, &mut rng);
    }
    let atp_constrained = spikes_first > n / 2 && spikes_exhausted < spikes_first / 3;

    let mut seen_patterns = Vec::new();
    let mut heard_patterns = Vec::new();
    for _ in 0..5 {
        let seen: Vec<f32> = (0..n).map(|_| if rng.next_f32() < 0.5 { 1.0 } else { -1.0 }).collect();
        let heard: Vec<f32> = (0..n).map(|_| if rng.next_f32() < 0.5 { 1.0 } else { -1.0 }).collect();
        org.heatmap.learn_pattern(&seen, 1.0);
        org.heatmap.bind_patterns(&seen, &heard, 1.0);
        seen_patterns.push(seen);
        heard_patterns.push(heard);
    }

    let target_seen = &seen_patterns[0];
    let target_heard = &heard_patterns[0];
    let noisy: Vec<f32> = target_seen
        .iter()
        .map(|&v| if rng.next_f32() < 0.30 { -v } else { v })
        .collect();
    let recovered = org.heatmap.settle(&noisy, 12, 0.05, &mut rng);
    let recalled_word = org.heatmap.follow(&recovered);

    let rec_acc = recovered
        .iter()
        .zip(target_seen.iter())
        .filter(|(&a, &b)| (a - b).abs() < 1e-4)
        .count() as f32
        / (n as f32);
    let word_acc = recalled_word
        .iter()
        .zip(target_heard.iter())
        .filter(|(&a, &b)| (a - b).abs() < 1e-4)
        .count() as f32
        / (n as f32);

    let _ = org.discovery.propose_and_settle(&org.heatmap, &noisy, 0.4, -0.05, &mut rng);

    let passed = atp_constrained && rec_acc > 0.92 && word_acc > 0.92;
    (rec_acc, word_acc, atp_constrained, passed)
}

pub fn run_b2_curiosity_learning_progress() -> (f32, f32, f32, bool) {
    let dim = 16usize;
    let mut rng = Rng64::new(202);
    let mut tr_static = StreamCuriosityTracker::new(dim);
    let mut tr_noisy_tv = StreamCuriosityTracker::new(dim);
    let mut tr_learnable = StreamCuriosityTracker::new(dim);

    let static_pat = vec![0.5f32; dim];
    let mut cycle = vec![vec![0.0f32; dim]; 4];
    for c in 0..4 {
        for i in 0..dim {
            cycle[c][i] = if (i + c) % 4 == 0 { 0.85 } else { -0.35 };
        }
    }

    for step in 0..28 {
        let _ = tr_static.observe_and_learn(&static_pat, 0.04);
        let noise_vec: Vec<f32> = (0..dim).map(|_| rng.standard_normal().clamp(-1.0, 1.0)).collect();
        let _ = tr_noisy_tv.observe_and_learn(&noise_vec, 0.04);
        let _ = tr_learnable.observe_and_learn(&cycle[step % 4], 0.04);
    }

    let p_learn = tr_learnable.learning_progress;
    let p_noise = tr_noisy_tv.learning_progress;
    let p_stat = tr_static.learning_progress;
    let passed = p_learn > 0.05 && p_learn > p_noise + 0.05 && p_learn > p_stat + 0.05;
    (p_learn, p_noise, p_stat, passed)
}

pub fn run_b3_homeostatic_feelings(org: &mut Organism) -> (f32, f32, f32, bool) {
    org.body = HomeostaticBody::default();
    let (val_relief, ar_relief) = org.body.step_homeostasis(&[0.15, 0.05, 0.20, 0.12], 0.02, 0.3);
    let lr_emotional = org.body.emotional_lr(0.01);

    let (_, _) = org.body.step_homeostasis(&[0.004, 0.002, 0.003, 0.002], 0.0, 0.0);
    let lr_neutral = org.body.emotional_lr(0.01);

    let (val_harm, _) = org.body.step_homeostasis(&[-0.18, -0.15, -0.20, -0.10], 0.25, 0.4);

    let gain_ratio = lr_emotional / lr_neutral.max(1e-6);
    let passed = val_relief > 0.05 && val_harm < -0.05 && ar_relief > 0.05 && gain_ratio > 1.25;
    (val_relief, val_harm, gain_ratio, passed)
}

pub fn run_b4_feeling_driven_speech(org: &mut Organism) -> (f32, bool) {
    let mut rng = Rng64::new(404);
    let dim = org.speech_engine.pat_dim;

    let mut repertoire = Vec::with_capacity(4);
    for k in 0..4 {
        let mut p = vec![0.0f32; dim];
        for i in 0..dim {
            p[i] = if i % 4 == k { 1.0 } else { -0.25 };
        }
        normalize_l2_in_place(&mut p);
        repertoire.push(p);
    }

    let context = vec![0.25f32; dim];

    for ep in 0..160 {
        let need_idx = ep % 4;
        let mut body_before = org.body.set_point;
        body_before[need_idx] = 0.20;
        org.body.state = body_before;
        org.body.distress = HomeostaticBody::compute_distress_for(&body_before, &org.body.set_point);
        org.body.distress_prev = org.body.distress;

        let (chosen_idx, utter_pat, _) = org.speech_engine.choose_utterance(
            &org.body,
            &context,
            &repertoire,
            8,
            0.35,
            &mut rng,
        );

        let mut effect_dim = chosen_idx % 4;
        let mut best_cos = f32::NEG_INFINITY;
        for (k, proto) in repertoire.iter().enumerate() {
            let c = dot(&utter_pat, proto);
            if c > best_cos {
                best_cos = c;
                effect_dim = k;
            }
        }
        let mut body_after = body_before;
        body_after[effect_dim] = (body_after[effect_dim] + 0.30).clamp(0.0, 1.2);
        let d_after = HomeostaticBody::compute_distress_for(&body_after, &org.body.set_point);
        let valence = org.body.distress - d_after;
        let arousal = valence.abs();

        org.speech_engine.learn_from_outcome(
            &body_before,
            &body_after,
            &org.body.set_point,
            &context,
            &utter_pat,
            valence,
            arousal,
            0.05,
        );
    }

    let mut hits = 0usize;
    let trials = 60usize;
    for t in 0..trials {
        let need_idx = t % 4;
        let mut body_state = org.body.set_point;
        body_state[need_idx] = 0.15;
        org.body.state = body_state;
        org.body.distress = HomeostaticBody::compute_distress_for(&body_state, &org.body.set_point);

        let (_, utter_pat, _) = org.speech_engine.choose_utterance(
            &org.body,
            &context,
            &repertoire,
            6,
            0.05,
            &mut rng,
        );
        let mut best_k = 0usize;
        let mut best_c = f32::NEG_INFINITY;
        for (k, proto) in repertoire.iter().enumerate() {
            let c = dot(&utter_pat, proto);
            if c > best_c {
                best_c = c;
                best_k = k;
            }
        }
        if best_k == need_idx {
            hits += 1;
        }
    }
    let acc = (hits as f32) / (trials as f32);
    (acc, acc > 0.80)
}


pub fn run_b5_quadripartite_insula(org: &mut Organism) -> (f32, f32, f32, bool) {
    // 1. Posterior Region I (pain/temperature) vs Region III (visceral/anxiety) specificity
    org.body = HomeostaticBody::default();
    let _ = org.body.insula.step_insula([0.9, 0.8, 0.7], [0.1, 0.0, 0.0], [0.05, 0.05], 0.1, 0.0);
    let r1_on_pain = org.body.insula.region1_pain_temp[0];
    let _ = org.body.insula.step_insula([0.05, 0.05, 0.0], [0.1, 0.0, 0.0], [0.9, 0.8], 0.1, 0.0);
    let r1_on_visceral = org.body.insula.region1_pain_temp[0];
    let posterior_specificity = r1_on_pain - r1_on_visceral;

    // 2. Top-down cognitive reappraisal (PFC -> Region IV -> Region III) shapes how identical bodily deficit is felt
    org.body = HomeostaticBody::default();
    let _ = org.body.step_homeostasis_with_cognition(&[-0.08, -0.05, -0.08, -0.04], 0.05, 0.2, -0.6);
    let distress_catastrophized = org.body.distress;

    org.body = HomeostaticBody::default();
    let _ = org.body.step_homeostasis_with_cognition(&[-0.08, -0.05, -0.08, -0.04], 0.05, 0.2, 0.8);
    let distress_reappraised = org.body.distress;
    let top_down_relief_delta = distress_catastrophized - distress_reappraised;

    // 3. Polar Region IV salience & action-mode switch invariance across Go (+surprise) vs Withhold (-surprise)
    let (_, sw_go) = org.body.insula.step_insula([0.0; 3], [0.0; 3], [0.2, 0.2], 0.75, 0.0);
    let (_, sw_withhold) = org.body.insula.step_insula([0.0; 3], [0.0; 3], [0.2, 0.2], -0.75, 0.0);
    let (_, sw_freq_neutral) = org.body.insula.step_insula([0.0; 3], [0.0; 3], [0.05, 0.05], 0.02, 0.0);
    let polar_invariance = 0.5 * (sw_go + sw_withhold) - sw_freq_neutral;

    let passed = posterior_specificity > 0.25 && top_down_relief_delta > 0.03 && polar_invariance > 0.40;
    (posterior_specificity, top_down_relief_delta, polar_invariance, passed)
}

pub fn run_s0_senses(org: &mut Organism) -> (f32, f32, bool) {
    let mut rng = Rng64::new(777);
    for s in 0..80 {
        let pcm = synthesize_word_utterance(s % NUM_WORDS, s % 12, 0.0, &mut rng);
        org.develop_hearing(&pcm);
    }
    for i in 0..80 {
        let objs = vec![
            WorldObject { colour_id: i % NUM_COLOURS, shape_id: i % NUM_SHAPES, cy: 40.0, cx: 40.0 },
            WorldObject { colour_id: (i + 2) % NUM_COLOURS, shape_id: (i + 3) % NUM_SHAPES, cy: 85.0, cx: 85.0 },
        ];
        let sc = render_scene(&objs, &mut rng);
        org.develop_vision(&sc.rgb, sc.width, sc.height);
    }

    let mut h_bar = vec![0.0f32; 24 * 24];
    let mut v_bar = vec![0.0f32; 24 * 24];
    for y in 9..15 {
        for x in 3..21 {
            h_bar[y * 24 + x] = 1.0;
            v_bar[x * 24 + y] = 1.0;
        }
    }
    let fh = org.v1_features(&h_bar, false);
    let fv = org.v1_features(&v_bar, false);
    let v1_selectivity = 1.0 - (dot(&fh, &fv) / (l2_norm(&fh) * l2_norm(&fv) + 1e-8));

    let mut hits = 0usize;
    let total = 120usize;
    for t in 0..total {
        let w_a = t % NUM_WORDS;
        let w_b = (w_a + 1 + (t % (NUM_WORDS - 1))) % NUM_WORDS;
        let a = org.word_form(&synthesize_word_utterance(w_a, 20 + (t % 8), 0.0, &mut rng), false);
        let b = org.word_form(&synthesize_word_utterance(w_b, 28 + (t % 8), 0.0, &mut rng), false);
        let x = org.word_form(&synthesize_word_utterance(w_a, 36 + (t % 8), 0.0, &mut rng), false);
        if dot(&x, &a) > dot(&x, &b) {
            hits += 1;
        }
    }
    let abx_acc = (hits as f32) / (total as f32);
    (v1_selectivity, abx_acc, v1_selectivity > 0.15 && abx_acc > 0.75)
}

pub fn run_s1_saffran(org: &mut Organism) -> (f32, f32, bool) {
    let mut rng = Rng64::new(888);
    let words: [[usize; 3]; 4] = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]];
    org.wa.prev_active.clear();
    for _ in 0..200 {
        let w = words[(rng.next_u64() as usize) % 4];
        for &syl in &w {
            let pcm = synthesize_word_utterance(syl, (rng.next_u64() as usize) % 6, 0.0, &mut rng);
            let wf = org.word_form(&pcm, false);
            org.wa.fire_and_learn(&wf, None);
        }
    }
    let mut syl_assemblies: Vec<Vec<f32>> = Vec::with_capacity(12);
    for s in 0..12 {
        let pcm = synthesize_word_utterance(s, 2, 0.0, &mut rng);
        let wf = org.word_form(&pcm, false);
        syl_assemblies.push(org.wa.fire(&wf));
    }
    let mut word_score = 0.0f32;
    for w in &words {
        word_score += org.wa.sequence_transition_score(&syl_assemblies[w[0]], &syl_assemblies[w[1]]);
        word_score += org.wa.sequence_transition_score(&syl_assemblies[w[1]], &syl_assemblies[w[2]]);
    }
    word_score /= 8.0;

    let mut part_score = 0.0f32;
    let mut part_n = 0.0f32;
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                part_score += org.wa.sequence_transition_score(&syl_assemblies[words[i][2]], &syl_assemblies[words[j][0]]);
                part_n += 1.0;
            }
        }
    }
    part_score /= part_n;
    (word_score, part_score, word_score > part_score * 1.50)
}

pub fn run_s2_grounding(org: &mut Organism, shuffled: bool) -> (f32, f32, f32, f32, f32, f32, f32) {
    let mut rng = Rng64::new(999);
    let held_combos = [(0usize, 0usize), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];
    let positions = [(36.0f32, 36.0f32), (36.0, 92.0), (92.0, 36.0), (92.0, 92.0)];

    for step in 0..720 {
        let mut objs = Vec::with_capacity(3);
        let perm = rng.shuffle_indices(4);
        let c_perm = rng.shuffle_indices(NUM_COLOURS);
        let s_perm = rng.shuffle_indices(NUM_SHAPES);
        for k in 0..3 {
            let cid = c_perm[k];
            let mut sid = s_perm[k];
            if held_combos.contains(&(cid, sid)) {
                sid = s_perm[(k + 3) % NUM_SHAPES];
            }
            let (cy, cx) = positions[perm[k]];
            objs.push(WorldObject { colour_id: cid, shape_id: sid, cy, cx });
        }
        let target = &objs[(rng.next_u64() as usize) % objs.len()];
        let mut word_id = if (step % 2) == 0 {
            target.colour_id
        } else {
            NUM_COLOURS + target.shape_id
        };
        if shuffled {
            word_id = (rng.next_u64() as usize) % NUM_WORDS;
        }
        let spk = (rng.next_u64() as usize) % 16;
        let scene = render_scene(&objs, &mut rng);
        let pcm = synthesize_word_utterance(word_id, spk, 0.0, &mut rng);
        org.experience(&scene.rgb, scene.width, scene.height, &pcm, 3);
    }

    let eval = |org: &mut Organism, spk_base: usize, voice_shift: f32, novel_combos: bool, rng: &mut Rng64| -> (f32, f32) {
        let mut hits = 0usize;
        let mut total = 0usize;
        let mut chance_sum = 0.0f32;
        for t in 0..120 {
            let c0 = (t / 2) % NUM_COLOURS;
            let s0 = if novel_combos { c0 } else { (c0 + 1 + (t % 3)) % NUM_SHAPES };
            let raw_objs = [
                (c0, s0),
                ((c0 + 2) % NUM_COLOURS, (s0 + 2) % NUM_SHAPES),
                ((c0 + 4) % NUM_COLOURS, (s0 + 4) % NUM_SHAPES),
            ];
            let pos_perm = rng.shuffle_indices(3);
            let coords = [(36.0f32, 36.0f32), (36.0, 92.0), (92.0, 64.0)];
            let mut objs = Vec::with_capacity(3);
            for k in 0..3 {
                let (cy, cx) = coords[pos_perm[k]];
                objs.push(WorldObject {
                    colour_id: raw_objs[k].0,
                    shape_id: raw_objs[k].1,
                    cy,
                    cx,
                });
            }
            let scene = render_scene(&objs, rng);
            let (perceived, _) = org.perceive_objects(&scene.rgb, scene.width, scene.height, false);
            if perceived.len() < 2 {
                continue;
            }
            let ask_colour = (t % 2) == 0;
            let word_id = if ask_colour { c0 } else { NUM_COLOURS + s0 };
            let pcm = synthesize_word_utterance(word_id, spk_base + (t % 8), voice_shift, rng);
            if let Some(choice) = org.attend_to_word(&pcm, &perceived) {
                let chosen = &perceived[choice];
                let mut best_o = &objs[0];
                let mut best_d = f32::INFINITY;
                for o in &objs {
                    let d = (o.cy - chosen.cy).powi(2) + (o.cx - chosen.cx).powi(2);
                    if d < best_d {
                        best_d = d;
                        best_o = o;
                    }
                }
                let ok = if ask_colour { best_o.colour_id == c0 } else { best_o.shape_id == s0 };
                if ok {
                    hits += 1;
                }
                chance_sum += 1.0 / (perceived.len() as f32);
                total += 1;
            }
        }
        ((hits as f32) / (total.max(1) as f32), chance_sum / (total.max(1) as f32))
    };

    let (c1_acc, c1_chance) = eval(org, 20, 0.0, false, &mut rng);
    let (c2_acc, _) = eval(org, 30, 0.06, false, &mut rng);
    let (c3_acc, _) = eval(org, 20, 0.0, true, &mut rng);

    let mut col_pref = 0.0f32;
    for w in 0..NUM_COLOURS {
        let pcm = synthesize_word_utterance(w, 22, 0.0, &mut rng);
        col_pref += org.stream_preference(&pcm);
    }
    col_pref /= NUM_COLOURS as f32;

    let mut shp_pref = 0.0f32;
    for w in NUM_COLOURS..NUM_WORDS {
        let pcm = synthesize_word_utterance(w, 22, 0.0, &mut rng);
        shp_pref += org.stream_preference(&pcm);
    }
    shp_pref /= NUM_SHAPES as f32;

    let n_proto = org.binding.lexicon.len().max(1);
    let mut counts = vec![vec![0usize; NUM_WORDS]; n_proto];
    let mut total_tok = 0usize;
    for w in 0..NUM_WORDS {
        for spk in 20..26 {
            let pcm = synthesize_word_utterance(w, spk, 0.0, &mut rng);
            let (a, _) = org.hear_word(&pcm, false);
            if !org.binding.lexicon.is_empty() {
                let mut bj = 0usize;
                let mut bov = f32::NEG_INFINITY;
                for (j, p) in org.binding.lexicon.iter().enumerate() {
                    let ov = dot(&a, p);
                    if ov > bov {
                        bov = ov;
                        bj = j;
                    }
                }
                counts[bj][w] += 1;
                total_tok += 1;
            }
        }
    }
    let mut max_sum = 0usize;
    for row in &counts {
        max_sum += row.iter().copied().max().unwrap_or(0);
    }
    let purity = (max_sum as f32) / (total_tok.max(1) as f32);

    (c1_acc, c1_chance, c2_acc, c3_acc, col_pref, shp_pref, purity)
}

pub fn run_s3_babbling(org: &mut Organism) -> (f32, f32, bool) {
    for _ in 0..220 {
        org.babble_step(0.08);
    }
    let mut rng = Rng64::new(1234);
    let mut im_cos = 0.0f32;
    let mut null_cos = 0.0f32;
    let n = 30usize;
    for _ in 0..n {
        let mut target_m = [0.0f32; 8];
        let mut rand_m = [0.0f32; 8];
        for d in 0..8 {
            target_m[d] = rng.next_f32();
            rand_m[d] = rng.next_f32();
        }
        let target_pcm = org.vocal.synthesize_pcm(&target_m);
        let (_, im_pcm) = org.imitate_audio(&target_pcm);
        let rand_pcm = org.vocal.synthesize_pcm(&rand_m);

        let wf_t = org.word_form(&target_pcm, false);
        let wf_i = org.word_form(&im_pcm, false);
        let wf_r = org.word_form(&rand_pcm, false);

        im_cos += dot(&wf_t, &wf_i);
        null_cos += dot(&wf_t, &wf_r);
    }
    im_cos /= n as f32;
    null_cos /= n as f32;
    (im_cos, null_cos, im_cos > null_cos + 0.05)
}


pub fn run_l1_longitudinal_autonomy() -> (f32, f32, f32, f32, usize, bool) {
    let mut org = Organism::new(Genome::default());
    let _ = run_s0_senses(&mut org);

    // T0: Naive newborn evaluation before cross-situational experience
    let acc_t0 = run_s2_lesion_eval(&mut org);

    // T1: Early autonomous experience (180 episodes with independent random distractors)
    let mut rng = Rng64::new(909);
    let positions = [(36.0f32, 36.0f32), (36.0, 92.0), (92.0, 36.0), (92.0, 92.0)];
    for step in 0..180 {
        let mut objs = Vec::with_capacity(3);
        let perm = rng.shuffle_indices(4);
        let c_perm = rng.shuffle_indices(NUM_COLOURS);
        let s_perm = rng.shuffle_indices(NUM_SHAPES);
        for k in 0..3 {
            let (cy, cx) = positions[perm[k]];
            objs.push(WorldObject { colour_id: c_perm[k], shape_id: s_perm[k], cy, cx });
        }
        let target = &objs[(rng.next_u64() as usize) % objs.len()];
        let word_id = if (step % 2) == 0 { target.colour_id } else { NUM_COLOURS + target.shape_id };
        let spk = (rng.next_u64() as usize) % 16;
        let scene = render_scene(&objs, &mut rng);
        let pcm = synthesize_word_utterance(word_id, spk, 0.0, &mut rng);
        let _ = org.body.step_homeostasis_with_cognition(&[0.01, 0.01, 0.02, 0.01], 0.03, 0.25, 0.2);
        org.experience(&scene.rgb, scene.width, scene.height, &pcm, 3);
    }
    let acc_t1 = run_s2_lesion_eval(&mut org);

    // T2: Mature experience (full cross-situational curriculum)
    let (acc_t2, _, _, _, _, _, _) = run_s2_grounding(&mut org, false);

    // T3: Offline Sleep Consolidation (SWR replay + metabolic pruning with ZERO new world data)
    let (pruned, _) = org.sleep_consolidate(24, 0.02);
    let acc_t3 = run_s2_lesion_eval(&mut org);

    let passed = acc_t1 > acc_t0 + 0.10 && acc_t2 >= acc_t1 && acc_t3 >= acc_t2 && pruned > 100;
    (acc_t0, acc_t1, acc_t2, acc_t3, pruned, passed)
}

pub fn run_all_gates(organism_src_dir: &Path) -> GateReport {
    let (g0_ok, g0_files) = run_g0_anti_cheat(organism_src_dir);
    let mut org = Organism::new(Genome::default());

    let (b1_rec, b1_word, b1_atp, b1_ok) = run_b1_biophysical_heatmap(&mut org);
    let (b2_learn, b2_noise, b2_stat, b2_ok) = run_b2_curiosity_learning_progress();
    let (b3_pos, b3_neg, b3_gain, b3_ok) = run_b3_homeostatic_feelings(&mut org);
    let (b4_acc, b4_ok) = run_b4_feeling_driven_speech(&mut org);
    let (b5_spec, b5_td, b5_inv, b5_ok) = run_b5_quadripartite_insula(&mut org);

    let (v1_sel, abx_acc, s0_ok) = run_s0_senses(&mut org);
    let (w_sc, pw_sc, s1_ok) = run_s1_saffran(&mut org);
    let (c1, c1_ch, c2, c3, col_p, shp_p, purity) = run_s2_grounding(&mut org, false);

    let mut org_lesion = Organism::new(Genome::default());
    org_lesion.cochlea = org.cochlea.clone();
    org_lesion.a1 = org.a1.clone();
    org_lesion.wa = org.wa.clone();
    org_lesion.v1 = org.v1.clone();
    org_lesion.v4 = org.v4.clone();
    org_lesion.it = org.it.clone();
    org_lesion.c4 = org.c4.clone();
    org_lesion.binding = org.binding.clone();
    org_lesion.binding.lesion();
    let c4_lesion = run_s2_lesion_eval(&mut org_lesion);

    let mut org_shuf = Organism::new(Genome::default());
    let _ = run_s0_senses(&mut org_shuf);
    let (c5_shuf, _, _, _, _, _, _) = run_s2_grounding(&mut org_shuf, true);

    let (im_cos, null_cos, s3_ok) = run_s3_babbling(&mut org);
    let (l1_t0, l1_t1, l1_t2, l1_t3, l1_pruned, l1_ok) = run_l1_longitudinal_autonomy();
    let ram_mb = (org.memory_bytes() as f32) / (1024.0 * 1024.0);
    let ram_ok = org.ram_pressure_ok();

    let s2_ok = c1 > c1_ch + 0.25 && c2 > c1_ch + 0.20 && c3 > c1_ch + 0.20 && col_p > shp_p && c5_shuf < c1_ch + 0.12;

    GateReport {
        g0_anti_cheat_zero_strings: g0_ok,
        g0_files_scanned: g0_files,
        b1_corrupted_cue_recovery: b1_rec,
        b1_bound_word_recall: b1_word,
        b1_atp_energy_constrained: b1_atp,
        b1_passed: b1_ok,
        b2_curiosity_learnable_prog: b2_learn,
        b2_curiosity_noisy_tv_prog: b2_noise,
        b2_curiosity_static_prog: b2_stat,
        b2_passed: b2_ok,
        b3_positive_valence_on_relief: b3_pos,
        b3_negative_valence_on_harm: b3_neg,
        b3_emotional_memory_gain_ratio: b3_gain,
        b3_passed: b3_ok,
        b4_feeling_speech_need_acc: b4_acc,
        b4_passed: b4_ok,
        b5_posterior_pain_specificity: b5_spec,
        b5_top_down_relief_delta: b5_td,
        b5_polar_action_mode_invariance: b5_inv,
        b5_passed: b5_ok,
        s0_v1_selectivity: v1_sel,
        s0_a1_abx_acc: abx_acc,
        s0_passed: s0_ok,
        s1_saffran_word_score: w_sc,
        s1_saffran_partword_score: pw_sc,
        s1_passed: s1_ok,
        s2_c1_heldout_spk_acc: c1,
        s2_c1_chance: c1_ch,
        s2_c2_heldout_voice_acc: c2,
        s2_c3_novel_combos_acc: c3,
        s2_c4_lesion_acc: c4_lesion,
        s2_c5_shuffled_acc: c5_shuf,
        s2_c6_colour_stream_pref: col_p,
        s2_c6_shape_stream_pref: shp_p,
        s2_c7_lexicon_purity: purity,
        s2_passed: s2_ok,
        s3_babble_imitation_cos: im_cos,
        s3_babble_null_cos: null_cos,
        s3_passed: s3_ok,
        l1_acc_t0_naive: l1_t0,
        l1_acc_t1_early: l1_t1,
        l1_acc_t2_late: l1_t2,
        l1_acc_t3_post_sleep: l1_t3,
        l1_pruned_synapses: l1_pruned,
        l1_passed: l1_ok,
        organism_ram_mb: ram_mb,
        ram_budget_ok: ram_ok,
    }
}

fn run_s2_lesion_eval(org: &mut Organism) -> f32 {
    let mut rng = Rng64::new(555);
    let mut hits = 0usize;
    let mut total = 0usize;
    let coords = [(36.0f32, 36.0f32), (36.0, 92.0), (92.0, 64.0)];
    for t in 0..90 {
        let c0 = t % NUM_COLOURS;
        let s0 = (t + 1) % NUM_SHAPES;
        let pos_perm = rng.shuffle_indices(3);
        let objs = vec![
            WorldObject { colour_id: c0, shape_id: s0, cy: coords[pos_perm[0]].0, cx: coords[pos_perm[0]].1 },
            WorldObject { colour_id: (c0 + 2) % NUM_COLOURS, shape_id: (s0 + 2) % NUM_SHAPES, cy: coords[pos_perm[1]].0, cx: coords[pos_perm[1]].1 },
            WorldObject { colour_id: (c0 + 4) % NUM_COLOURS, shape_id: (s0 + 4) % NUM_SHAPES, cy: coords[pos_perm[2]].0, cx: coords[pos_perm[2]].1 },
        ];
        let scene = render_scene(&objs, &mut rng);
        let (perceived, _) = org.perceive_objects(&scene.rgb, scene.width, scene.height, false);
        if perceived.len() < 2 {
            continue;
        }
        let pcm = synthesize_word_utterance(c0, 20 + (t % 6), 0.0, &mut rng);
        if let Some(choice) = org.attend_to_word(&pcm, &perceived) {
            let chosen = &perceived[choice];
            let d0 = (objs[0].cy - chosen.cy).powi(2) + (objs[0].cx - chosen.cx).powi(2);
            if d0 < 225.0 {
                hits += 1;
            }
            total += 1;
        }
    }
    (hits as f32) / (total.max(1) as f32)
}
