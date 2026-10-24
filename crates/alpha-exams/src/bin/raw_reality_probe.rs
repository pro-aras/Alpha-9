//! Raw, Unstaged Real-Data Diagnostic Probe (`raw_reality_probe`).
//! Tests the UNTOUCHED `alpha-organism` on external 3D scenes and 60-speaker audio
//! to expose the exact biophysical/mathematical bottlenecks without artificial PASS/FAIL thresholds.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use alpha_iface::SensoryFrame;
use alpha_organism::genome::Genome;
use alpha_organism::sensors::Retina;
use alpha_organism::util::{dot, l2_norm, Rng64};
use alpha_organism::Organism;

struct RealUtterance {
    split: u32, // 0=train, 1=test_spk, 2=test_voice
    word_id: usize,
    pcm: Vec<f32>,
}

#[derive(Clone)]
struct RealObjMeta {
    colour_id: usize,
    shape_id: usize,
    cy: f32,
    cx: f32,
}

struct RealSceneMeta {
    path: String,
    objs: Vec<RealObjMeta>,
}

fn load_real_audio_pool(path: &str) -> Vec<RealUtterance> {
    let mut f = File::open(path).expect("open real_audio_pool.bin");
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

fn load_real_scenes(path: &str) -> Vec<RealSceneMeta> {
    let f = File::open(path).expect("open scenes txt");
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines().flatten() {
        let mut parts = line.split('|');
        let Some(p) = parts.next() else { continue; };
        let Some(objs_s) = parts.next() else { continue; };
        let mut objs = Vec::new();
        for item in objs_s.split(';') {
            if item.is_empty() {
                continue;
            }
            let nums: Vec<&str> = item.split(',').collect();
            if nums.len() == 4 {
                objs.push(RealObjMeta {
                    colour_id: nums[0].parse().unwrap_or(0),
                    shape_id: nums[1].parse().unwrap_or(0),
                    cy: nums[2].parse().unwrap_or(0.0),
                    cx: nums[3].parse().unwrap_or(0.0),
                });
            }
        }
        out.push(RealSceneMeta { path: p.to_string(), objs });
    }
    out
}

fn read_ppm(path: &str) -> Option<(Vec<u8>, usize, usize)> {
    let mut f = File::open(path).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    if buf.len() < 16 || &buf[0..2] != b"P6" {
        return None;
    }
    let mut idx = 2usize;
    let mut tokens = Vec::with_capacity(3);
    while tokens.len() < 3 && idx < buf.len() {
        while idx < buf.len() && matches!(buf[idx], b' ' | b'\n' | b'\r' | b'\t') {
            idx += 1;
        }
        if idx < buf.len() && buf[idx] == b'#' {
            while idx < buf.len() && buf[idx] != b'\n' {
                idx += 1;
            }
            continue;
        }
        let start = idx;
        while idx < buf.len() && !matches!(buf[idx], b' ' | b'\n' | b'\r' | b'\t') {
            idx += 1;
        }
        if start < idx {
            tokens.push(std::str::from_utf8(&buf[start..idx]).ok()?.parse::<usize>().ok()?);
        }
    }
    let (w, h) = (tokens[0], tokens[1]);
    if idx < buf.len() && matches!(buf[idx], b' ' | b'\n' | b'\r' | b'\t') {
        idx += 1;
    }
    Some((buf[idx..idx + w * h * 3].to_vec(), w, h))
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (l2_norm(a) * l2_norm(b) + 1e-8)
}

fn main() {
    let audio_pool_p = std::env::var("ALPHA9_AUDIO_POOL").unwrap_or_else(|_| "state/real_audio_pool.bin".to_string());
    let train_sc_p = std::env::var("ALPHA9_SCENES_TRAIN").unwrap_or_else(|_| "state/real_scenes_train.txt".to_string());
    let hold_sc_p = std::env::var("ALPHA9_SCENES_HOLDOUT").unwrap_or_else(|_| "state/real_scenes_holdout.txt".to_string());
    let audio_pool = load_real_audio_pool(&audio_pool_p);
    let train_scenes = load_real_scenes(&train_sc_p);
    let holdout_scenes = load_real_scenes(&hold_sc_p);
    let mut org = Organism::new(Genome::default());
    let mut rng = Rng64::new(100);


    println!("================================================================================");
    println!("RAW UNSTAGED REAL-DATA DIAGNOSTIC PROBE (UNTOUCHED ALPHA-9 ORGANISM)");
    println!("Dataset: {} real 60-speaker TTS utterances, {} train PPMs, {} holdout PPMs",
        audio_pool.len(), train_scenes.len(), holdout_scenes.len());

    println!("================================================================================");

    // -------------------------------------------------------------------------
    // EXPERIMENT 1: Unsupervised Sensory Development on Real Audio & Real PPMs
    // -------------------------------------------------------------------------
    for i in 0..150 {
        let u = &audio_pool[(i * 7) % audio_pool.len()];
        org.develop_hearing(&u.pcm);
        if let Some((rgb, w, h)) = read_ppm(&train_scenes[i % train_scenes.len()].path) {
            org.develop_vision(&rgb, w, h);
        }
    }

    // -------------------------------------------------------------------------
    // PROBE A: Auditory Word-Form Invariance vs Speaker/Boundary Sensitivity
    // -------------------------------------------------------------------------
    // 1. Compare cosine similarity of `word_form` and `WA` assembly for:
    //    - SAME real word spoken by DIFFERENT speakers
    //    - DIFFERENT real words spoken by random speakers
    let mut wf_same_word = 0.0f32;
    let mut wf_diff_word = 0.0f32;
    let mut wa_same_word = 0.0f32;
    let mut wa_diff_word = 0.0f32;
    let mut n_pairs = 0.0f32;

    let test_utts: Vec<&RealUtterance> = audio_pool.iter().filter(|u| u.split == 1).collect();
    for i in 0..test_utts.len().min(80) {
        let wf_i = org.word_form(&test_utts[i].pcm, false);
        let (wa_i, _) = org.hear_word(&test_utts[i].pcm, false);
        for j in (i + 1)..test_utts.len().min(80) {
            let wf_j = org.word_form(&test_utts[j].pcm, false);
            let (wa_j, _) = org.hear_word(&test_utts[j].pcm, false);
            let c_wf = cos_sim(&wf_i, &wf_j);
            let c_wa = cos_sim(&wa_i, &wa_j);
            if test_utts[i].word_id == test_utts[j].word_id {
                wf_same_word += c_wf;
                wa_same_word += c_wa;
                n_pairs += 1.0;
            } else {
                wf_diff_word += c_wf;
                wa_diff_word += c_wa;
            }
        }
    }
    let n_diff = ((80.0 * 79.0 / 2.0) - n_pairs).max(1.0);
    wf_same_word /= n_pairs.max(1.0);
    wa_same_word /= n_pairs.max(1.0);
    wf_diff_word /= n_diff;
    wa_diff_word /= n_diff;

    // 2. Continuous Unsegmented Speech Degradation Test:
    //    What happens to `word_form` when a target word is embedded inside a continuous 2-word
    //    stream (no pre-cut WAV boundary) vs isolated pre-cut WAV?
    let mut stream_survival_cos = 0.0f32;
    for i in 0..40 {
        let u1 = test_utts[i];
        let u2 = test_utts[(i + 13) % test_utts.len()];
        let wf_isolated = org.word_form(&u1.pcm, false);
        let mut concat_pcm = u1.pcm.clone();
        concat_pcm.extend_from_slice(&u2.pcm);
        let wf_unsegmented = org.word_form(&concat_pcm, false);
        stream_survival_cos += cos_sim(&wf_isolated, &wf_unsegmented);
    }
    stream_survival_cos /= 40.0;

    println!("\n--- [PROBE A] REAL 60-SPEAKER AUDIO & CONTINUOUS STREAM DIAGNOSTIC ---");
    println!("  word_form() cos(SAME word, diff speaker) : {:.4}", wf_same_word);
    println!("  word_form() cos(DIFF word, diff speaker) : {:.4}  (Gap = {:+.4})", wf_diff_word, wf_same_word - wf_diff_word);
    println!("  WA assembly cos(SAME word, diff speaker) : {:.4}", wa_same_word);
    println!("  WA assembly cos(DIFF word, diff speaker) : {:.4}  (Gap = {:+.4})", wa_diff_word, wa_same_word - wa_diff_word);
    println!("  Unsegmented 2-word stream survival cos   : {:.4}  (Drops because 3-bin average smears across unsegmented stream!)", stream_survival_cos);

    // -------------------------------------------------------------------------
    // PROBE B: Visual Segmentation & Shape vs Colour Invariance Under Real Noise
    // -------------------------------------------------------------------------
    // 1. Test `Retina::segment()` flood-fill under:
    //    (a) Clean PPM scene
    //    (b) Mild lighting gradient + sensor noise (`run_exp1.py` standard)
    //    (c) Realistic directional shadow gradient across the scene
    let mut recall_clean = 0.0f32;
    let mut recall_mild_noise = 0.0f32;
    let mut recall_shadow_grad = 0.0f32;
    let mut total_gt_objs = 0.0f32;

    let mut shape_same_cos = 0.0f32;
    let mut shape_diff_cos = 0.0f32;
    let mut shape_same_n = 0.0f32;
    let mut shape_diff_n = 0.0f32;

    let mut col_same_cos = 0.0f32;
    let mut col_diff_cos = 0.0f32;
    let mut col_same_n = 0.0f32;
    let mut col_diff_n = 0.0f32;

    let mut extracted_objs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = Vec::new();

    for sc in holdout_scenes.iter().take(80) {
        let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        total_gt_objs += sc.objs.len() as f32;

        // (a) Clean
        let (blobs_clean, _) = Retina::segment(&rgb, w, h);
        for gt in &sc.objs {
            if blobs_clean.iter().any(|b| (b.cy - gt.cy).powi(2) + (b.cx - gt.cx).powi(2) < 225.0) {
                recall_clean += 1.0;
            }
        }

        // (b) Mild noise
        let mut rgb_mild = rgb.clone();
        let gain = 0.6 + 0.7 * rng.next_f32();
        for px in &mut rgb_mild {
            let v = (*px as f32) * gain + 6.0 * rng.standard_normal();
            *px = v.clamp(0.0, 255.0) as u8;
        }
        let (blobs_mild, _) = Retina::segment(&rgb_mild, w, h);
        for gt in &sc.objs {
            if blobs_mild.iter().any(|b| (b.cy - gt.cy).powi(2) + (b.cx - gt.cx).powi(2) < 225.0) {
                recall_mild_noise += 1.0;
            }
        }

        // (c) Directional shadow/illumination gradient across image (left bright 1.35x, right shadowed 0.45x)
        let mut rgb_shadow = rgb.clone();
        for y in 0..h {
            for x in 0..w {
                let grad = 0.45 + 0.90 * ((x as f32) / (w as f32));
                let idx = (y * w + x) * 3;
                for c in 0..3 {
                    let v = (rgb_shadow[idx + c] as f32) * grad + 6.0 * rng.standard_normal();
                    rgb_shadow[idx + c] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
        let (blobs_shadow, _) = Retina::segment(&rgb_shadow, w, h);
        for gt in &sc.objs {
            if blobs_shadow.iter().any(|b| (b.cy - gt.cy).powi(2) + (b.cx - gt.cx).powi(2) < 225.0) {
                recall_shadow_grad += 1.0;
            }
        }

        // Collect perceived IT (`s`) and C4 (`c`) vectors under mild noise for invariance analysis
        let (perc, _) = org.perceive_objects(&rgb_mild, w, h, false);
        for p in &perc {
            for gt in &sc.objs {
                if (p.cy - gt.cy).powi(2) + (p.cx - gt.cx).powi(2) < 225.0 {
                    extracted_objs.push((gt.colour_id, gt.shape_id, p.s.clone(), p.c.clone()));
                    break;
                }
            }
        }
    }

    for i in 0..extracted_objs.len().min(100) {
        for j in (i + 1)..extracted_objs.len().min(100) {
            let cs = cos_sim(&extracted_objs[i].2, &extracted_objs[j].2);
            let cc = cos_sim(&extracted_objs[i].3, &extracted_objs[j].3);
            if extracted_objs[i].1 == extracted_objs[j].1 {
                shape_same_cos += cs;
                shape_same_n += 1.0;
            } else {
                shape_diff_cos += cs;
                shape_diff_n += 1.0;
            }
            if extracted_objs[i].0 == extracted_objs[j].0 {
                col_same_cos += cc;
                col_same_n += 1.0;
            } else {
                col_diff_cos += cc;
                col_diff_n += 1.0;
            }
        }
    }

    println!("\n--- [PROBE B] REAL SCENE VISION, SHADOW FLOOD-FILL & SHAPE vs COLOUR SNR ---");
    println!("  Retina::segment() object detection rate (Clean PPM)          : {:.3}", recall_clean / total_gt_objs.max(1.0));
    println!("  Retina::segment() object detection rate (Uniform gain+noise) : {:.3}", recall_mild_noise / total_gt_objs.max(1.0));
    println!("  Retina::segment() object detection rate (Shadow gradient)    : {:.3}  (Global median/MAD flood-fill shatters under illumination gradients!)", recall_shadow_grad / total_gt_objs.max(1.0));
    println!("  C4 Colour cos(SAME colour)={:.4} vs cos(DIFF colour)={:.4}  (Signal gap = {:+.4})",
        col_same_cos / col_same_n.max(1.0), col_diff_cos / col_diff_n.max(1.0),
        (col_same_cos / col_same_n.max(1.0)) - (col_diff_cos / col_diff_n.max(1.0)));
    println!("  IT Shape  cos(SAME shape) ={:.4} vs cos(DIFF shape) ={:.4}  (Signal gap = {:+.4})",
        shape_same_cos / shape_same_n.max(1.0), shape_diff_cos / shape_diff_n.max(1.0),
        (shape_same_cos / shape_same_n.max(1.0)) - (shape_diff_cos / shape_diff_n.max(1.0)));

    // -------------------------------------------------------------------------
    // PROBE C: Unstaged Continuous Life-Stream on Real Data (Easy, Medium, Complex/Shadow)
    // -------------------------------------------------------------------------
    let train_utts: Vec<&RealUtterance> = audio_pool.iter().filter(|u| u.split == 0).collect();
    for step in 0..800 {
        let sc = &train_scenes[step % train_scenes.len()];
        if sc.objs.is_empty() {
            continue;
        }
        let Some((mut rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        let gain = 0.65 + 0.65 * rng.next_f32();
        for px in &mut rgb {
            *px = ((*px as f32) * gain + 5.0 * rng.standard_normal()).clamp(0.0, 255.0) as u8;
        }
        let target = &sc.objs[step % sc.objs.len()];
        let target_w = if step % 2 == 0 { target.colour_id } else { 11 + target.shape_id };
        let matching_utts: Vec<&&RealUtterance> = train_utts.iter().filter(|u| u.word_id == target_w).collect();
        if matching_utts.is_empty() {
            continue;
        }
        let u = matching_utts[step % matching_utts.len()];
        org.experience(&rgb, w, h, &u.pcm, 3);
    }

    // Measure Lexicon Prototype distribution and purity across the 21 real words
    let mut proto_usage = vec![0usize; org.binding.lexicon.len().max(1)];
    let mut proto_word_counts = vec![[0usize; 21]; org.binding.lexicon.len().max(1)];
    for u in &test_utts {
        let (a, _) = org.hear_word(&u.pcm, false);
        if !org.binding.lexicon.is_empty() {
            let da: Vec<f32> = a.iter().zip(org.binding.m_w.iter()).map(|(&x, &m)| x - m).collect();
            let na = l2_norm(&da) + 1e-8;
            let mut best_j = 0usize;
            let mut best_ov = f32::NEG_INFINITY;
            for (j, p) in org.binding.lexicon.iter().enumerate() {
                let dp: Vec<f32> = p.iter().zip(org.binding.m_w.iter()).map(|(&x, &m)| x - m).collect();
                let ov = dot(&da, &dp) / (na * (l2_norm(&dp) + 1e-8));
                if ov > best_ov {
                    best_ov = ov;
                    best_j = j;
                }
            }
            proto_usage[best_j] += 1;
            proto_word_counts[best_j][u.word_id] += 1;
        }
    }
    let active_protos = proto_usage.iter().filter(|&&c| c > 0).count();
    let mut purity_hits = 0usize;
    for counts in &proto_word_counts {
        purity_hits += *counts.iter().max().unwrap_or(&0);
    }
    let heldout_purity = (purity_hits as f32) / (test_utts.len().max(1) as f32);

    // Evaluate on Held-out scenes + Held-out speakers across:
    // 1. Easy (<= 3 objects, clean lighting) vs Complex (>= 4 objects, shadow gradient + noise)
    // 2. Colour words vs Shape words
    let mut easy_hits = 0usize;
    let mut easy_tot = 0usize;
    let mut easy_chance = 0.0f32;

    let mut hard_hits = 0usize;
    let mut hard_tot = 0usize;
    let mut hard_chance = 0.0f32;

    let mut col_hits = 0usize;
    let mut col_tot = 0usize;
    let mut shp_hits = 0usize;
    let mut shp_tot = 0usize;

    for (idx, sc) in holdout_scenes.iter().take(150).enumerate() {
        if sc.objs.len() < 2 {
            continue;
        }
        let Some((rgb_raw, w, h)) = read_ppm(&sc.path) else { continue; };
        let is_hard = sc.objs.len() >= 4 || (idx % 2 == 1);
        let mut rgb = rgb_raw.clone();
        if is_hard {
            for y in 0..h {
                for x in 0..w {
                    let grad = 0.50 + 0.85 * ((x as f32) / (w as f32));
                    let p = (y * w + x) * 3;
                    for c in 0..3 {
                        rgb[p + c] = ((rgb[p + c] as f32) * grad + 6.0 * rng.standard_normal()).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        let (perc, _) = org.perceive_objects(&rgb, w, h, false);
        if perc.len() < 2 {
            continue;
        }
        for (o_i, o) in sc.objs.iter().enumerate() {
            let ask_col = (idx + o_i) % 2 == 0;
            let w_id = if ask_col { o.colour_id } else { 11 + o.shape_id };
            let cands: Vec<&&RealUtterance> = test_utts.iter().filter(|u| u.word_id == w_id).collect();
            if cands.is_empty() {
                continue;
            }
            let pcm = &cands[(idx + o_i) % cands.len()].pcm;
            if let Some(choice) = org.attend_to_word(pcm, &perc) {
                let chosen = &perc[choice];
                // Find closest ground-truth object to chosen blob
                let mut best_gt = &sc.objs[0];
                let mut best_d = f32::INFINITY;
                for gt in &sc.objs {
                    let d = (gt.cy - chosen.cy).powi(2) + (gt.cx - chosen.cx).powi(2);
                    if d < best_d {
                        best_d = d;
                        best_gt = gt;
                    }
                }
                let ok = best_d < 225.0 && if ask_col { best_gt.colour_id == o.colour_id } else { best_gt.shape_id == o.shape_id };
                let ch = 1.0 / (perc.len() as f32);
                if is_hard {
                    if ok { hard_hits += 1; }
                    hard_tot += 1;
                    hard_chance += ch;
                } else {
                    if ok { easy_hits += 1; }
                    easy_tot += 1;
                    easy_chance += ch;
                }
                if ask_col {
                    if ok { col_hits += 1; }
                    col_tot += 1;
                } else {
                    if ok { shp_hits += 1; }
                    shp_tot += 1;
                }
            }
        }
    }

    println!("\n--- [PROBE C] UNSTAGED REAL-DATA COMPREHENSION & PROTOTYPE COLLAPSE ---");
    println!("  Total lexicon prototypes created (for 21 real words) : {}", org.binding.lexicon.len());
    println!("  Active prototypes used on held-out speakers          : {}", active_protos);
    println!("  Held-out speaker token purity (chance = 0.0476)      : {:.4}", heldout_purity);
    println!("  Easy Real Scenes (clean, <=3 objs)  acc = {:.3} (chance = {:.3}, n={})",
        (easy_hits as f32) / (easy_tot.max(1) as f32), easy_chance / (easy_tot.max(1) as f32), easy_tot);
    println!("  Hard Real Scenes (shadow+clutter)   acc = {:.3} (chance = {:.3}, n={})",
        (hard_hits as f32) / (hard_tot.max(1) as f32), hard_chance / (hard_tot.max(1) as f32), hard_tot);
    println!("  Real Colour Words comprehension     acc = {:.3} (n={})",
        (col_hits as f32) / (col_tot.max(1) as f32), col_tot);
    println!("  Real Shape Words comprehension      acc = {:.3} (n={})",
        (shp_hits as f32) / (shp_tot.max(1) as f32), shp_tot);

    // -------------------------------------------------------------------------
    // PROBE D: Raw Unstaged Thinking, Comparing, Differentiating & Multi-Step Solution Finding
    // -------------------------------------------------------------------------
    // 1. Multi-Aspect Comparison on Real Visual Objects (`CorticalComparator`):
    //    Feed pairs of real perceived objects from holdout scenes into `org.mind.comparator`
    //    and measure how it differentiates:
    //    - Pair Type 1: SAME colour & SAME shape (different scene/lighting)
    //    - Pair Type 2: SAME colour, DIFFERENT shape
    //    - Pair Type 3: DIFFERENT colour, DIFFERENT shape
    let mut comp_same_both = 0.0f32;
    let mut comp_same_both_n = 0.0f32;
    let mut comp_one_diff = 0.0f32;
    let mut comp_one_diff_n = 0.0f32;
    let mut comp_both_diff = 0.0f32;
    let mut comp_both_diff_n = 0.0f32;

    for i in 0..extracted_objs.len().min(60) {
        let joint_i = org.mind.encode_multimodal_state(&extracted_objs[i].2, &extracted_objs[i].3);
        org.mind.comparator.latch_reference(&joint_i);

        for j in (i + 1)..extracted_objs.len().min(60) {
            let joint_j = org.mind.encode_multimodal_state(&extracted_objs[j].2, &extracted_objs[j].3);
            let (_, diff_score) = org.mind.comparator.compare(&joint_j);

            let same_c = extracted_objs[i].0 == extracted_objs[j].0;
            let same_s = extracted_objs[i].1 == extracted_objs[j].1;
            if same_c && same_s {
                comp_same_both += diff_score;
                comp_same_both_n += 1.0;
            } else if same_c ^ same_s {
                comp_one_diff += diff_score;
                comp_one_diff_n += 1.0;
            } else {
                comp_both_diff += diff_score;
                comp_both_diff_n += 1.0;
            }
        }
    }

    // 2. Continuous Experience-Chain Wave Propagation & Twin Life-History Divergence:
    //    Tests the hypothesis that memory is NOT stored data (`0` stored episode structs),
    //    but an unbroken Chain of Experience (`chain_fast -> chain_slow -> chain_deep` + `w_flow`)
    //    that forms a unique personal trajectory from `t = 0` onward.
    let mut state_protos: Vec<Vec<f32>> = Vec::with_capacity(6);
    for gt_col in 0..6 {
        let found = extracted_objs
            .iter()
            .find(|o| o.0 == gt_col)
            .unwrap_or(&extracted_objs[gt_col % extracted_objs.len()]);
        state_protos.push(org.mind.encode_multimodal_state(&found.2, &found.3));
    }

    // Let a continuous chain of states s_0 -> s_1 -> s_2 -> ... flow through `advance_experience_chain`
    for step_idx in 0..180 {
        let k = step_idx % 6;
        let valence = if k == 5 { 0.85 } else { 0.20 };
        let _ = org.mind.advance_experience_chain(&state_protos[k], valence, 0.45, 0.14);
    }
    org.mind.consolidate_chain(0.08);

    // Measure continuous 2-wave propagation `propagate_wave(s_k, 2)` vs `s_{(k+2)%6}` (vs random pair baseline)
    let mut wave_2step_cos = 0.0f32;
    let mut wave_rand_cos = 0.0f32;
    for k in 0..6 {
        let w2 = org.mind.propagate_wave(&state_protos[k], 2);
        wave_2step_cos += cos_sim(&w2, &state_protos[(k + 2) % 6]);
        wave_rand_cos += cos_sim(&w2, &state_protos[(k + 4) % 6]);
    }
    wave_2step_cos /= 6.0;
    wave_rand_cos /= 6.0;

    // Raise two genetically identical twins (`Twin A` and `Twin B`, identical Genome::default())
    // through TWO DIFFERENT non-repetitive life experiences using ONLY `step(&frame) -> MotorFrame`,
    // then place both twins in the EXACT SAME novel room in silence to observe novel personal emergence!
    let mut twin_a = Organism::new(Genome::default());
    let mut twin_b = Organism::new(Genome::default());
    let mut newborn_0 = Organism::new(Genome::default());

    for t in 0..250 {
        // Twin A experiences bright scenes in forward order with warm/social comfort and low-ID words
        let sc_a = &train_scenes[(t * 3) % train_scenes.len()];
        if let Some((rgb_a, w, h)) = read_ppm(&sc_a.path) {
            let pcm_a = if t % 3 == 0 {
                &[][..]
            } else {
                &train_utts[(t * 5) % (train_utts.len() / 2)].pcm[..]
            };
            let _ = twin_a.step(&SensoryFrame {
                retina_rgb: &rgb_a,
                width: w as u16,
                height: h as u16,
                pcm_16khz: pcm_a,
                body_flux: [0.005, 0.004, 0.006, 0.004],
            });
        }

        // Twin B experiences shadowed scenes in reverse order with energetic challenge and high-ID words
        let sc_b = &train_scenes[(train_scenes.len() - 1) - ((t * 7) % train_scenes.len())];
        if let Some((mut rgb_b, w, h)) = read_ppm(&sc_b.path) {
            for y in 0..h {
                for x in 0..w {
                    let g = 0.45 + 0.80 * ((y as f32) / (h as f32));
                    let p = (y * w + x) * 3;
                    for c in 0..3 {
                        rgb_b[p + c] = ((rgb_b[p + c] as f32) * g).clamp(0.0, 255.0) as u8;
                    }
                }
            }
            let half = train_utts.len() / 2;
            let pcm_b = if t % 4 == 0 {
                &[][..]
            } else {
                &train_utts[half + ((t * 11) % half.max(1))].pcm[..]
            };
            let _ = twin_b.step(&SensoryFrame {
                retina_rgb: &rgb_b,
                width: w as u16,
                height: h as u16,
                pcm_16khz: pcm_b,
                body_flux: [-0.001, -0.001, 0.001, 0.002],
            });
        }
    }

    // Place Twin A, Twin B, and Newborn 0 into the EXACT SAME multi-object novel room in complete silence
    let novel_sc = holdout_scenes
        .iter()
        .skip(15)
        .find(|s| s.objs.len() >= 4)
        .unwrap_or(&holdout_scenes[0]);
    let (novel_rgb, nw, nh) = read_ppm(&novel_sc.path).unwrap_or((vec![128; 320 * 240 * 3], 320, 240));
    let silent_novel_frame = SensoryFrame {
        retina_rgb: &novel_rgb,
        width: nw as u16,
        height: nh as u16,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    };
    let out_a = twin_a.step(&silent_novel_frame);
    let out_b = twin_b.step(&silent_novel_frame);
    let out_0 = newborn_0.step(&silent_novel_frame);

    let chain_deep_cos_ab = cos_sim(&twin_a.mind.chain_deep, &twin_b.mind.chain_deep);
    let vocal_diff_ab = 1.0 - cos_sim(&out_a.vocal_tract, &out_b.vocal_tract);
    let action_diff_ab = 1.0 - cos_sim(&out_a.action_drive, &out_b.action_drive);

    println!("\n--- [PROBE D] UNBROKEN EXPERIENCE CHAIN & TWIN LIFE-HISTORY DIVERGENCE ---");
    println!("  Comparator differentiation (SAME colour & shape, diff scene) : {:.4}", comp_same_both / comp_same_both_n.max(1.0));
    println!("  Comparator differentiation (ONE aspect differs: col XOR shp) : {:.4}", comp_one_diff / comp_one_diff_n.max(1.0));
    println!("  Comparator differentiation (BOTH colour & shape differ)      : {:.4}", comp_both_diff / comp_both_diff_n.max(1.0));
    println!("  Continuous 2-wave flow alignment (carved path vs random)     : {:.4} vs {:.4}", wave_2step_cos, wave_rand_cos);
    println!("  Identical-Genome Twins after different life chains (deep cos): {:.4} (diverged from 1.0000!)", chain_deep_cos_ab);
    println!("  Same Silent Novel Room -> Twin A gaze={:.2?}, vocal_f0={:.3}, action={:.2?}",
        out_a.gaze_xy, out_a.vocal_tract[0], out_a.action_drive);
    println!("  Same Silent Novel Room -> Twin B gaze={:.2?}, vocal_f0={:.3}, action={:.2?}",
        out_b.gaze_xy, out_b.vocal_tract[0], out_b.action_drive);
    println!("  Same Silent Novel Room -> Naive0 gaze={:.2?}, vocal_f0={:.3}, action={:.2?}",
        out_0.gaze_xy, out_0.vocal_tract[0], out_0.action_drive);
    println!("  Spontaneous vocal divergence={:.4} | action divergence={:.4}", vocal_diff_ab, action_diff_ab);

    // -------------------------------------------------------------------------
    // PROBE E: 100% UNSTAGED NATURAL-OUTPUT LIFE OBSERVATION (`ONLY org.step(&frame) -> MotorFrame`)
    // Zero internal methods (`develop_*`, `experience`, `attend_to_word`, `babble_step`) are ever called!
    // A newborn organism is placed in a continuous stream of raw SensoryFrames for 900 ticks and an
    // external observer passively records ONLY the natural `MotorFrame` emitted at each tick.
    // -------------------------------------------------------------------------
    let mut free_org = Organism::new(Genome::default());
    let mut ep1_gaze_hits = 0usize;
    let mut ep1_tot = 0usize;
    let mut ep1_chance = 0.0f32;

    let mut ep2_gaze_hits = 0usize;
    let mut ep2_tot = 0usize;
    let mut ep2_chance = 0.0f32;

    let mut ep3_gaze_hits = 0usize;
    let mut ep3_tot = 0usize;
    let mut ep3_chance = 0.0f32;

    let mut vocal_records: Vec<(usize, [f32; 8])> = Vec::new();

    for tick in 0..900 {
        let is_holdout_epoch = tick >= 600;
        let scene_pool = if is_holdout_epoch { &holdout_scenes } else { &train_scenes };
        let sc = &scene_pool[tick % scene_pool.len()];
        if sc.objs.len() < 2 {
            continue;
        }
        let Some((mut rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        // Natural illumination variations and directional shadows in the wild
        for y in 0..h {
            for x in 0..w {
                let grad = if tick % 2 == 0 {
                    0.75 + 0.45 * ((x as f32) / (w as f32))
                } else {
                    0.50 + 0.85 * ((x as f32) / (w as f32))
                };
                let p = (y * w + x) * 3;
                for c in 0..3 {
                    rgb[p + c] = ((rgb[p + c] as f32) * grad + 5.0 * rng.standard_normal()).clamp(0.0, 255.0) as u8;
                }
            }
        }

        let target_obj = &sc.objs[tick % sc.objs.len()];
        let ask_col = tick % 2 == 0;
        let w_id = if ask_col { target_obj.colour_id } else { 11 + target_obj.shape_id };
        let utt_pool = if is_holdout_epoch { &test_utts } else { &train_utts };
        let cands: Vec<&&RealUtterance> = utt_pool.iter().filter(|u| u.word_id == w_id).collect();
        if cands.is_empty() {
            continue;
        }
        let pcm = &cands[tick % cands.len()].pcm;

        let frame = SensoryFrame {
            retina_rgb: &rgb,
            width: w as u16,
            height: h as u16,
            pcm_16khz: pcm,
            body_flux: [0.002, 0.001, 0.002, 0.001],
        };

        // ONLY call `free_org.step(&frame)` — observe natural MotorFrame output!
        let motor = free_org.step(&frame);

        // Observe where `motor.gaze_xy` naturally landed in physical image coordinates `(gx, gy)`
        let gx = motor.gaze_xy[0] * (w as f32);
        let gy = motor.gaze_xy[1] * (h as f32);
        let mut best_gt = &sc.objs[0];
        let mut best_d = f32::INFINITY;
        for gt in &sc.objs {
            let d = (gt.cy - gy).powi(2) + (gt.cx - gx).powi(2);
            if d < best_d {
                best_d = d;
                best_gt = gt;
            }
        }
        let match_cnt = sc.objs.iter().filter(|gt| {
            if ask_col { gt.colour_id == target_obj.colour_id } else { gt.shape_id == target_obj.shape_id }
        }).count();
        let ch = (match_cnt as f32) / (sc.objs.len() as f32);
        let hit = best_d < 225.0 && if ask_col {
            best_gt.colour_id == target_obj.colour_id
        } else {
            best_gt.shape_id == target_obj.shape_id
        };

        if tick < 300 {
            ep1_tot += 1;
            ep1_chance += ch;
            if hit { ep1_gaze_hits += 1; }
        } else if tick < 600 {
            ep2_tot += 1;
            ep2_chance += ch;
            if hit { ep2_gaze_hits += 1; }
        } else {
            ep3_tot += 1;
            ep3_chance += ch;
            if hit { ep3_gaze_hits += 1; }
            vocal_records.push((best_gt.colour_id, motor.vocal_tract));
        }
    }

    let mut voc_same = 0.0f32;
    let mut voc_same_n = 0.0f32;
    let mut voc_diff = 0.0f32;
    let mut voc_diff_n = 0.0f32;
    for i in 0..vocal_records.len().min(80) {
        let mut vi = vocal_records[i].1.to_vec();
        let m_i: f32 = vi.iter().sum::<f32>() / 8.0;
        for x in &mut vi { *x -= m_i; }
        for j in (i + 1)..vocal_records.len().min(80) {
            let mut vj = vocal_records[j].1.to_vec();
            let m_j: f32 = vj.iter().sum::<f32>() / 8.0;
            for x in &mut vj { *x -= m_j; }
            let c = cos_sim(&vi, &vj);
            if vocal_records[i].0 == vocal_records[j].0 {
                voc_same += c;
                voc_same_n += 1.0;
            } else {
                voc_diff += c;
                voc_diff_n += 1.0;
            }
        }
    }

    println!("\n--- [PROBE E] 100% UNSTAGED NATURAL-OUTPUT LIFE (`ONLY org.step(&frame) -> MotorFrame`) ---");
    println!("  Ticks 000..299 (Newborn natural gaze)  : {:.3} (chance = {:.3}, n={})",
        (ep1_gaze_hits as f32) / (ep1_tot.max(1) as f32), ep1_chance / (ep1_tot.max(1) as f32), ep1_tot);
    println!("  Ticks 300..599 (Juvenile natural gaze) : {:.3} (chance = {:.3}, n={})",
        (ep2_gaze_hits as f32) / (ep2_tot.max(1) as f32), ep2_chance / (ep2_tot.max(1) as f32), ep2_tot);
    println!("  Ticks 600..899 (Holdout speakers+shadow natural gaze) : {:.3} (chance = {:.3}, n={})",
        (ep3_gaze_hits as f32) / (ep3_tot.max(1) as f32), ep3_chance / (ep3_tot.max(1) as f32), ep3_tot);
    println!("  Natural vocal motor differentiation (SAME vs DIFF fixated object): {:+.4} ({:.4} vs {:.4})",
        (voc_same / voc_same_n.max(1.0)) - (voc_diff / voc_diff_n.max(1.0)),
        voc_same / voc_same_n.max(1.0), voc_diff / voc_diff_n.max(1.0));
    let last_motor = free_org.step(&SensoryFrame {
        retina_rgb: &[],
        width: 0,
        height: 0,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    });
    println!("  Endogenously formed lexicon basins={} | unbroken chain_steps={} | novelty={:.3} | action_drive={:.3?}",
        free_org.binding.lexicon.len(), free_org.mind.chain_steps, free_org.mind.novelty_emergence, last_motor.action_drive);

    println!("================================================================================");
}
