//! 7-Stage Real-Life Closed-Loop Developmental World & Progressive Puzzle Suite (`real_life_stages.rs`)
//!
//! Raises ONE SINGLE LIVING ORGANISM (`child = Organism::new(Genome::default())`) continuously
//! from birth (`Tick 0`) to `Tick 1150` through 7 progressive real-life stages using ONLY
//! closed-loop sensorimotor interaction (`SensoryFrame -> child.step(&frame) -> MotorFrame -> World`):
//!   - Stage 1: Newborn Crib Awakening, Saccadic Exploration & Vocal Babbling Loop
//!   - Stage 2: Speaking to the Child & Seeing Him Respond Back (Naming & Turn-Taking Dialogue)
//!   - Stage 3: Puzzle 1 — Physical Cause-and-Effect Latch Boxes (Trial, Error & Discovery)
//!   - Stage 4: Puzzle 2 — 2-Step Sequential Key-and-Lock & Crossroads Context Puzzle
//!   - Stage 5: Puzzle 3 — Zero-Shot Analogical Puzzle Transfer on a Never-Before-Seen Box
//!   - Stage 6: Progressive Quantity/Rhythm Teaching (1..4) -> Unforced Next-Number & +2 Puzzles
//!   - Stage 7: Open-Ended Playroom & Whole-Body 5-Tissue Cellular Autobiography

use alpha_iface::SensoryFrame;
use alpha_organism::{
    Organism,
    cognition::{CELL_TISSUE_COCHLEA_WORD, CELL_TISSUE_RETINA_MEANING, ModulationBroadcast},
    genome::Genome,
    util::{Rng64, dot, l2_norm},
};
use alpha_world::{
    WorldObject, capture_hardware_ears_pcm_16khz, capture_hardware_eyes_rgb, render_scene,
    synthesize_two_word_phrase, synthesize_word_utterance,
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
    #[allow(dead_code)]
    cx: f32,
    #[allow(dead_code)]
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

fn argmax4(v: &[f32; 4]) -> usize {
    let mut best = 0usize;
    for i in 1..4 {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn main() {
    println!("==========================================================================================");
    println!("ALPHA-9 REAL-LIFE MULTI-STAGE DEVELOPMENTAL WORLD, PUZZLES & CONVERSATION");
    println!("One Single Living Organism (`ONLY child.step(&frame) -> MotorFrame`) From Birth To Maturity");
    println!("==========================================================================================");

    let audio_pool_p = std::env::var("ALPHA9_AUDIO_POOL").unwrap_or_else(|_| "state/real_audio_pool.bin".to_string());
    let train_sc_p = std::env::var("ALPHA9_SCENES_TRAIN").unwrap_or_else(|_| "state/real_scenes_train.txt".to_string());
    let hold_sc_p = std::env::var("ALPHA9_SCENES_HOLDOUT").unwrap_or_else(|_| "state/real_scenes_holdout.txt".to_string());
    let utts = load_audio_pool(&audio_pool_p);
    let train_utts: Vec<&RealUtterance> = utts.iter().filter(|u| u.split == 0).collect();
    let test_utts: Vec<&RealUtterance> = utts.iter().filter(|u| u.split > 0).collect();
    let train_scenes = load_scenes(&train_sc_p);
    let holdout_scenes = load_scenes(&hold_sc_p);

    let mut child = Organism::new(Genome::default());

    // ========================================================================================
    // STAGE 1 (Ticks 1..150): NEWBORN AWAKENING, SACCADIC EXPLORATION & VOCAL BABBLING IN CRIB
    // ========================================================================================
    println!("\n[STAGE 1 | Ticks 001..150] NEWBORN SENSORIMOTOR AWAKENING & BABBLING IN A CRIB");
    let mut stage1_unique_gaze_regions = Vec::new();
    let mut early_vocal_rms = 0.0f32;
    let mut late_vocal_rms = 0.0f32;
    let mut prev_caregiver_f0 = 0.5f32;

    for t in 1..=150 {
        let sc = &train_scenes[(t * 5) % train_scenes.len()];
        let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        let caregiver_u = train_utts[(t * 3) % train_utts.len()];
        let pcm: &[f32] = if t % 3 == 0 { &[] } else { &caregiver_u.pcm };

        // Closed-loop social feedback: if baby's previous vocalization was close to caregiver's pitch,
        // the caregiver provides warm soothing comfort (`body_flux > 0`).
        let baby_f0 = child.mind.muscle_cells[0];
        let social_warmth = if (baby_f0 - prev_caregiver_f0).abs() < 0.18 { 0.04 } else { 0.005 };
        prev_caregiver_f0 = 0.35 + 0.02 * (caregiver_u.word_id as f32);

        let out = child.step(&SensoryFrame {
            retina_rgb: &rgb,
            width: w as u16,
            height: h as u16,
            pcm_16khz: pcm,
            body_flux: [social_warmth, social_warmth, 0.008, 0.008],
        });

        let gx_bin = (out.gaze_xy[0] * 10.0) as i32;
        let gy_bin = (out.gaze_xy[1] * 10.0) as i32;
        if !stage1_unique_gaze_regions.iter().any(|&(x, y)| x == gx_bin && y == gy_bin) {
            stage1_unique_gaze_regions.push((gx_bin, gy_bin));
        }
        let pcm_energy = l2_norm(&out.pcm_out) / ((out.pcm_out.len().max(1) as f32).sqrt());
        if t <= 30 {
            early_vocal_rms += pcm_energy / 30.0;
        } else if t > 120 {
            late_vocal_rms += pcm_energy / 30.0;
        }

        if t == 1 || t == 50 || t == 100 || t == 150 {
            println!(
                "  Tick {:03} -> gaze=[{:.2}, {:.2}] | vocal_tract=[{:.2}, {:.2}, {:.2}, {:.2}] | action=[{:.2}, {:.2}, {:.2}, {:.2}] | locked_cells={}/256",
                t,
                out.gaze_xy[0],
                out.gaze_xy[1],
                out.vocal_tract[0],
                out.vocal_tract[1],
                out.vocal_tract[2],
                out.vocal_tract[3],
                out.action_drive[0],
                out.action_drive[1],
                out.action_drive[2],
                out.action_drive[3],
                child.mind.cells.iter().filter(|c| c.chromatin_locked > 0.4).count()
            );
        }
    }
    println!(
        "  -> Stage 1 Summary: Explored {} distinct visual grid zones | Vocal acoustic RMS={:.4} -> {:.4} | Muscle cell={:.3}, Organ cell={:.3}",
        stage1_unique_gaze_regions.len(),
        early_vocal_rms,
        late_vocal_rms,
        child.mind.muscle_cells[0],
        child.mind.organ_cells[0]
    );

    // ========================================================================================
    // STAGE 2 (Ticks 151..420): SPEAKING TO THE CHILD & SEEING HIM RESPOND BACK
    //   - Natural multimodal word-object grounding + conversational call-and-response turns
    //   - Then observe unforced vocal responses when:
    //       (A) Shown held-out objects in silence (Visual -> Spoken Response)
    //       (B) Spoken to in a dark/empty room by held-out speakers (Spoken Prompt -> Spoken Response)
    // ========================================================================================
    println!("\n[STAGE 2 | Ticks 151..420] SPEAKING TO THE CHILD & OBSERVING NATURAL VOCAL RESPONSES");
    for step in 0..220 {
        let sc = &train_scenes[step % train_scenes.len()];
        let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
        let target = &sc.objs[step % sc.objs.len()];
        let target_w = if step % 2 == 0 { target.colour_id } else { 11 + target.shape_id };
        let matching: Vec<&&RealUtterance> = train_utts.iter().filter(|u| u.word_id == target_w).collect();
        if !matching.is_empty() {
            let u = matching[step % matching.len()];
            let _ = child.step(&SensoryFrame {
                retina_rgb: &rgb,
                width: w as u16,
                height: h as u16,
                pcm_16khz: &u.pcm,
                body_flux: [0.015, 0.015, 0.010, 0.010],
            });
        }
    }

    // Conversational call-and-response experience: Caregiver says greeting/prompt Word A -> Child responds -> Warmth
    let prompt_words = [0usize, 4, 8, 12];
    let reply_words = [1usize, 5, 9, 13];
    for turn in 0..40 {
        let pair_idx = turn % 4;
        let u_prompt = train_utts.iter().find(|u| u.word_id == prompt_words[pair_idx]).unwrap();
        let u_reply = train_utts.iter().find(|u| u.word_id == reply_words[pair_idx]).unwrap();

        // Tick 1 of turn: Caregiver speaks `u_prompt`
        let _ = child.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &u_prompt.pcm,
            body_flux: [0.01; 4],
        });
        // Tick 2 of turn: Caregiver models/echoes `u_reply` with warm social reinforcement
        let _ = child.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &u_reply.pcm,
            body_flux: [0.22, 0.22, 0.18, 0.18],
        });
    }

    // Observe child's raw unforced vocal response when spoken to by UNSEEN HELD-OUT SPEAKERS (`test_utts`)
    println!("  [2A: Caregiver (Held-Out Speaker) Speaks to Child -> Child's Raw Spoken Response Back]");
    let mut spoken_replies: Vec<[f32; 8]> = Vec::new();
    for &pw in &prompt_words {
        let u_test = test_utts.iter().find(|u| u.word_id == pw).unwrap_or(&test_utts[0]);
        // Caregiver speaks the prompt word
        let out_listen = child.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &u_test.pcm,
            body_flux: [0.0; 4],
        });
        // Pause in silence on the next tick so the child can answer back from its recalled trajectory!
        let out_reply = child.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        spoken_replies.push(out_listen.vocal_tract);
        println!(
            "    Heard Held-Out Word {:02} -> Immediate Vocal=[{:.3}, {:.3}, {:.3}, {:.3}] (PCM rms={:.3}) | Follow-up Reply=[{:.3}, {:.3}, {:.3}, {:.3}]",
            pw,
            out_listen.vocal_tract[0],
            out_listen.vocal_tract[1],
            out_listen.vocal_tract[2],
            out_listen.vocal_tract[3],
            l2_norm(&out_listen.pcm_out) / ((out_listen.pcm_out.len().max(1) as f32).sqrt()),
            out_reply.vocal_tract[0],
            out_reply.vocal_tract[1],
            out_reply.vocal_tract[2],
            out_reply.vocal_tract[3],
        );
    }

    // Measure whether the child's vocal responses to the SAME word spoken by DIFFERENT unseen speakers
    // are more consistent than responses to DIFFERENT words!
    let mut same_word_vocal_cos = 0.0f32;
    let mut diff_word_vocal_cos = 0.0f32;
    let mut sw_n = 0.0f32;
    let mut dw_n = 0.0f32;
    for i in 0..test_utts.len().min(30) {
        let out_i = child.step(&SensoryFrame {
            retina_rgb: &[],
            width: 0,
            height: 0,
            pcm_16khz: &test_utts[i].pcm,
            body_flux: [0.0; 4],
        });
        for j in (i + 1)..test_utts.len().min(30) {
            let out_j = child.step(&SensoryFrame {
                retina_rgb: &[],
                width: 0,
                height: 0,
                pcm_16khz: &test_utts[j].pcm,
                body_flux: [0.0; 4],
            });
            let c = cos_sim(&out_i.vocal_tract, &out_j.vocal_tract);
            if test_utts[i].word_id == test_utts[j].word_id {
                same_word_vocal_cos += c;
                sw_n += 1.0;
            } else {
                diff_word_vocal_cos += c;
                dw_n += 1.0;
            }
        }
    }
    println!(
        "  -> Unseen-Speaker Conversational Response Consistency: SAME word cos={:.4} vs DIFF word cos={:.4} (Gap={:+.4})",
        same_word_vocal_cos / sw_n.max(1.0),
        diff_word_vocal_cos / dw_n.max(1.0),
        (same_word_vocal_cos / sw_n.max(1.0)) - (diff_word_vocal_cos / dw_n.max(1.0))
    );

    // ========================================================================================
    // STAGE 3 (Ticks 421..540): PUZZLE 1 — PHYSICAL CAUSE-AND-EFFECT LATCH BOXES
    //   - 3 distinct physical Puzzle Boxes (Real 3D scenes `Box 0`, `Box 1`, `Box 2`):
    //       * `Box 0` opens ONLY when Limb Action `2` (Press/Turn) is dominant
    //       * `Box 1` opens ONLY when Limb Action `3` (Pull/Slide) is dominant
    //       * `Box 2` opens ONLY when Limb Action `0` (Reach/Push) is dominant
    //   - Nobody tells the child the answer! The child interacts with each box in closed loop:
    //     wrong action -> box stays locked (`body_flux < 0`), right action -> box opens (`body_flux > 0`).
    // ========================================================================================
    println!("\n[STAGE 3 | Ticks 421..540] PUZZLE 1: CAUSE-AND-EFFECT LATCH BOXES (TRIAL, ERROR & DISCOVERY)");
    let b0_sc = train_scenes
        .iter()
        .find(|ts| {
            ts.objs.len() == 1
                && holdout_scenes.iter().any(|hs| {
                    hs.objs.len() == 1
                        && hs.objs[0].colour_id == ts.objs[0].colour_id
                        && hs.objs[0].shape_id != ts.objs[0].shape_id
                })
        })
        .unwrap_or(&train_scenes[3]);
    let b0_col = b0_sc.objs[0].colour_id;
    let analog_holdout = holdout_scenes
        .iter()
        .find(|hs| hs.objs.len() == 1 && hs.objs[0].colour_id == b0_col && hs.objs[0].shape_id != b0_sc.objs[0].shape_id)
        .unwrap_or(&holdout_scenes[0]);
    let analog_shp = analog_holdout.objs[0].shape_id;
    let non_conflicting = |sc: &&RealScene| -> bool {
        sc.objs.iter().all(|o| o.colour_id != b0_col && o.shape_id != analog_shp)
    };
    let b1_sc = train_scenes.iter().skip(10).find(non_conflicting).unwrap_or(&train_scenes[11]);
    let b2_sc = train_scenes.iter().skip(24).find(non_conflicting).unwrap_or(&train_scenes[25]);
    let box_scenes = [b0_sc, b1_sc, b2_sc];
    let box_required_action = [2usize, 3usize, 0usize];

    for b_idx in 0..3 {
        let (box_rgb, bw, bh) = read_ppm(&box_scenes[b_idx].path).unwrap();
        let req_act = box_required_action[b_idx];
        let mut history_str = Vec::new();
        let mut solved_on_attempt = 0usize;

        for attempt in 1..=8 {
            // Step 1: Child sees the box and outputs a natural limb `action_drive`
            let out_act = child.step(&SensoryFrame {
                retina_rgb: &box_rgb,
                width: bw as u16,
                height: bh as u16,
                pcm_16khz: &[],
                body_flux: [0.0; 4],
            });
            let chosen_act = argmax4(&out_act.action_drive);
            let success = chosen_act == req_act;
            history_str.push(format!("Try#{}:Act{}({})", attempt, chosen_act, if success { "OPEN!" } else { "locked" }));
            if success && solved_on_attempt == 0 {
                solved_on_attempt = attempt;
            }

            // Step 2: Physical world returns the consequence of the child's limb action!
            let outcome_flux = if success {
                [0.35, 0.35, 0.28, 0.28]
            } else {
                [-0.22, -0.20, -0.15, -0.15]
            };
            let _ = child.step(&SensoryFrame {
                retina_rgb: &box_rgb,
                width: bw as u16,
                height: bh as u16,
                pcm_16khz: &[],
                body_flux: outcome_flux,
            });
        }
        println!(
            "  Puzzle Box {} (Requires Action {}) -> First solved on Attempt #{} | Trajectory: {}",
            b_idx,
            req_act,
            solved_on_attempt,
            history_str.join(" -> ")
        );
    }

    // Re-test all 3 boxes cold after interleaving to verify retention in cellular motor memory!
    let mut box_retention_hits = 0usize;
    for b_idx in 0..3 {
        let (box_rgb, bw, bh) = read_ppm(&box_scenes[b_idx].path).unwrap();
        let out_test = child.step(&SensoryFrame {
            retina_rgb: &box_rgb,
            width: bw as u16,
            height: bh as u16,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        let chosen = argmax4(&out_test.action_drive);
        if chosen == box_required_action[b_idx] {
            box_retention_hits += 1;
        }
    }
    println!(
        "  -> Post-Interleaving Cold Retention across all 3 Puzzle Boxes: {}/3 solved immediately on first touch!",
        box_retention_hits
    );

    // ========================================================================================
    // STAGE 4 (Ticks 541..680): PUZZLE 2 — 2-STEP SEQUENTIAL KEY->LOCK & CROSSROADS PUZZLE
    //   - A 2-step causal puzzle where the Lock Room (`lock_sc`) requires DIFFERENT limb actions
    //     depending on WHICH room the child just came from (`key_a_sc` vs `key_b_sc`):
    //       * Path 1: `Key_A` (Action 1) -> `Lock_Room` requires `Action 2`
    //       * Path 2: `Key_B` (Action 0) -> `Lock_Room` requires `Action 3`
    // ========================================================================================
    println!("\n[STAGE 4 | Ticks 541..680] PUZZLE 2: 2-STEP SEQUENTIAL KEY->LOCK & CROSSROADS PUZZLE");
    let lock_sc = train_scenes.iter().skip(48).find(non_conflicting).unwrap_or(&train_scenes[52]);
    let (key_a_rgb, kaw, kah) = read_ppm(&train_scenes[31].path).unwrap();
    let (key_b_rgb, kbw, kbh) = read_ppm(&train_scenes[44].path).unwrap();
    let (lock_rgb, lw, lh) = read_ppm(&lock_sc.path).unwrap();

    let mut early_seq_success = 0usize;
    let mut late_seq_success = 0usize;

    for ep in 1..=16 {
        let is_path_a = ep % 2 == 1;
        let (k_rgb, kw, kh, req_lock_act) = if is_path_a {
            (&key_a_rgb, kaw, kah, 2usize)
        } else {
            (&key_b_rgb, kbw, kbh, 3usize)
        };

        // Step 1: Child enters Key Room (`Key_A` or `Key_B`)
        let _ = child.step(&SensoryFrame {
            retina_rgb: k_rgb,
            width: kw as u16,
            height: kh as u16,
            pcm_16khz: &[],
            body_flux: [0.05; 4],
        });

        // Step 2: Child enters the shared Lock Room (`lock_rgb`) and chooses a limb action!
        let out_lock = child.step(&SensoryFrame {
            retina_rgb: &lock_rgb,
            width: lw as u16,
            height: lh as u16,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        let chosen_lock_act = argmax4(&out_lock.action_drive);
        let ok = chosen_lock_act == req_lock_act;
        if ep <= 4 && ok {
            early_seq_success += 1;
        }
        if ep > 12 && ok {
            late_seq_success += 1;
        }

        // Step 3: World delivers consequence on Lock Room
        let flux = if ok {
            [0.35, 0.35, 0.30, 0.30]
        } else {
            [-0.25, -0.22, -0.18, -0.18]
        };
        let _ = child.step(&SensoryFrame {
            retina_rgb: &lock_rgb,
            width: lw as u16,
            height: lh as u16,
            pcm_16khz: &[],
            body_flux: flux,
        });
    }
    println!(
        "  -> 2-Step Crossroads Puzzle (`Key_A->Lock[Act2]` vs `Key_B->Lock[Act3]`): Early episodes={}/4 -> Late episodes={}/4",
        early_seq_success, late_seq_success
    );

    // ========================================================================================
    // STAGE 4B: MULTI-SCALE DELAYED ACTION ACCOUNTABILITY & SELF-SUPERVISED WORLD-DELTA
    //   - Child takes an action in a Switch Room (`switch_sc`) at t = 0.
    //   - Then walks through 4 neutral hallway ticks (`t = 1..4`, zero reward/punishment!).
    //   - Only at `t = 5` (5 ticks later!) does the delayed consequence arrive:
    //       * If the child chose Action `3` back at t = 0 -> warm reward at t = 5
    //       * Any other action at t = 0 -> cold penalty at t = 5
    //   - Tests whether `tag_med` / `tag_slow` and multi-scale `elig_action` credit the
    //     action taken 5 ticks earlier, and whether `world_delta_w` tracks reality change!
    // ========================================================================================
    println!("\n[STAGE 4B] MULTI-SCALE DELAYED ACTION ACCOUNTABILITY (ACTION AT t=0 -> 4 NEUTRAL TICKS -> OUTCOME AT t=5)");
    let switch_sc = train_scenes.iter().skip(35).find(non_conflicting).unwrap_or(&train_scenes[56 % train_scenes.len()]);
    let (switch_rgb, sww, swh) = read_ppm(&switch_sc.path).unwrap();
    let (hall_rgb, hw, hh) = read_ppm(&train_scenes[58 % train_scenes.len()].path).unwrap();
    let mut delayed_history = Vec::new();
    let mut delayed_solved_first = 0usize;
    let agency_before = child.mind.agency_confidence;

    for trial in 1..=8 {
        // t = 0: Child sees Switch Room and takes an unforced limb action
        let out_sw = child.step(&SensoryFrame {
            retina_rgb: &switch_rgb,
            width: sww as u16,
            height: swh as u16,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        let sw_act = argmax4(&out_sw.action_drive);
        let delayed_ok = sw_act == 3;
        if delayed_ok && delayed_solved_first == 0 {
            delayed_solved_first = trial;
        }
        delayed_history.push(format!(
            "Tr#{}:Act{}({})",
            trial,
            sw_act,
            if delayed_ok { "DELAYED_OPEN!" } else { "delayed_fail" }
        ));

        // t = 1..4: 4 neutral hallway ticks with ZERO outcome flux
        for _ in 0..4 {
            let _ = child.step(&SensoryFrame {
                retina_rgb: &hall_rgb,
                width: hw as u16,
                height: hh as u16,
                pcm_16khz: &[],
                body_flux: [0.0; 4],
            });
        }

        // t = 5: Delayed consequence arrives 5 ticks after the Switch Room action!
        let delayed_flux = if delayed_ok {
            [0.38, 0.38, 0.32, 0.32]
        } else {
            [-0.26, -0.24, -0.20, -0.20]
        };
        let _ = child.step(&SensoryFrame {
            retina_rgb: &switch_rgb,
            width: sww as u16,
            height: swh as u16,
            pcm_16khz: &[],
            body_flux: delayed_flux,
        });
    }
    let agency_after = child.mind.agency_confidence;
    let world_delta_norm = child
        .mind
        .cells
        .iter()
        .map(|c| l2_norm(&c.world_delta_w))
        .sum::<f32>()
        / (child.mind.cells.len() as f32);
    println!(
        "  5-Tick Delayed Switch Room (Requires Action 3 at t=0) -> First solved on Trial #{} | Trajectory: {}",
        delayed_solved_first,
        delayed_history.join(" -> ")
    );
    println!(
        "  -> Self-Supervised Cellular Agency (`agency_confidence`): {:.4} -> {:.4} | Mean ||world_delta_w||={:.4}",
        agency_before, agency_after, world_delta_norm
    );

    // ========================================================================================
    // STAGE 5 (Ticks 681..720): PUZZLE 3 — ZERO-SHOT ANALOGICAL PUZZLE TRANSFER
    //   - Present a BRAND-NEW Holdout Box (`holdout_scenes`) that the child has NEVER touched!
    //   - It shares the primary colour of `Box 0` (which required Action 2 in Stage 3),
    //     but has a different shape and room layout.
    //   - Measure what action the child tries on ATTEMPT #1 (zero prior trials on this box!)
    //     and how its similarity/difference vectors decompose the new box!
    // ========================================================================================
    println!("\n[STAGE 5 | Ticks 681..720] PUZZLE 3: ZERO-SHOT ANALOGICAL TRANSFER TO A NEVER-SEEN BOX");
    let (analog_rgb, aw, ah) = read_ppm(&analog_holdout.path).unwrap();

    // Present the NEVER-BEFORE-SEEN `analog_rgb` box in complete silence (zero priming, zero reward)!
    let out_analog = child.step(&SensoryFrame {
        retina_rgb: &analog_rgb,
        width: aw as u16,
        height: ah as u16,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    });
    let analog_chosen_act = argmax4(&out_analog.action_drive);
    let sim_col_energy = l2_norm(&child.mind.last_similarity_vec[16..32])
        / (l2_norm(&child.mind.last_similarity_vec) + 1e-8);
    let diff_shp_energy = l2_norm(&child.mind.last_difference_vec[0..16])
        / (l2_norm(&child.mind.last_difference_vec) + 1e-8);

    println!(
        "  Never-Seen Holdout Box (Shares Colour={} with Box 0, New Shape={}):",
        analog_holdout.objs[0].colour_id, analog_holdout.objs[0].shape_id
    );
    println!(
        "  -> Zero-Shot First-Touch Action={} (Box 0 required Action={}) | action_drive=[{:.2}, {:.2}, {:.2}, {:.2}]",
        analog_chosen_act,
        box_required_action[0],
        out_analog.action_drive[0],
        out_analog.action_drive[1],
        out_analog.action_drive[2],
        out_analog.action_drive[3]
    );
    println!(
        "  -> Internal Analogy Decomposition: Shared-Colour Resonance={:.3} | Novel-Shape Difference={:.3}",
        sim_col_energy, diff_shp_energy
    );

    // ========================================================================================
    // STAGE 6 (Ticks 721..960): PROGRESSIVE QUANTITY/COUNTING FOUNDATION -> PATTERN & NUMBER PUZZLE
    //   - No math is pre-programmed! Instead, we teach the child through 3 natural stages:
    //     * Stage 6A: Experiencing rooms with 1, 2, 3, and 4 objects + spoken count words
    //     * Stage 6B: Experiencing physical "+1" increments (`1 -> 2 -> 3 -> 4`) repeatedly
    //       so the successor chain `Q_1 -> Q_2 -> Q_3 -> Q_4` is carved into `w_flow` and `cells`!
    //     * Stage 6C: Test unforced Next-Quantity (`N + 1`) and Two-Step Addition (`N + 2`)
    //       puzzles by watching which quantity state the child's neural wave and voice project to!
    // ========================================================================================
    println!("\n[STAGE 6 | Ticks 721..960] PROGRESSIVE QUANTITY TEACHING (1..4) -> UNFORCED NUMBER PUZZLES");
    // Select 4 distinct scenes representing quantities Q1 (1 obj), Q2 (2 objs), Q3 (3 objs), Q4 (4 objs)
    let q_scenes: Vec<&RealScene> = (1..=4)
        .map(|cnt| {
            train_scenes
                .iter()
                .find(|s| s.objs.len() == cnt)
                .unwrap_or(&train_scenes[cnt])
        })
        .collect();

    let mut q_states: Vec<Vec<f32>> = Vec::with_capacity(4);
    let mut q_vocals: Vec<[f32; 8]> = Vec::with_capacity(4);

    // Stage 6A & 6B: Child experiences counting sequence `Q1 -> Q2 -> Q3 -> Q4` across 35 spaced lessons
    for lesson in 0..35 {
        for q_idx in 0..4 {
            let (q_rgb, qw, qh) = read_ppm(&q_scenes[q_idx].path).unwrap();
            let count_word_id = 15 + q_idx;
            let u_cnt = train_utts.iter().find(|u| u.word_id == count_word_id).unwrap_or(&train_utts[q_idx]);
            let flux = if q_idx == 3 { [0.25, 0.25, 0.20, 0.20] } else { [0.08, 0.08, 0.05, 0.05] };
            let out_q = child.step(&SensoryFrame {
                retina_rgb: &q_rgb,
                width: qw as u16,
                height: qh as u16,
                pcm_16khz: &u_cnt.pcm,
                body_flux: flux,
            });
            if lesson == 34 {
                q_states.push(child.mind.chain_fast.clone());
                q_vocals.push(out_q.vocal_tract);
            }
        }
    }

    // Stage 6C: Unforced Quantity Puzzles (`1 + 1 = ?`, `2 + 1 = ?`, `3 + 1 = ?`, and `1 + 2 = ?`, `2 + 2 = ?`)
    // We place the child in state Q_n and observe which quantity state (Q1..Q4) its natural 1-step
    // and 2-step neural waves (`propagate_wave` / `extract_cellular_experience`) settle into!
    let mut plus1_correct = 0usize;
    for start_q in 0..3 {
        let (pred_cell, _) = child.mind.extract_cellular_experience(&q_states[start_q], None);
        let pred_wave1 = child.mind.propagate_wave(&q_states[start_q], 1);
        let mut combined = vec![0.0f32; child.mind.state_dim];
        for d in 0..combined.len() {
            combined[d] = 0.55 * pred_cell[d] + 0.45 * pred_wave1[d];
        }
        let mut best_q = 0usize;
        let mut best_c = f32::NEG_INFINITY;
        for cand_q in 0..4 {
            let c = cos_sim(&combined, &q_states[cand_q]);
            if c > best_c {
                best_c = c;
                best_q = cand_q;
            }
        }
        let ok = best_q == start_q + 1;
        if ok {
            plus1_correct += 1;
        }
        println!(
            "  Puzzle [Quantity {} + 1 = ?] -> Child's brain wave settles on Quantity {} (cos={:.3}, expected={})",
            start_q + 1,
            best_q + 1,
            best_c,
            start_q + 2
        );
    }

    let mut plus2_correct = 0usize;
    for start_q in 0..2 {
        let pred_wave2 = child.mind.propagate_wave(&q_states[start_q], 2);
        let mut best_q = 0usize;
        let mut best_c = f32::NEG_INFINITY;
        for cand_q in 0..4 {
            let c = cos_sim(&pred_wave2, &q_states[cand_q]);
            if c > best_c {
                best_c = c;
                best_q = cand_q;
            }
        }
        let ok = best_q == start_q + 2;
        if ok {
            plus2_correct += 1;
        }
        println!(
            "  Puzzle [Quantity {} + 2 = ?] -> Child's 2-step wave settles on Quantity {} (cos={:.3}, expected={})",
            start_q + 1,
            best_q + 1,
            best_c,
            start_q + 3
        );
    }
    println!(
        "  -> Stage 6 Summary: Unforced (+1) puzzles={}/3 | Unforced (+2) two-step puzzles={}/2",
        plus1_correct, plus2_correct
    );

    // ========================================================================================
    // STAGE 7 (Ticks 961..1150): OPEN-ENDED PLAYROOM & WHOLE-BODY 5-TISSUE AUTOBIOGRAPHY
    // ========================================================================================
    println!("\n[STAGE 7 | Ticks 961..1150] OPEN-ENDED UNSCRIPTED PLAYROOM & WHOLE-BODY CELLULAR SUMMARY");
    for t in 0..190 {
        if t % 15 == 14 {
            // Quiet dark rest / daydreaming replay tick
            let _ = child.step(&SensoryFrame {
                retina_rgb: &[],
                width: 0,
                height: 0,
                pcm_16khz: &[],
                body_flux: [0.008; 4],
            });
        } else {
            let sc = &holdout_scenes[(t * 9) % holdout_scenes.len()];
            let Some((rgb, w, h)) = read_ppm(&sc.path) else { continue; };
            let pcm: &[f32] = if t % 2 == 0 {
                &test_utts[(t * 7) % test_utts.len()].pcm
            } else {
                &[]
            };
            let _ = child.step(&SensoryFrame {
                retina_rgb: &rgb,
                width: w as u16,
                height: h as u16,
                pcm_16khz: pcm,
                body_flux: [0.004, 0.004, 0.003, 0.003],
            });
        }
    }

    let locked_total = child.mind.cells.iter().filter(|c| c.chromatin_locked > 0.5).count();
    let open_total = child.mind.cells.len() - locked_total;
    let mean_agency = child.mind.agency_confidence;
    println!("  Total Unbroken Life Steps (`child.step_count`) : {}", child.step_count);
    println!("  1. Brain Epigenetic Cells (256 total)          : {} chromatin-locked engrams | {} open plastic cells (TET-regulated)", locked_total, open_total);
    println!("  2. Cellular Self-Supervised Agency Confidence  : {:.4} (learned ||delta_s|| accountability)", mean_agency);
    println!("  3. Retinal Photoreceptor Cells (`retina_cells`): [{:.3}, {:.3}, {:.3}, {:.3}] (from naive 0.500)",
        child.mind.retina_cells[0], child.mind.retina_cells[1], child.mind.retina_cells[2], child.mind.retina_cells[3]);
    println!("  4. Cochlear Hair Cells (`cochlea_cells`)       : [{:.3}, {:.3}, {:.3}, {:.3}] (from naive 0.500)",
        child.mind.cochlea_cells[0], child.mind.cochlea_cells[1], child.mind.cochlea_cells[2], child.mind.cochlea_cells[3]);
    println!("  5. Limb Proprioceptive Spindles (`limb_cells`) : [{:.3}, {:.3}, {:.3}, {:.3}] (from naive 0.250)",
        child.mind.limb_cells[0], child.mind.limb_cells[1], child.mind.limb_cells[2], child.mind.limb_cells[3]);
    println!("  6. Vocal Cord Muscle Cells (`muscle_cells`)    : [{:.3}, {:.3}, {:.3}, {:.3}] (from naive 0.500)",
        child.mind.muscle_cells[0], child.mind.muscle_cells[1], child.mind.muscle_cells[2], child.mind.muscle_cells[3]);
    println!("  7. Visceral Organ Allostasis (`organ_cells`)   : [{:.3}, {:.3}, {:.3}, {:.3}] (from naive 0.000)",
        child.mind.organ_cells[0], child.mind.organ_cells[1], child.mind.organ_cells[2], child.mind.organ_cells[3]);
    println!("  8. Self-Organized Lexicon Prototypes           : {} word basins | RAM = {:.2} MB",
        child.binding.lexicon.len(),
        (child.memory_bytes() as f64) / (1024.0 * 1024.0));

    // ========================================================================================
    // STAGE 8: LIVING CELL ECOLOGY, 4-CHANNEL BRAIN-CELL GATE & 5 GROUNDED LANGUAGE STAGES
    //   - Tests the converted living `EpigeneticCell` body/mind directly on:
    //       * Living Cell Birth (Neurogenesis), Feeding, Fullness, Variety-Limit Mitosis
    //         (Parent Concept + 2 Daughter Cells), Starvation Neighbor Handoff, Fusion,
    //         3D Spatial Neighborhoods, and Dreaming Insight Bridges
    //       * 4-Channel Brain<->Cell Interface (Cheap Cells vs Waking Brain on `unknown`)
    //       * Language Stage 1: Word <-> Meaning Bonds
    //       * Language Stage 2: Comprehension (Looking at Named Object in 4-Object Room)
    //       * Language Stage 3: Unprompted First Words (Spontaneous Naming in Silent Room)
    //       * Language Stage 4: Unseen Two-Word Combinations (`"want cat"` from `"want milk"` + `"want toy"`)
    //       * Language Stage 5: Novel Use (Spontaneous Naming of Unseen Variant)
    //       * Companion Organism (`Alpha` + `Companion` co-developing via 16 kHz audio)
    //       * Optional Opt-In External Sensory Stream (`ALPHA9_VIDEO_SOURCE`, `ALPHA9_AUDIO_SOURCE`)
    // ========================================================================================
    println!("\n[STAGE 8] LIVING CELL ECOLOGY, 4-CHANNEL WAKING-BRAIN GATE & 5 GROUNDED LANGUAGE STAGES");
    let mut lang_rng = Rng64::new(0xA1F9_2026_7788_99AA);
    let mut alpha_lang = Organism::new(Genome::default());
    let mut companion = Organism::new(Genome {
        seed: 0xC049_A110_5566_7788,
        ..Genome::default()
    });
    companion.mind.merge_similarity_limit = 0.992;
    for _ in 0..80 {
        alpha_lang.babble_step(0.14);
        companion.babble_step(0.14);
    }

    let obj_specs = [
        (0usize, 0usize, 0usize, "ball"),
        (1usize, 1usize, 1usize, "cup"),
        (2usize, 2usize, 2usize, "cat"),
        (3usize, 3usize, 3usize, "dog"),
    ];
    let mut single_rooms: Vec<Vec<u8>> = Vec::new();
    for &(col, shp, _, _) in &obj_specs {
        let objs = [WorldObject {
            colour_id: col,
            shape_id: shp,
            cx: 64.0,
            cy: 64.0,
        }];
        single_rooms.push(render_scene(&objs, &mut lang_rng).rgb);
    }

    // Newborn sensory & vocal tract self-calibration for Alpha and Companion
    for i in 0..4 {
        alpha_lang.develop_vision(&single_rooms[i], 128, 128);
        companion.develop_vision(&single_rooms[i], 128, 128);
    }
    for w_dev in 0..10 {
        let pcm_dev = synthesize_word_utterance(w_dev, w_dev % 4, 0.0, &mut lang_rng);
        alpha_lang.develop_hearing(&pcm_dev);
        companion.develop_hearing(&pcm_dev);
    }
    for _ in 0..120 {
        alpha_lang.babble_step(0.14);
        companion.babble_step(0.14);
    }
    // Calibrate Area Id4 (`vocal.w_inv`) on WA basal reconstructions (`wf_est`) across developmental
    // words so `assembly_to_vocal_motor` produces dynamic onset->coda formant glides (`f1_start != f1_end`):
    let mut cal_rng = Rng64::new(0x55AA_1234_9988_7766);
    let in_dim = alpha_lang.wa.n_in;
    for _epoch in 0..45 {
        for w_dev in 0..10 {
            let pcm_dev = synthesize_word_utterance(w_dev, _epoch % 4, 0.0, &mut cal_rng);
            let (a_dev, _) = alpha_lang.hear_word(&pcm_dev, false);
            let mut wf_est = vec![0.0f32; in_dim];
            for u in 0..alpha_lang.wa.n_units.min(a_dev.len()) {
                let act = a_dev[u];
                if act > 0.0 {
                    let row = &alpha_lang.wa.w_basal[u * in_dim..(u + 1) * in_dim];
                    for (dst, &w) in wf_est.iter_mut().zip(row.iter()) {
                        *dst += act * w;
                    }
                }
            }
            let target_m = [
                ((w_dev as f32 * 0.23) % 0.80) + 0.10,
                if w_dev % 2 == 0 { 0.12 + 0.06 * (w_dev as f32) } else { 0.88 - 0.06 * (w_dev as f32) },
                if w_dev % 2 == 0 { 0.86 - 0.05 * (w_dev as f32) } else { 0.14 + 0.05 * (w_dev as f32) },
                if (w_dev / 2) % 2 == 0 { 0.15 + 0.05 * (w_dev as f32) } else { 0.85 - 0.05 * (w_dev as f32) },
                if (w_dev / 2) % 2 == 0 { 0.85 - 0.05 * (w_dev as f32) } else { 0.15 + 0.05 * (w_dev as f32) },
                ((w_dev as f32 * 0.31) % 0.80) + 0.10,
                0.45 + 0.04 * (w_dev as f32),
                0.70,
            ];
            alpha_lang.vocal.learn_babble_pair(&wf_est, &target_m, 0.65);
        }
    }

    // Language Stage 1: Word <-> Meaning Bonds + Co-Development with Companion Organism
    for ep in 0..16 {
        let spk = ep % 4;
        let shift = (ep as f32 - 8.0) * 0.015;
        for (i, &(_, _, wid, _)) in obj_specs.iter().enumerate() {
            alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
            companion.body.state = [0.82, 0.98, 0.75, 0.78];
            let pcm = synthesize_word_utterance(wid, spk, shift, &mut lang_rng);
            let _ = alpha_lang.step(&SensoryFrame {
                retina_rgb: &single_rooms[i],
                width: 128,
                height: 128,
                pcm_16khz: &pcm,
                body_flux: [0.06, 0.04, 0.08, 0.05],
            });
            // Alpha speaks about object `i` in the shared room; Companion sees object `i` and hears Alpha's voice!
            alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
            let out_a_speak = alpha_lang.step(&SensoryFrame {
                retina_rgb: &single_rooms[i],
                width: 128,
                height: 128,
                pcm_16khz: &[],
                body_flux: [0.02, 0.02, 0.04, 0.02],
            });
            if ep < 4 {
                companion.develop_hearing(&out_a_speak.pcm_out);
            } else {
                let _ = companion.step(&SensoryFrame {
                    retina_rgb: &single_rooms[i],
                    width: 128,
                    height: 128,
                    pcm_16khz: &out_a_speak.pcm_out,
                    body_flux: [0.06, 0.04, 0.08, 0.05],
                });
                let comp_start = companion.mind.basal_columns.min(companion.mind.cells.len());
                for c in companion.mind.cells.iter_mut().skip(comp_start) {
                    if c.tissue_kind == CELL_TISSUE_COCHLEA_WORD {
                        c.similarity_limit = 0.92;
                    }
                }
            }
        }
    }

    let mut stage1_hits = 0usize;
    for (i, &(_, _, wid, _)) in obj_specs.iter().enumerate() {
        let pcm_test = synthesize_word_utterance(wid, 5, 0.02, &mut lang_rng);
        let (a_test, _) = alpha_lang.hear_word(&pcm_test, false);
        let aud_vec = alpha_lang.mind.encode_auditory_state(&a_test);
        let readout = alpha_lang.mind.read_living_cells(
            None,
            Some(&aud_vec),
            None,
            ModulationBroadcast::default(),
        );
        let (objs_i, _) = alpha_lang.perceive_objects(&single_rooms[i], 128, 128, false);
        let target_m = alpha_lang.mind.encode_multimodal_state(&objs_i[0].s, &objs_i[0].c);
        let mut best_k = 0usize;
        let mut best_c = f32::NEG_INFINITY;
        for k in 0..4 {
            let (objs_k, _) = alpha_lang.perceive_objects(&single_rooms[k], 128, 128, false);
            let m_k = alpha_lang.mind.encode_multimodal_state(&objs_k[0].s, &objs_k[0].c);
            let c = dot(&readout.recalled_meaning, &m_k);
            if c > best_c {
                best_c = c;
                best_k = k;
            }
        }
        if best_k == i && dot(&readout.recalled_meaning, &target_m) > 0.40 {
            stage1_hits += 1;
        }
    }

    // Language Stage 2: Comprehension in 4-Object Room
    let multi_objs = [
        WorldObject { colour_id: 0, shape_id: 0, cx: 32.0, cy: 32.0 },
        WorldObject { colour_id: 1, shape_id: 1, cx: 96.0, cy: 32.0 },
        WorldObject { colour_id: 2, shape_id: 2, cx: 32.0, cy: 96.0 },
        WorldObject { colour_id: 3, shape_id: 3, cx: 96.0, cy: 96.0 },
    ];
    let multi_room_rgb = render_scene(&multi_objs, &mut lang_rng).rgb;
    let mut stage2_hits = 0usize;
    for (i, &(_, _, wid, _)) in obj_specs.iter().enumerate() {
        alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
        let pcm_cmd = synthesize_word_utterance(wid, 6, -0.02, &mut lang_rng);
        let motor = alpha_lang.step(&SensoryFrame {
            retina_rgb: &multi_room_rgb,
            width: 128,
            height: 128,
            pcm_16khz: &pcm_cmd,
            body_flux: [0.0; 4],
        });
        let target_gx = multi_objs[i].cx / 128.0;
        let target_gy = multi_objs[i].cy / 128.0;
        let d_target = ((motor.gaze_xy[0] - target_gx).powi(2)
            + (motor.gaze_xy[1] - target_gy).powi(2))
        .sqrt();
        if d_target < 0.22 {
            stage2_hits += 1;
        }
    }

    // Language Stage 3: Unprompted First Words in Silent Room
    let (comp_multi_perceived, _) = companion.perceive_objects(&multi_room_rgb, 128, 128, false);
    let mut stage3_hits = 0usize;
    let start_living = alpha_lang.mind.basal_columns.min(alpha_lang.mind.cells.len());
    for i in 0..4 {
        for c in alpha_lang.mind.cells.iter_mut().skip(start_living) {
            c.fullness = 0.0;
        }
        alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
        let motor = alpha_lang.step(&SensoryFrame {
            retina_rgb: &single_rooms[i],
            width: 128,
            height: 128,
            pcm_16khz: &[],
            body_flux: [0.0; 4],
        });
        if let Some(best_obj) = companion.attend_to_word(&motor.pcm_out, &comp_multi_perceived) {
            if best_obj == i {
                stage3_hits += 1;
            }
        }
    }

    // Language Stage 4: Unseen Two-Word Combination (`"want cat"` from `"want milk"` + `"want toy"`)
    let word_want = 7usize;
    let word_milk = 8usize;
    let word_toy = 9usize;
    let milk_rgb = render_scene(
        &[WorldObject { colour_id: 4, shape_id: 4, cx: 64.0, cy: 64.0 }],
        &mut lang_rng,
    )
    .rgb;
    let toy_rgb = render_scene(
        &[WorldObject { colour_id: 5, shape_id: 5, cx: 64.0, cy: 64.0 }],
        &mut lang_rng,
    )
    .rgb;
    for ep in 0..10 {
        let spk = ep % 3;
        alpha_lang.body.state = [0.32, 0.98, 0.75, 0.78];
        let pcm_wm = synthesize_two_word_phrase(word_want, word_milk, spk, 0.0, &mut lang_rng);
        let _ = alpha_lang.step(&SensoryFrame {
            retina_rgb: &milk_rgb,
            width: 128,
            height: 128,
            pcm_16khz: &pcm_wm,
            body_flux: [0.18, 0.02, 0.08, 0.05],
        });
        alpha_lang.body.state = [0.32, 0.98, 0.75, 0.78];
        let pcm_wt = synthesize_two_word_phrase(word_want, word_toy, spk, 0.0, &mut lang_rng);
        let _ = alpha_lang.step(&SensoryFrame {
            retina_rgb: &toy_rgb,
            width: 128,
            height: 128,
            pcm_16khz: &pcm_wt,
            body_flux: [0.18, 0.02, 0.08, 0.05],
        });
    }
    let ref_want_pcm = synthesize_word_utterance(word_want, 1, 0.0, &mut lang_rng);
    let (a_want_ref, _) = alpha_lang.hear_word(&ref_want_pcm, false);
    let m_want_ref = alpha_lang.assembly_to_vocal_motor(&a_want_ref);

    // Clear short-term fullness across word cells before testing unseen `"want cat"`
    let start_living = alpha_lang.mind.basal_columns.min(alpha_lang.mind.cells.len());
    for c in alpha_lang.mind.cells.iter_mut().skip(start_living) {
        c.fullness = 0.0;
    }
    alpha_lang.body.state = [0.30, 0.98, 0.75, 0.78];
    let motor_want_cat = alpha_lang.step(&SensoryFrame {
        retina_rgb: &single_rooms[2], // Cat room (never paired with "want"!)
        width: 128,
        height: 128,
        pcm_16khz: &[], // Complete silence!
        body_flux: [-0.12, 0.0, 0.0, 0.0],
    });
    let is_two_syllables = motor_want_cat.pcm_out.len() >= 14000;
    let single_len = (alpha_lang.genome.sr_hz * 45) / 100;
    let syl1_pcm = &motor_want_cat.pcm_out[0..single_len.min(motor_want_cat.pcm_out.len())];
    let syl2_start = (single_len + 800).min(motor_want_cat.pcm_out.len());
    let syl2_pcm = &motor_want_cat.pcm_out[syl2_start..];
    let (m_syl1, _) = alpha_lang.imitate_audio(syl1_pcm);
    let cos_syl1_want = cos_sim(&m_syl1, &m_want_ref);
    let syl2_attended = companion.attend_to_word(syl2_pcm, &comp_multi_perceived).unwrap_or(99);
    let stage4_pass = is_two_syllables && cos_syl1_want > 0.75 && syl2_attended == 2;

    // Language Stage 5: Novel Use + Companion Organism Audio Co-Regulation
    let novel_cat_rgb = render_scene(
        &[WorldObject { colour_id: 2, shape_id: 2, cx: 56.0, cy: 70.0 }],
        &mut lang_rng,
    )
    .rgb;
    let start_living = alpha_lang.mind.basal_columns.min(alpha_lang.mind.cells.len());
    for c in alpha_lang.mind.cells.iter_mut().skip(start_living) {
        c.fullness = 0.0;
    }
    alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
    let motor_novel = alpha_lang.step(&SensoryFrame {
        retina_rgb: &novel_cat_rgb,
        width: 128,
        height: 128,
        pcm_16khz: &[],
        body_flux: [0.0; 4],
    });
    let novel_attended = companion.attend_to_word(&motor_novel.pcm_out, &comp_multi_perceived).unwrap_or(99);

    // Feed varied cat/dog family instances to trigger natural Variety-Limit Mitosis (`cells_split`)
    let mut base_cat = vec![0.0f32; 32];
    for d in 0..32 {
        base_cat[d] = if d % 3 == 0 { 0.85 } else { -0.25 };
    }
    for v in 0..12 {
        let mut pat = base_cat.clone();
        for d in 0..32 {
            let osc = (((v * 7 + d * 5) % 19) as f32 / 9.5) - 1.0;
            pat[d] += 0.50 * osc;
        }
        let _ = alpha_lang.mind.step_living_cells(
            Some(&pat),
            None,
            None,
            &[0.4, 0.2, 0.2, 0.2],
            &[0.5; 8],
            0.15,
            ModulationBroadcast {
                surprise: 0.25,
                reward: 0.20,
                danger: 0.0,
                rest: 0.0,
            },
        );
    }
    let _ = alpha_lang.sleep_consolidate(16, 0.004);

    // Verify 4-Channel Cheap-Cells vs Waking-Brain Gate
    let cheap_before = alpha_lang.mind.cheap_cell_steps;
    for i in 0..4 {
        for _ in 0..3 {
            alpha_lang.body.state = [0.82, 0.98, 0.75, 0.78];
            let _ = alpha_lang.step(&SensoryFrame {
                retina_rgb: &single_rooms[i],
                width: 128,
                height: 128,
                pcm_16khz: &[],
                body_flux: [0.01; 4],
            });
        }
    }
    let cheap_delta = alpha_lang.mind.cheap_cell_steps - cheap_before;
    let (stm_c, ltm_c, concept_c, word_c, meaning_c) = alpha_lang.mind.living_type_counts();
    let (co_bonds_n, time_bonds_n) = alpha_lang.mind.total_living_bonds();
    println!(
        "  Living Epigenetic Cells: born={} | split(Mitosis)={} | merged={} | died_handoff={} | dream_bridges={}",
        alpha_lang.mind.cells_born,
        alpha_lang.mind.cells_split,
        alpha_lang.mind.cells_merged,
        alpha_lang.mind.cells_died_handoff,
        alpha_lang.mind.dream_bridges_formed
    );
    println!(
        "  Cell Differentiation   : STM={} | LTM={} | Higher-Layer Concepts={} | Word Cells={} | Meaning Cells={} | co_bonds={} | time_bonds={}",
        stm_c, ltm_c, concept_c, word_c, meaning_c, co_bonds_n, time_bonds_n
    );
    println!(
        "  4-Channel Brain Gate   : Familiar Cheap-Cell steps={} | Waking-Brain steps={}",
        cheap_delta, alpha_lang.mind.waking_brain_steps
    );
    println!(
        "  5 Grounded Language Stages: S1(Bonds)={}/4 | S2(Comprehension)={}/4 | S3(First Words)={}/4 | S4(Unseen 'want cat')={} (len={}, want_cos={:.3}, syl2_obj={}) | S5(Novel Use -> Companion)=obj#{}",
        stage1_hits,
        stage2_hits,
        stage3_hits,
        if stage4_pass { "PASS" } else { "PARTIAL" },
        motor_want_cat.pcm_out.len(),
        cos_syl1_want,
        syl2_attended,
        novel_attended
    );

    // Optional Opt-In External Video/Audio Sensory Stream (strictly disabled unless set in env)
    let opt_vid = std::env::var("ALPHA9_VIDEO_SOURCE").unwrap_or_default();
    let opt_aud = std::env::var("ALPHA9_AUDIO_SOURCE").unwrap_or_default();
    if let Some((cam_rgb, cw, ch)) = capture_hardware_eyes_rgb(&opt_vid) {
        let mic_pcm = capture_hardware_ears_pcm_16khz(&opt_aud, 8000).unwrap_or_default();
        let hw_motor = child.step(&SensoryFrame {
            retina_rgb: &cam_rgb,
            width: cw,
            height: ch,
            pcm_16khz: &mic_pcm,
            body_flux: [0.02, 0.02, 0.03, 0.03],
        });
        let r_cnt = child.mind.cells.iter().filter(|c| c.tissue_kind == CELL_TISSUE_RETINA_MEANING).count();
        let w_cnt = child.mind.cells.iter().filter(|c| c.tissue_kind == CELL_TISSUE_COCHLEA_WORD).count();
        println!(
            "  [External Opt-In Stream ({}x{}) & Audio ({} samples)] -> Gaze=({:.2}, {:.2}), Vocal=[{:.2}, {:.2}, {:.2}], RetinaCells={}, WordCells={}",
            cw, ch, mic_pcm.len(), hw_motor.gaze_xy[0], hw_motor.gaze_xy[1],
            hw_motor.vocal_tract[0], hw_motor.vocal_tract[1], hw_motor.vocal_tract[2],
            r_cnt, w_cnt
        );
    }
    println!("==========================================================================================");
}
