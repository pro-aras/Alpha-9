//! Alpha 9 Sensory Input Feeder & Autonomous Knowledge Forager (`alpha_forager`).
//!
//! Features:
//! 1. Direct visual (.ppm) and auditory (.wav) sensory feeding (`--mode feed`)
//! 2. Autonomous Curiosity-Driven Knowledge Foraging (`--mode forage`) across:
//!    - Stream 0: Speech WAVs (`ALPHA9_SPEECH_DIR`)
//!    - Stream 1: Visual Scenes sampled via Biological Log-Polar Retina (`ALPHA9_SCENES_DIR`)
//!    - Stream 2: Cross-Situational Grounded Naming Stream
//!    - Stream 3: Local Markdown Document Reader (`ALPHA9_DOCS_DIR` rendered onto a visual canvas & scanned via Log-Polar Saccades)
//!    - Stream 4: Unlearnable White-Noise ("Noisy TV" trap control)
//! 3. Persistent Binary Brain Checkpointing (`--checkpoint state/organism.bin`)
//!    so synaptic weights, lexicon prototypes, and homeostatic state persist across runs.

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use alpha_iface::SensoryFrame;
use alpha_organism::curiosity::StreamCuriosityTracker;
use alpha_organism::genome::Genome;
use alpha_organism::sensors::Retina;
use alpha_organism::util::{l2_norm, Rng64};
use alpha_organism::Organism;
use alpha_world::{render_scene, synthesize_word_utterance, WorldObject, NUM_COLOURS, NUM_SHAPES, NUM_WORDS};

const MAGIC: &[u8; 8] = b"ALPHA9V1";

fn write_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn write_f32_slice(w: &mut Vec<u8>, s: &[f32]) {
    write_u64(w, s.len() as u64);
    for &x in s {
        w.extend_from_slice(&x.to_le_bytes());
    }
}
fn read_u64(r: &[u8], pos: &mut usize) -> Option<u64> {
    if *pos + 8 > r.len() {
        return None;
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&r[*pos..*pos + 8]);
    *pos += 8;
    Some(u64::from_le_bytes(b))
}
fn read_f32_vec(r: &[u8], pos: &mut usize) -> Option<Vec<f32>> {
    let len = read_u64(r, pos)? as usize;
    if *pos + len * 4 > r.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut b = [0u8; 4];
        b.copy_from_slice(&r[*pos..*pos + 4]);
        *pos += 4;
        out.push(f32::from_le_bytes(b));
    }
    Some(out)
}

fn save_checkpoint(org: &Organism, path: &Path) -> std::io::Result<usize> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = Vec::with_capacity(8 * 1024 * 1024);
    buf.extend_from_slice(MAGIC);
    write_u64(&mut buf, org.step_count);
    write_f32_slice(&mut buf, &org.body.state);
    write_f32_slice(&mut buf, &org.binding.bs);
    write_f32_slice(&mut buf, &org.binding.bc);
    write_u64(&mut buf, org.binding.lexicon.len() as u64);
    for (p, &cnt) in org.binding.lexicon.iter().zip(org.binding.lex_count.iter()) {
        write_u64(&mut buf, cnt as u64);
        write_f32_slice(&mut buf, p);
    }
    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    Ok(buf.len())
}

fn load_checkpoint(org: &mut Organism, path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() || buf.len() < 16 || &buf[0..8] != MAGIC {
        return false;
    }
    let mut pos = 8usize;
    let Some(steps) = read_u64(&buf, &mut pos) else { return false; };
    let Some(body_st) = read_f32_vec(&buf, &mut pos) else { return false; };
    let Some(bs) = read_f32_vec(&buf, &mut pos) else { return false; };
    let Some(bc) = read_f32_vec(&buf, &mut pos) else { return false; };
    let Some(n_lex) = read_u64(&buf, &mut pos) else { return false; };

    if body_st.len() == 4 && bs.len() == org.binding.bs.len() && bc.len() == org.binding.bc.len() {
        org.step_count = steps;
        org.body.state.copy_from_slice(&body_st);
        org.binding.bs = bs;
        org.binding.bc = bc;
        org.binding.lexicon.clear();
        org.binding.lex_count.clear();
        for _ in 0..(n_lex as usize) {
            if let (Some(cnt), Some(p)) = (read_u64(&buf, &mut pos), read_f32_vec(&buf, &mut pos)) {
                org.binding.lex_count.push(cnt as u32);
                org.binding.lexicon.push(p);
            }
        }
        return true;
    }
    false
}

/// Render a slice of ASCII/UTF-8 bytes from a local document onto a 128x128 RGB visual canvas
/// as structured 5x7 visual glyph patterns so the organism reads text exclusively through its
/// biological Log-Polar Retina (zero raw strings ever cross `SensoryFrame`).
fn render_document_page_to_rgb(doc_bytes: &[u8], offset: usize) -> (Vec<u8>, usize, usize, usize) {
    let (w, h) = (128usize, 128usize);
    let mut rgb = vec![24u8; w * h * 3]; // Dark-slate paper background
    let slice_len = 64usize.min(doc_bytes.len().saturating_sub(offset));
    let mut dominant_byte = 0usize;

    for idx in 0..slice_len {
        let ch = doc_bytes[offset + idx];
        if idx == 0 {
            dominant_byte = (ch as usize) % NUM_WORDS;
        }
        let row = idx / 8;
        let col = idx % 8;
        let y0 = 12 + row * 13;
        let x0 = 12 + col * 13;
        // Deterministic 5x7 visual glyph bitmap from byte bits
        for gy in 0..7 {
            for gx in 0..5 {
                let bit = ((ch as usize).wrapping_mul(gy + 3).wrapping_add(gx * 7) >> ((gy + gx) % 6)) & 1;
                if bit == 1 && y0 + gy < h && x0 + gx < w {
                    let p = ((y0 + gy) * w + (x0 + gx)) * 3;
                    rgb[p] = 235;
                    rgb[p + 1] = 220;
                    rgb[p + 2] = 180;
                }
            }
        }
    }
    (rgb, w, h, dominant_byte)
}

fn read_ppm_p6(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
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
            let s = std::str::from_utf8(&buf[start..idx]).ok()?;
            tokens.push(s.parse::<usize>().ok()?);
        }
    }
    if tokens.len() < 3 {
        return None;
    }
    let (w, h, _maxval) = (tokens[0], tokens[1], tokens[2]);
    if idx < buf.len() && matches!(buf[idx], b' ' | b'\n' | b'\r' | b'\t') {
        idx += 1;
    }
    let need = w * h * 3;
    if idx + need > buf.len() {
        return None;
    }
    Some((buf[idx..idx + need].to_vec(), w, h))
}

fn read_wav_pcm16(path: &Path, max_samples: usize) -> Option<Vec<f32>> {
    let mut f = File::open(path).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    if buf.len() < 44 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        pos += 8;
        if chunk_id == b"data" {
            let end = (pos + chunk_len).min(buf.len());
            let bytes = &buf[pos..end];
            let n = (bytes.len() / 2).min(max_samples);
            let mut pcm = Vec::with_capacity(n);
            for i in 0..n {
                let sample = i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                pcm.push((sample as f32) / 32768.0);
            }
            return Some(pcm);
        }
        pos += chunk_len;
    }
    None
}

fn read_minipc_hardware_vitals() -> (f32, f32) {
    let mut ram_free_ratio = 0.80f32;
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        let mut total_kb = 0.0f32;
        let mut avail_kb = 0.0f32;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            } else if line.starts_with("MemAvailable:") {
                avail_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
        }
        if total_kb > 0.0 {
            ram_free_ratio = (avail_kb / total_kb).clamp(0.0, 1.0);
        }
    }
    let mut cpu_thermal_comfort = 0.90f32;
    if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
        if let Some(load1) = loadavg.split_whitespace().next().and_then(|s| s.parse::<f32>().ok()) {
            cpu_thermal_comfort = (1.0 - (load1 / 16.0)).clamp(0.1, 1.0);
        }
    }
    (ram_free_ratio, cpu_thermal_comfort)
}

fn collect_files_bounded(dir: &Path, ext: &str, max_files: usize, out: &mut Vec<PathBuf>) {
    if out.len() >= max_files {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= max_files {
            break;
        }
        let p = entry.path();
        if p.is_dir() {
            collect_files_bounded(&p, ext, max_files, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut mode = "forage".to_string();
    let mut ppm_path: Option<PathBuf> = None;
    let mut wav_path: Option<PathBuf> = None;
    let mut ckpt_path = PathBuf::from(
        env::var("ALPHA9_CHECKPOINT").unwrap_or_else(|_| "state/organism.bin".to_string()),
    );
    let mut steps = 120usize;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" if i + 1 < args.len() => {
                mode = args[i + 1].clone();
                i += 2;
            }
            "--ppm" if i + 1 < args.len() => {
                ppm_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--wav" if i + 1 < args.len() => {
                wav_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--checkpoint" if i + 1 < args.len() => {
                ckpt_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                steps = args[i + 1].parse().unwrap_or(120);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let mut org = Organism::new(Genome::default());
    let loaded = load_checkpoint(&mut org, &ckpt_path);

    if mode == "feed" {
        let (rgb, w, h) = ppm_path
            .as_ref()
            .and_then(|p| read_ppm_p6(p))
            .unwrap_or_else(|| (vec![0u8; 128 * 128 * 3], 128, 128));
        let pcm = wav_path
            .as_ref()
            .and_then(|p| read_wav_pcm16(p, 16000))
            .unwrap_or_else(|| vec![0.0f32; 4000]);
        let (ram_ok, cpu_ok) = read_minipc_hardware_vitals();
        let frame = SensoryFrame {
            retina_rgb: &rgb,
            width: w as u16,
            height: h as u16,
            pcm_16khz: &pcm,
            body_flux: [0.02 * ram_ok, 0.02 * cpu_ok, 0.03, 0.02],
        };
        let motor = org.step(&frame);
        let (lp_sheet, cone_opp) = Retina::sample_log_polar_foveation(
            &rgb,
            w,
            h,
            motor.gaze_xy[1] * (h as f32),
            motor.gaze_xy[0] * (w as f32),
            12,
            24,
            48.0,
        );
        println!(
            "[FEED] ckpt_loaded={}, image={}x{}, logpolar_rings=12x24(energy={:.2}, cone_opp=[{:.2},{:.2},{:.2}]), gaze=({:.2},{:.2}), vocal_pcm_out={} samples, distress={:.4}, valence={:+.4}, arousal={:.4}, ram_mb={:.2}",
            loaded,
            w,
            h,
            l2_norm(&lp_sheet),
            cone_opp[0],
            cone_opp[1],
            cone_opp[2],
            motor.gaze_xy[0],
            motor.gaze_xy[1],
            motor.pcm_out.len(),
            org.body.distress,
            org.body.valence,
            org.body.arousal,
            (org.memory_bytes() as f32) / (1024.0 * 1024.0)
        );
        return;
    }

    let mut ppm_files = Vec::new();
    if let Ok(sc_dir) = env::var("ALPHA9_SCENES_DIR") {
        collect_files_bounded(Path::new(&sc_dir), "ppm", 400, &mut ppm_files);
    }
    let mut wav_files = Vec::new();
    if let Ok(sp_dir) = env::var("ALPHA9_SPEECH_DIR") {
        collect_files_bounded(Path::new(&sp_dir), "wav", 400, &mut wav_files);
    }
    let mut doc_files = Vec::new();
    if let Ok(dc_dir) = env::var("ALPHA9_DOCS_DIR") {
        collect_files_bounded(Path::new(&dc_dir), "md", 100, &mut doc_files);
    }
    collect_files_bounded(Path::new("."), "md", 20, &mut doc_files);

    println!("==========================================================================");
    println!("ALPHA 9 AUTONOMOUS MINI-PC KNOWLEDGE FORAGER (LOG-POLAR RETINA + PERSISTENT BRAIN)");
    println!(
        "Checkpoint loaded: {} (prior steps={}, prior lexicon={})",
        loaded,
        org.step_count,
        org.binding.lexicon.len()
    );
    println!(
        "Mini-PC Streams: {} PPM scenes, {} LibriSpeech WAVs, {} local Markdown/Research docs",
        ppm_files.len(),
        wav_files.len(),
        doc_files.len()
    );
    println!("==========================================================================");

    // 5 Streams:
    // 0: LibriSpeech WAVs | 1: Log-Polar PPM Scenes | 2: Cross-Situational Naming | 3: Visual-Auditory Document Reader | 4: Noisy-TV
    let mut trackers = [
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
    ];
    let mut stream_counts = [0usize; 5];
    let mut rng = Rng64::new(2026 + org.step_count);
    let mut total_pruned = 0usize;

    for step in 0..steps {
        let mut best_s = step % 5;
        if step >= 20 {
            let mut best_score = f32::NEG_INFINITY;
            for s in 0..5 {
                let noise = 0.02 * rng.standard_normal();
                let score = trackers[s].learning_progress + noise;
                if score > best_score {
                    best_score = score;
                    best_s = s;
                }
            }
        }
        stream_counts[best_s] += 1;

        let (ram_vital, cpu_vital) = read_minipc_hardware_vitals();

        let summary_16: Vec<f32> = match best_s {
            0 => {
                let pcm = if !wav_files.is_empty() {
                    let p = &wav_files[step % wav_files.len()];
                    read_wav_pcm16(p, 12000).unwrap_or_else(|| synthesize_word_utterance(step % NUM_WORDS, step % 12, 0.0, &mut rng))
                } else {
                    synthesize_word_utterance(step % NUM_WORDS, step % 12, 0.0, &mut rng)
                };
                org.develop_hearing(&pcm);
                let wf = org.word_form(&pcm, false);
                wf.iter().take(16).copied().collect()
            }
            1 => {
                if !ppm_files.is_empty() {
                    let p = &ppm_files[step % ppm_files.len()];
                    if let Some((rgb, w, h)) = read_ppm_p6(p) {
                        let (lp_sheet, _) = Retina::sample_log_polar_foveation(&rgb, w, h, 64.0, 64.0, 12, 24, 48.0);
                        org.develop_vision(&rgb, w, h);
                        lp_sheet.iter().take(16).copied().collect()
                    } else {
                        vec![0.1f32; 16]
                    }
                } else {
                    vec![0.1f32; 16]
                }
            }
            2 => {
                let c0 = step % NUM_COLOURS;
                let s0 = (step + 1) % NUM_SHAPES;
                let objs = vec![
                    WorldObject { colour_id: c0, shape_id: s0, cy: 40.0, cx: 40.0 },
                    WorldObject { colour_id: (c0 + 2) % NUM_COLOURS, shape_id: (s0 + 2) % NUM_SHAPES, cy: 86.0, cx: 86.0 },
                ];
                let sc = render_scene(&objs, &mut rng);
                let pcm = synthesize_word_utterance(c0, step % 14, 0.0, &mut rng);
                org.experience(&sc.rgb, sc.width, sc.height, &pcm, 2);
                let mut pat = vec![0.0f32; 16];
                for i in 0..16 {
                    pat[i] = if (i + (step % 4)) % 4 == 0 { 0.85 } else { -0.35 };
                }
                pat
            }
            3 => {
                // Visual-Auditory Document Reader via Log-Polar Saccades
                let doc_bytes = if !doc_files.is_empty() {
                    fs::read(&doc_files[step % doc_files.len()]).unwrap_or_else(|_| vec![b'A'; 256])
                } else {
                    vec![b'A'; 256]
                };
                let offset = (step * 64) % doc_bytes.len().max(1);
                let (rgb, w, h, syl_id) = render_document_page_to_rgb(&doc_bytes, offset);
                let saccade_x = 24.0 + ((step % 4) as f32) * 24.0;
                let (lp_sheet, _) = Retina::sample_log_polar_foveation(&rgb, w, h, 64.0, saccade_x, 12, 24, 42.0);
                let pcm = synthesize_word_utterance(syl_id % NUM_WORDS, step % 10, 0.0, &mut rng);
                org.experience(&rgb, w, h, &pcm, 2);
                lp_sheet.iter().take(16).copied().collect()
            }
            _ => {
                (0..16).map(|_| rng.standard_normal().clamp(-1.0, 1.0)).collect()
            }
        };

        let (_, prog) = trackers[best_s].observe_and_learn(&summary_16, 0.05);
        let _ = org.body.step_homeostasis_with_cognition(
            &[0.015 * ram_vital, 0.015 * cpu_vital, 0.01, 0.02 * prog.max(0.0)],
            0.03,
            l2_norm(&summary_16) * 0.1,
            0.25,
        );

        if (step + 1) % 40 == 0 {
            let (pruned, _) = org.sleep_consolidate(16, 0.015);
            total_pruned += pruned;
        }
    }

    let saved_bytes = save_checkpoint(&org, &ckpt_path).unwrap_or(0);
    println!(
        "[FORAGE DONE] total_steps={}, lexicon_prototypes={}, pruned_synapses={}, ckpt_saved={:.2} MB, heap={:.2} MB",
        org.step_count,
        org.binding.lexicon.len(),
        total_pruned,
        (saved_bytes as f32) / (1024.0 * 1024.0),
        (org.memory_bytes() as f32) / (1024.0 * 1024.0)
    );
    println!(
        "[CURIOSITY ALLOCATION] Stream0(LibriSpeech)={} | Stream1(LogPolar Scenes)={} | Stream2(Naming)={} | Stream3(DocReader)={} | Stream4(Noisy-TV)={}",
        stream_counts[0], stream_counts[1], stream_counts[2], stream_counts[3], stream_counts[4]
    );
}
