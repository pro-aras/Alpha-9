//! Alpha 9 Sensory Input Feeder & Autonomous Mini-PC Knowledge Forager (`alpha_forager`).

use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use alpha_iface::SensoryFrame;
use alpha_organism::curiosity::StreamCuriosityTracker;
use alpha_organism::genome::Genome;
use alpha_organism::util::{l2_norm, Rng64};
use alpha_organism::Organism;
use alpha_world::{render_scene, synthesize_word_utterance, WorldObject, NUM_COLOURS, NUM_SHAPES, NUM_WORDS};

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
        println!(
            "[FEED] image={}x{}, pcm_samples={}, gaze=({:.2},{:.2}), fixation={:?}, vocal_pcm_out={} samples, distress={:.4}, valence={:+.4}, arousal={:.4}, ram_mb={:.2}",
            w,
            h,
            pcm.len(),
            motor.gaze_xy[0],
            motor.gaze_xy[1],
            motor.fixation_blob,
            motor.pcm_out.len(),
            org.body.distress,
            org.body.valence,
            org.body.arousal,
            (org.memory_bytes() as f32) / (1024.0 * 1024.0)
        );
        return;
    }

    let mut ppm_files = Vec::new();
    collect_files_bounded(Path::new("/data/alpha3-library/K/scenes"), "ppm", 400, &mut ppm_files);
    let mut wav_files = Vec::new();
    collect_files_bounded(Path::new("/data/alpha3-library/audio/librispeech/dev-clean-wav"), "wav", 400, &mut wav_files);

    println!("==========================================================================");
    println!("ALPHA 9 AUTONOMOUS MINI-PC KNOWLEDGE FORAGER");
    println!(
        "Discovered local Mini-PC streams: {} real PPM scenes, {} real LibriSpeech WAVs",
        ppm_files.len(),
        wav_files.len()
    );
    println!("==========================================================================");

    let mut trackers = [
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
        StreamCuriosityTracker::new(16),
    ];
    let mut stream_counts = [0usize; 4];
    let mut rng = Rng64::new(2026);
    let mut total_pruned = 0usize;

    for step in 0..steps {
        let mut best_s = step % 4;
        if step >= 16 {
            let mut best_score = f32::NEG_INFINITY;
            for s in 0..4 {
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
                        org.develop_vision(&rgb, w, h);
                        let (objs, _) = org.perceive_objects(&rgb, w, h, true);
                        if let Some(first) = objs.first() {
                            first.s.iter().take(16).copied().collect()
                        } else {
                            vec![0.1f32; 16]
                        }
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

    println!(
        "[FORAGE DONE] steps={}, lexicon_prototypes={}, total_pruned_synapses={}, heap={:.2} MB",
        steps,
        org.binding.lexicon.len(),
        total_pruned,
        (org.memory_bytes() as f32) / (1024.0 * 1024.0)
    );
    println!(
        "[CURIOSITY ALLOCATION] Stream0(LibriSpeech)={} | Stream1(PPM Scenes)={} | Stream2(Grounded Naming)={} | Stream3(Noisy-TV)={}",
        stream_counts[0], stream_counts[1], stream_counts[2], stream_counts[3]
    );
    println!(
        "[INSULA & BODY VITALS] energy_atp={:.3}, thermal={:.3}, distress={:.4}, valence={:+.4}, arousal={:.4}",
        org.body.state[0], org.body.state[1], org.body.distress, org.body.valence, org.body.arousal
    );
}
