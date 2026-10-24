use std::f32::consts::PI;
use alpha_organism::util::Rng64;

pub const NUM_COLOURS: usize = 6;
pub const NUM_SHAPES: usize = 6;
pub const NUM_WORDS: usize = NUM_COLOURS + NUM_SHAPES;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldObject {
    pub colour_id: usize,
    pub shape_id: usize,
    pub cy: f32,
    pub cx: f32,
}

#[derive(Clone, Debug)]
pub struct WorldScene {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub objects: Vec<WorldObject>,
}

pub fn colour_rgb(colour_id: usize) -> [f32; 3] {
    match colour_id % NUM_COLOURS {
        0 => [235.0, 45.0, 45.0],
        1 => [45.0, 220.0, 55.0],
        2 => [45.0, 85.0, 240.0],
        3 => [240.0, 225.0, 45.0],
        4 => [215.0, 50.0, 220.0],
        _ => [45.0, 225.0, 225.0],
    }
}

pub fn shape_inside(shape_id: usize, dy: f32, dx: f32, radius: f32) -> bool {
    let ny = dy / radius;
    let nx = dx / radius;
    match shape_id % NUM_SHAPES {
        0 => ny * ny + nx * nx <= 1.0,
        1 => ny.abs().max(nx.abs()) <= 0.85,
        2 => ny.abs() + nx.abs() <= 1.05,
        3 => (ny.abs() <= 0.26 && nx.abs() <= 0.95) || (nx.abs() <= 0.26 && ny.abs() <= 0.95),
        4 => ny >= -0.85 && ny <= 0.85 && nx.abs() <= (0.85 - ny) * 0.55,
        _ => (ny * ny + nx * nx <= 1.0) && (ny * ny + nx * nx >= 0.36),
    }
}

pub fn render_scene(objects: &[WorldObject], rng: &mut Rng64) -> WorldScene {
    let (w, h) = (128usize, 128usize);
    let light = 0.80 + 0.35 * rng.next_f32();
    let mut rgb = vec![0u8; w * h * 3];

    for p in 0..(w * h) {
        let base = 110.0 * light;
        for c in 0..3 {
            let v = (base + 2.5 * rng.standard_normal()).clamp(10.0, 245.0) as u8;
            rgb[p * 3 + c] = v;
        }
    }

    for obj in objects {
        let col = colour_rgb(obj.colour_id);
        let r_px = 14.0f32;
        let y0 = (obj.cy - r_px - 2.0).max(2.0) as usize;
        let y1 = (obj.cy + r_px + 2.0).min((h - 2) as f32) as usize;
        let x0 = (obj.cx - r_px - 2.0).max(2.0) as usize;
        let x1 = (obj.cx + r_px + 2.0).min((w - 2) as f32) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                if shape_inside(obj.shape_id, (y as f32) - obj.cy, (x as f32) - obj.cx, r_px) {
                    let p = y * w + x;
                    for c in 0..3 {
                        let v = (col[c] * light + 2.5 * rng.standard_normal()).clamp(0.0, 255.0) as u8;
                        rgb[p * 3 + c] = v;
                    }
                }
            }
        }
    }

    WorldScene {
        width: w,
        height: h,
        rgb,
        objects: objects.to_vec(),
    }
}

/// Distinct 3-segment formant trajectories `(F1, F2, F3)` for each of the 12 words,
/// with realistic speaker fundamental pitch variation (`±25%`) and vocal-tract length variation (`±3.5%`).
pub fn synthesize_word_utterance(word_id: usize, speaker_id: usize, voice_model_shift: f32, rng: &mut Rng64) -> Vec<f32> {
    const TABLE: [([f32; 3], [f32; 3], [f32; 3]); 12] = [
        ([260.0, 720.0, 320.0], [2200.0, 1150.0, 2050.0], [2900.0, 2400.0, 2750.0]),
        ([750.0, 300.0, 680.0], [1100.0, 2350.0, 1250.0], [2450.0, 3100.0, 2500.0]),
        ([310.0, 320.0, 760.0], [950.0, 2400.0, 1300.0],  [2300.0, 3050.0, 2600.0]),
        ([720.0, 740.0, 280.0], [1350.0, 1000.0, 2300.0], [2600.0, 2250.0, 3000.0]),
        ([450.0, 270.0, 520.0], [1850.0, 920.0, 1750.0],  [2750.0, 2300.0, 2850.0]),
        ([280.0, 580.0, 290.0], [2450.0, 1500.0, 950.0],  [3200.0, 2650.0, 2250.0]),
        ([620.0, 420.0, 740.0], [920.0, 1950.0, 2350.0],  [2250.0, 2800.0, 3150.0]),
        ([340.0, 760.0, 440.0], [1600.0, 2300.0, 1020.0], [2550.0, 3050.0, 2350.0]),
        ([780.0, 290.0, 310.0], [2350.0, 1650.0, 960.0],  [3100.0, 2600.0, 2300.0]),
        ([500.0, 680.0, 270.0], [1050.0, 1700.0, 2450.0], [2350.0, 2700.0, 3200.0]),
        ([290.0, 480.0, 650.0], [2050.0, 2450.0, 1100.0], [2850.0, 3200.0, 2450.0]),
        ([660.0, 540.0, 360.0], [1450.0, 950.0, 1900.0],  [2650.0, 2250.0, 2800.0]),
    ];

    let sr = 16000usize;
    let dur = 5400usize;
    let pad = 1000usize;
    let total = dur + 2 * pad;
    let mut pcm = vec![0.0f32; total];

    let (f1s, f2s, f3s) = TABLE[word_id % NUM_WORDS];
    let vtl_scale = 1.0 + 0.03 * (((speaker_id as f32) * 1.618).sin()) + 0.35 * voice_model_shift;
    let f0 = 115.0 + 75.0 * (((speaker_id as f32) * 0.73).cos().abs()) + 25.0 * voice_model_shift;
    let amp = 0.50 + 0.35 * rng.next_f32();
    let dt = 1.0 / (sr as f32);

    let mut ph0 = 0.0f32;
    let mut ph1 = 0.0f32;
    let mut ph2 = 0.0f32;
    let mut ph3 = 0.0f32;

    for i in 0..dur {
        let t = (i as f32) / (dur as f32);
        let seg = if t < 0.333 { 0 } else if t < 0.666 { 1 } else { 2 };
        let f1 = f1s[seg] * vtl_scale;
        let f2 = f2s[seg] * vtl_scale;
        let f3 = f3s[seg] * vtl_scale;

        let env = (PI * t).sin().powi(2);
        ph0 = (ph0 + 2.0 * PI * f0 * dt) % (2.0 * PI);
        ph1 = (ph1 + 2.0 * PI * f1 * dt) % (2.0 * PI);
        ph2 = (ph2 + 2.0 * PI * f2 * dt) % (2.0 * PI);
        ph3 = (ph3 + 2.0 * PI * f3 * dt) % (2.0 * PI);

        pcm[pad + i] = amp * env * (0.20 * ph0.sin() + 0.40 * ph1.sin() + 0.25 * ph2.sin() + 0.15 * ph3.sin());
    }

    for v in &mut pcm {
        *v += 0.004 * rng.standard_normal();
    }
    pcm
}

/// Synthesize a connected 2-word 16 kHz spoken phrase (`word_a` followed by `word_b`)
/// with a natural 50 ms inter-syllable trough so the organism's cochlear theta envelope
/// segments both words naturally.
pub fn synthesize_two_word_phrase(
    word_a: usize,
    word_b: usize,
    speaker_id: usize,
    voice_model_shift: f32,
    rng: &mut Rng64,
) -> Vec<f32> {
    let pcm_a = synthesize_word_utterance(word_a, speaker_id, voice_model_shift, rng);
    let pcm_b = synthesize_word_utterance(word_b, speaker_id, voice_model_shift, rng);
    let mut out = Vec::with_capacity(pcm_a.len() + 800 + pcm_b.len());
    out.extend_from_slice(&pcm_a);
    out.resize(pcm_a.len() + 800, 0.0);
    out.extend_from_slice(&pcm_b);
    out
}

/// Literal Hardware Eyes (`/dev/video0`, e.g. Razer Kiyo):
/// Captures a live 640x480 YUYV frame directly from `/dev/video0` in memory and transduces it
/// into a 128x128 RGB retinal image (`width = 128, height = 128`).
pub fn capture_hardware_eyes_rgb(video_dev: &str) -> Option<(Vec<u8>, u16, u16)> {
    use std::process::Command;
    let out = Command::new("v4l2-ctl")
        .args([
            "-d",
            video_dev,
            "--set-fmt-video=width=640,height=480,pixelformat=YUYV",
            "--stream-mmap",
            "--stream-count=1",
            "--stream-to=-",
        ])
        .output()
        .ok()?;
    let yuyv = out.stdout;
    if yuyv.len() < 640 * 480 * 2 {
        return None;
    }
    let (dst_w, dst_h) = (128usize, 128usize);
    let mut rgb = vec![0u8; dst_w * dst_h * 3];
    for dy in 0..dst_h {
        let sy = (dy * 480) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * 640) / dst_w;
            let pair_idx = (sy * 640 + (sx & !1)) * 2;
            let y_val = if (sx & 1) == 0 {
                yuyv[pair_idx] as f32
            } else {
                yuyv[pair_idx + 2] as f32
            };
            let u_val = (yuyv[pair_idx + 1] as f32) - 128.0;
            let v_val = (yuyv[pair_idx + 3] as f32) - 128.0;
            let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
            let p = (dy * dst_w + dx) * 3;
            rgb[p] = r;
            rgb[p + 1] = g;
            rgb[p + 2] = b;
        }
    }
    Some((rgb, dst_w as u16, dst_h as u16))
}

/// Literal Hardware Ears (`plughw:2,0`, Razer Kiyo USB Microphone):
/// Captures live 16 kHz mono PCM audio from the physical USB microphone in memory.
pub fn capture_hardware_ears_pcm_16khz(alsa_dev: &str, max_samples: usize) -> Option<Vec<f32>> {
    use std::process::Command;
    let out = Command::new("arecord")
        .args([
            "-q",
            "-D",
            alsa_dev,
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "raw",
            "-d",
            "1",
        ])
        .output()
        .ok()?;
    let raw = out.stdout;
    if raw.len() < 2 {
        return None;
    }
    let n = (raw.len() / 2).min(max_samples.max(1600));
    let mut pcm = Vec::with_capacity(n);
    for i in 0..n {
        let s = i16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        pcm.push((s as f32) / 32768.0);
    }
    Some(pcm)
}
