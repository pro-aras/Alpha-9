use std::f32::consts::PI;
use crate::util::{dot, normalize_l2_in_place, Rng64};

#[derive(Clone, Debug)]
pub struct VocalTract {
    pub sr: usize,
    pub feat_dim: usize,
    pub w_inv: Vec<f32>,
}

impl VocalTract {
    pub fn new(sr: usize, feat_dim: usize, rng: &mut Rng64) -> Self {
        let mut w_inv = vec![0.0f32; 8 * feat_dim];
        for v in &mut w_inv {
            *v = 0.01 * rng.standard_normal();
        }
        Self { sr, feat_dim, w_inv }
    }

    pub fn memory_bytes(&self) -> usize {
        self.w_inv.capacity() * 4
    }

    pub fn synthesize_pcm(&self, m: &[f32; 8]) -> Vec<f32> {
        let n_samples = (self.sr * 45) / 100;
        let mut pcm = vec![0.0f32; n_samples];
        let f0 = 110.0 + 140.0 * m[0].clamp(0.0, 1.0);
        let f1_start = 250.0 + 650.0 * m[1].clamp(0.0, 1.0);
        let f1_end = 250.0 + 650.0 * m[2].clamp(0.0, 1.0);
        let f2_start = 800.0 + 1800.0 * m[3].clamp(0.0, 1.0);
        let f2_end = 800.0 + 1800.0 * m[4].clamp(0.0, 1.0);
        let f3 = 2200.0 + 1100.0 * m[5].clamp(0.0, 1.0);
        let harm_richness = 0.3 + 0.7 * m[6].clamp(0.0, 1.0);
        let amp = 0.25 + 0.55 * m[7].clamp(0.0, 1.0);

        let mut ph1 = 0.0f32;
        let mut ph2 = 0.0f32;
        let mut ph3 = 0.0f32;
        let mut ph0 = 0.0f32;
        let dt = 1.0 / (self.sr as f32);

        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (i as f32) / (n_samples as f32);
            let env = (PI * t).sin().powi(2);
            let f1 = f1_start + (f1_end - f1_start) * t;
            let f2 = f2_start + (f2_end - f2_start) * t;

            ph0 = (ph0 + 2.0 * PI * f0 * dt) % (2.0 * PI);
            ph1 = (ph1 + 2.0 * PI * f1 * dt) % (2.0 * PI);
            ph2 = (ph2 + 2.0 * PI * f2 * dt) % (2.0 * PI);
            ph3 = (ph3 + 2.0 * PI * f3 * dt) % (2.0 * PI);

            let glottal = ph0.sin() + 0.5 * harm_richness * (2.0 * ph0).sin();
            let formants = 0.55 * ph1.sin() + 0.35 * ph2.sin() + 0.15 * ph3.sin();
            *sample = amp * env * (0.35 * glottal + 0.65 * formants);
        }
        pcm
    }

    pub fn infer_motor(&self, word_feat: &[f32]) -> [f32; 8] {
        let mut m = [0.0f32; 8];
        for d in 0..8 {
            let row = &self.w_inv[d * self.feat_dim..(d + 1) * self.feat_dim];
            m[d] = (0.5 + dot(row, word_feat)).clamp(0.0, 1.0);
        }
        m
    }

    pub fn learn_babble_pair(&mut self, word_feat: &[f32], target_motor: &[f32; 8], eta: f32) {
        let mut fn_norm = word_feat.to_vec();
        normalize_l2_in_place(&mut fn_norm);
        let pred = self.infer_motor(&fn_norm);
        for d in 0..8 {
            let err = target_motor[d] - pred[d];
            let row = &mut self.w_inv[d * self.feat_dim..(d + 1) * self.feat_dim];
            for (wv, &fv) in row.iter_mut().zip(fn_norm.iter()) {
                *wv += eta * err * fv;
            }
        }
    }
}
