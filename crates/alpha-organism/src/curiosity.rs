//! Self-Supervised Prediction Error, Learning-Progress Curiosity, and Generator/Judge Discovery (Q3, Q4, Q5).

use crate::heatmap::BiophysicalHeatMap;
use crate::util::{dot, l2_norm, Rng64};

#[derive(Clone, Debug)]
pub struct StreamCuriosityTracker {
    pub dim: usize,
    pub w_pred: Vec<f32>,
    pub bias: Vec<f32>,
    pub prev_state: Vec<f32>,
    pub error_before: f32,
    pub error_now: f32,
    pub learning_progress: f32,
    pub step_count: usize,
}

impl StreamCuriosityTracker {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            w_pred: vec![0.0f32; dim * dim],
            bias: vec![0.0f32; dim],
            prev_state: vec![0.0f32; dim],
            error_before: 0.0,
            error_now: 0.0,
            learning_progress: 0.0,
            step_count: 0,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.w_pred.capacity() + self.bias.capacity() + self.prev_state.capacity()) * 4
    }

    pub fn observe_and_learn(&mut self, obs: &[f32], base_lr: f32) -> (f32, f32) {
        self.step_count += 1;
        if self.step_count == 1 {
            self.prev_state.copy_from_slice(&obs[..self.dim]);
            self.bias.copy_from_slice(&obs[..self.dim]);
            return (0.0, 0.0);
        }
        let mut pred = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = &self.w_pred[i * self.dim..(i + 1) * self.dim];
            pred[i] = (self.bias[i] + dot(row, &self.prev_state)).clamp(-1.5, 1.5);
        }
        let mut err_sq = 0.0f32;
        for i in 0..self.dim {
            let d = obs[i] - pred[i];
            err_sq += d * d;
        }
        let err = (err_sq / (self.dim as f32)).sqrt();

        if self.step_count == 2 {
            self.error_before = err;
            self.error_now = err;
        } else {
            self.error_before = 0.96 * self.error_before + 0.04 * err;
            self.error_now = 0.70 * self.error_now + 0.30 * err;
        }
        self.learning_progress = self.error_before - self.error_now;

        let lr = (base_lr * (1.0 + 1.5 * err)).min(0.15);
        for i in 0..self.dim {
            let err_i = obs[i] - pred[i];
            self.bias[i] += 0.25 * lr * err_i;
            let row = &mut self.w_pred[i * self.dim..(i + 1) * self.dim];
            for j in 0..self.dim {
                row[j] = (row[j] + lr * err_i * self.prev_state[j]).clamp(-2.0, 2.0);
            }
        }
        self.prev_state.copy_from_slice(&obs[..self.dim]);
        (err, self.learning_progress)
    }
}

#[derive(Clone, Debug)]
pub struct DiscoveryArchive {
    pub candidates: Vec<Vec<f32>>,
    pub max_size: usize,
}

impl DiscoveryArchive {
    pub fn new(max_size: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(max_size),
            max_size,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.candidates.iter().map(|v| v.capacity() * 4).sum()
    }

    pub fn propose_and_settle(
        &mut self,
        heatmap: &BiophysicalHeatMap,
        seed_pattern: &[f32],
        noise_temp: f32,
        accept_energy_thresh: f32,
        rng: &mut Rng64,
    ) -> Option<(Vec<f32>, f32)> {
        let mut proposal = Vec::with_capacity(heatmap.n);
        for &v in seed_pattern.iter().take(heatmap.n) {
            let noisy = v + noise_temp * rng.standard_normal();
            proposal.push(if noisy >= 0.0 { 1.0 } else { -1.0 });
        }
        let settled = heatmap.settle(&proposal, 15, noise_temp * 0.5, rng);
        let energy = heatmap.conflict_energy(&settled);
        if energy < accept_energy_thresh {
            let is_novel = self.candidates.iter().all(|c| {
                let cos = dot(c, &settled) / (l2_norm(c) * l2_norm(&settled) + 1e-8);
                cos < 0.85
            });
            if is_novel && self.candidates.len() < self.max_size {
                self.candidates.push(settled.clone());
            }
            Some((settled, energy))
        } else {
            None
        }
    }
}
