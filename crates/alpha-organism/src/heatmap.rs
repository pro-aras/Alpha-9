//! Biophysical Continuous Heat-Map Sheet & Hopfield/Boltzmann Attractor Memory (Q1, Q2, Q4, Q5).

use crate::util::{dot, Rng64};

#[derive(Clone, Debug)]
pub struct BiophysicalHeatMap {
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
    pub v: Vec<f32>,
    pub atp: Vec<f32>,
    pub s: Vec<f32>,
    pub w_mem: Vec<f32>,
    pub w_bind: Vec<f32>,
    pub basal_cost: f32,
    pub spike_cost: f32,
    pub k_in: f32,
    pub k_decay: f32,
    pub k_diff: f32,
    pub threshold: f32,
}

impl BiophysicalHeatMap {
    pub fn new(n: usize) -> Self {
        let rows = (n as f32).sqrt().floor().max(1.0) as usize;
        let cols = (n + rows - 1) / rows;
        Self {
            n,
            rows,
            cols,
            v: vec![0.0f32; n],
            atp: vec![1.0f32; n],
            s: vec![0.0f32; n],
            w_mem: vec![0.0f32; n * n],
            w_bind: vec![0.0f32; n * n],
            basal_cost: 0.002,
            spike_cost: 0.045,
            k_in: 0.65,
            k_decay: 0.22,
            k_diff: 0.08,
            threshold: 0.35,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.v.capacity()
            + self.atp.capacity()
            + self.s.capacity()
            + self.w_mem.capacity()
            + self.w_bind.capacity())
            * 4
    }

    pub fn step_biophysics(&mut self, input: &[f32], noise_sigma: f32, atp_supply: f32, rng: &mut Rng64) -> usize {
        let prev_v = self.v.clone();
        let mut spikes = 0usize;

        for i in 0..self.n {
            self.atp[i] = (self.atp[i] - self.basal_cost + atp_supply).clamp(0.0, 1.0);
            let r = i / self.cols;
            let c = i % self.cols;
            let mut nb_sum = 0.0f32;
            let mut nb_cnt = 0.0f32;
            if r > 0 && i >= self.cols {
                nb_sum += prev_v[i - self.cols];
                nb_cnt += 1.0;
            }
            if r + 1 < self.rows && i + self.cols < self.n {
                nb_sum += prev_v[i + self.cols];
                nb_cnt += 1.0;
            }
            if c > 0 && i > 0 {
                nb_sum += prev_v[i - 1];
                nb_cnt += 1.0;
            }
            if c + 1 < self.cols && i + 1 < self.n {
                nb_sum += prev_v[i + 1];
                nb_cnt += 1.0;
            }
            let diff = if nb_cnt > 0.0 {
                (nb_sum / nb_cnt) - prev_v[i]
            } else {
                0.0
            };
            let inp = if i < input.len() { input[i] } else { 0.0 };
            let noise = noise_sigma * rng.standard_normal();

            self.v[i] += self.k_in * inp - self.k_decay * prev_v[i] + self.k_diff * diff + noise;

            if self.v[i] > self.threshold && self.atp[i] >= self.spike_cost {
                self.s[i] = 1.0;
                self.atp[i] -= self.spike_cost;
                self.v[i] *= 0.35;
                spikes += 1;
            } else {
                self.s[i] = -1.0;
            }
        }
        spikes
    }

    pub fn learn_pattern(&mut self, p: &[f32], lr: f32) {
        let scale = lr / (self.n as f32);
        for i in 0..self.n {
            let row = &mut self.w_mem[i * self.n..(i + 1) * self.n];
            let pi = p[i];
            for j in 0..self.n {
                if i != j {
                    row[j] += scale * pi * p[j];
                }
            }
        }
    }

    pub fn bind_patterns(&mut self, a: &[f32], b: &[f32], lr: f32) {
        let scale = lr / (self.n as f32);
        for i in 0..self.n {
            let row = &mut self.w_bind[i * self.n..(i + 1) * self.n];
            let bi = b[i];
            for j in 0..self.n {
                row[j] += scale * bi * a[j];
            }
        }
    }

    pub fn conflict_energy(&self, candidate: &[f32]) -> f32 {
        let mut e = 0.0f32;
        for i in 0..self.n {
            let row = &self.w_mem[i * self.n..(i + 1) * self.n];
            e -= candidate[i] * dot(row, candidate);
        }
        0.5 * e / (self.n as f32)
    }

    pub fn settle(&self, cue: &[f32], steps: usize, temp_start: f32, rng: &mut Rng64) -> Vec<f32> {
        let mut s = cue.to_vec();
        for step in 0..steps {
            let temp = temp_start * (1.0 - (step as f32) / (steps.max(1) as f32));
            let mut next_s = vec![0.0f32; self.n];
            for i in 0..self.n {
                let row = &self.w_mem[i * self.n..(i + 1) * self.n];
                let drive = dot(row, &s) + temp * rng.standard_normal();
                next_s[i] = if drive >= 0.0 { 1.0 } else { -1.0 };
            }
            s = next_s;
        }
        s
    }

    pub fn follow(&self, a: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n];
        for i in 0..self.n {
            let row = &self.w_bind[i * self.n..(i + 1) * self.n];
            let d = dot(row, a);
            out[i] = if d >= 0.0 { 1.0 } else { -1.0 };
        }
        out
    }
}
