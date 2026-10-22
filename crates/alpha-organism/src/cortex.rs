use crate::util::{dot, kwta_in_place, l2_norm, normalize_l2_in_place, Rng64};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DendriticRegime {
    Silent,
    PredictedSingleSpike,
    SurpriseApicalCaPlateauBurst,
    OmissionFailedPredictionDip,
}

#[derive(Clone, Debug)]
pub struct CompetitiveArea {
    pub n_in: usize,
    pub n_units: usize,
    pub k: usize,
    pub eta: f32,
    pub w: Vec<f32>,
    pub usage: Vec<f32>,
}

impl CompetitiveArea {
    pub fn new(n_in: usize, n_units: usize, k: usize, eta: f32, rng: &mut Rng64) -> Self {
        let mut w = vec![0.0f32; n_units * n_in];
        for u in 0..n_units {
            let row = &mut w[u * n_in..(u + 1) * n_in];
            for v in row.iter_mut() {
                *v = rng.standard_normal();
            }
            normalize_l2_in_place(row);
        }
        Self {
            n_in,
            n_units,
            k,
            eta,
            w,
            usage: vec![1.0f32; n_units],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.w.capacity() + self.usage.capacity()) * 4
    }

    pub fn respond(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut xn = x.to_vec();
        normalize_l2_in_place(&mut xn);
        let mut r = vec![0.0f32; self.n_units];
        for u in 0..self.n_units {
            let row = &self.w[u * self.n_in..(u + 1) * self.n_in];
            r[u] = dot(&xn, row).max(0.0);
        }
        kwta_in_place(&mut r, self.k);
        (r, xn)
    }

    pub fn learn(&mut self, x: &[f32]) -> Vec<f32> {
        let (r, xn) = self.respond(x);
        let mut best_u = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for u in 0..self.n_units {
            let bias = 0.02 / (self.usage[u] + 1e-3);
            let val = r[u] + bias;
            if val > best_val {
                best_val = val;
                best_u = u;
            }
        }
        let eff_eta = self.eta / (1.0 + 0.012 * (self.usage[best_u] - 1.0).max(0.0));
        let row = &mut self.w[best_u * self.n_in..(best_u + 1) * self.n_in];
        for (wv, &xv) in row.iter_mut().zip(xn.iter()) {
            *wv += eff_eta * (xv - *wv);
        }
        normalize_l2_in_place(row);
        self.usage[best_u] = 0.999 * self.usage[best_u] + 1.0;
        for u in &mut self.usage {
            *u *= 0.9995;
        }
        r
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalArea {
    pub n_in: usize,
    pub n_units: usize,
    pub k: usize,
    pub beta: f32,
    pub w_basal: Vec<f32>,
    pub w_seq: Vec<f32>,
    pub mean_in: Vec<f32>,
    pub step_in: u64,
    pub usage: Vec<f32>,
    pub maturity: Vec<f32>,
    pub prev_active: Vec<usize>,
    pub last_surprise: f32,
    pub last_regime: DendriticRegime,
}

impl CanonicalArea {
    pub fn new(n_in: usize, n_units: usize, k: usize, p: f32, beta: f32, rng: &mut Rng64) -> Self {
        let mut w_basal = vec![0.0f32; n_units * n_in];
        for u in 0..n_units {
            let row = &mut w_basal[u * n_in..(u + 1) * n_in];
            for v in row.iter_mut() {
                if rng.next_f32() < p {
                    *v = rng.standard_normal();
                }
            }
            normalize_l2_in_place(row);
        }
        let target_rate = (k as f32) / (n_units.max(1) as f32);
        Self {
            n_in,
            n_units,
            k,
            beta,
            w_basal,
            w_seq: vec![0.0f32; n_units * n_units],
            mean_in: vec![0.0f32; n_in],
            step_in: 0,
            usage: vec![target_rate; n_units],
            maturity: vec![0.0f32; n_units],
            prev_active: Vec::new(),
            last_surprise: 0.0,
            last_regime: DendriticRegime::Silent,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.w_basal.capacity()
            + self.w_seq.capacity()
            + self.mean_in.capacity()
            + self.usage.capacity()
            + self.maturity.capacity()
            + self.prev_active.capacity())
            * 4
    }

    fn adapt_and_center(&self, x: &[f32]) -> Vec<f32> {
        let mut xn = Vec::with_capacity(self.n_in);
        for i in 0..self.n_in {
            let xi = if i < x.len() { x[i] } else { 0.0 };
            xn.push(xi - self.mean_in[i]);
        }
        if !xn.is_empty() {
            let mean: f32 = xn.iter().copied().sum::<f32>() / (xn.len() as f32);
            for v in &mut xn {
                *v -= mean;
            }
        }
        normalize_l2_in_place(&mut xn);
        xn
    }

    pub fn fire(&self, x: &[f32]) -> Vec<f32> {
        let xn = self.adapt_and_center(x);
        let mut drive = vec![0.0f32; self.n_units];
        for u in 0..self.n_units {
            let row = &self.w_basal[u * self.n_in..(u + 1) * self.n_in];
            drive[u] = dot(row, &xn);
        }
        let mut pairs: Vec<(usize, f32)> = drive.into_iter().enumerate().collect();
        let k = self.k.min(self.n_units);
        pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.total_cmp(&a.1));
        let mut a = vec![0.0f32; self.n_units];
        for &(idx, _) in &pairs[..k] {
            a[idx] = 1.0;
        }
        a
    }

    pub fn fire_and_learn(&mut self, x: &[f32], apical_pred: Option<&[f32]>) -> Vec<f32> {
        // Update per-afferent synaptic baseline (short-term depression of common-mode background)
        self.step_in += 1;
        let alpha_in = (1.0 / (self.step_in as f32)).clamp(0.015, 0.50);
        for i in 0..self.n_in.min(x.len()) {
            self.mean_in[i] += alpha_in * (x[i] - self.mean_in[i]);
        }

        let xn = self.adapt_and_center(x);
        let target_rate = (self.k as f32) / (self.n_units.max(1) as f32);

        let mut drive = vec![0.0f32; self.n_units];
        for u in 0..self.n_units {
            let row = &self.w_basal[u * self.n_in..(u + 1) * self.n_in];
            let homeo_bias = 0.08 * (target_rate - self.usage[u]);
            drive[u] = dot(row, &xn) + homeo_bias;
        }
        let mut pairs: Vec<(usize, f32)> = drive.into_iter().enumerate().collect();
        let k = self.k.min(self.n_units);
        pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.total_cmp(&a.1));
        let mut a = vec![0.0f32; self.n_units];
        let mut active_idx = Vec::with_capacity(k);
        for &(idx, _) in &pairs[..k] {
            a[idx] = 1.0;
            active_idx.push(idx);
        }

        for u in 0..self.n_units {
            let fired = if a[u] > 0.5 { 1.0 } else { 0.0 };
            self.usage[u] = 0.99 * self.usage[u] + 0.01 * fired;
        }

        let (gate, regime) = if let Some(pred) = apical_pred {
            let p_norm = l2_norm(pred);
            let a_norm = l2_norm(&a);
            if p_norm > 1e-6 && a_norm > 1e-6 {
                let match_cos = dot(&a, pred) / (a_norm * p_norm);
                if match_cos >= 0.35 {
                    (0.25, DendriticRegime::PredictedSingleSpike)
                } else {
                    (1.0 + (0.35 - match_cos).max(0.0), DendriticRegime::SurpriseApicalCaPlateauBurst)
                }
            } else if p_norm >= 0.55 && a_norm < 0.25 {
                (0.5, DendriticRegime::OmissionFailedPredictionDip)
            } else {
                (1.0, DendriticRegime::SurpriseApicalCaPlateauBurst)
            }
        } else {
            (1.0, DendriticRegime::SurpriseApicalCaPlateauBurst)
        };
        self.last_surprise = gate;
        self.last_regime = regime;

        for &u in &active_idx {
            // Perineuronal net (PNN) metaplastic stabilization: uncommitted columns learn fast,
            // while consolidated columns resist representational drift so downstream hippocampal
            // bindings remain stable over lifelong open-ended experience.
            let pnn_scale = 1.0 / (1.0 + 0.08 * self.maturity[u]);
            let eff_beta = self.beta * gate * pnn_scale;
            let row = &mut self.w_basal[u * self.n_in..(u + 1) * self.n_in];
            for (wv, &xv) in row.iter_mut().zip(xn.iter()) {
                *wv += eff_beta * (xv - *wv);
            }
            normalize_l2_in_place(row);
            self.maturity[u] = (self.maturity[u] + 1.0).min(200.0);
        }

        if !self.prev_active.is_empty() {
            let eta_seq = 0.08;
            for &pre in &self.prev_active {
                let row = &mut self.w_seq[pre * self.n_units..(pre + 1) * self.n_units];
                for wv in row.iter_mut() {
                    *wv *= 1.0 - eta_seq;
                }
                for &post in &active_idx {
                    row[post] += eta_seq;
                }
            }
        }
        self.prev_active = active_idx;
        a
    }

    pub fn sequence_transition_score(&self, pre_a: &[f32], post_a: &[f32]) -> f32 {
        let mut score = 0.0f32;
        let mut count = 0.0f32;
        for (i, &va) in pre_a.iter().enumerate() {
            if va > 0.5 {
                let row = &self.w_seq[i * self.n_units..(i + 1) * self.n_units];
                for (j, &vb) in post_a.iter().enumerate() {
                    if vb > 0.5 {
                        score += row[j];
                        count += 1.0;
                    }
                }
            }
        }
        if count > 0.0 { score / count } else { 0.0 }
    }
}
