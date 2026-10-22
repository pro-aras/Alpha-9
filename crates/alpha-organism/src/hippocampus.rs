use crate::util::{dot, l2_norm};

#[derive(Clone, Debug)]
pub struct PerceivedObject {
    pub cy: f32,
    pub cx: f32,
    pub s: Vec<f32>,
    pub c: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CrossModalBinding {
    pub n_w: usize,
    pub n_s: usize,
    pub n_c: usize,
    pub bs: Vec<f32>,
    pub bc: Vec<f32>,
    pub m_w: Vec<f32>,
    pub m_s: Vec<f32>,
    pub m_c: Vec<f32>,
    pub lexicon: Vec<Vec<f32>>,
    pub lex_count: Vec<u32>,
    pub overlap_thresh: f32,
}

impl CrossModalBinding {
    pub fn new(n_w: usize, k_w: usize, n_s: usize, k_s: usize, n_c: usize, k_c: usize, overlap_thresh: f32) -> Self {
        Self {
            n_w,
            n_s,
            n_c,
            bs: vec![0.0f32; n_w * n_s],
            bc: vec![0.0f32; n_w * n_c],
            m_w: vec![(k_w as f32) / (n_w as f32); n_w],
            m_s: vec![(k_s as f32) / (n_s as f32); n_s],
            m_c: vec![(k_c as f32) / (n_c as f32); n_c],
            lexicon: Vec::new(),
            lex_count: Vec::new(),
            overlap_thresh,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let base = (self.bs.capacity() + self.bc.capacity() + self.m_w.capacity() + self.m_s.capacity() + self.m_c.capacity()) * 4;
        let lex: usize = self.lexicon.iter().map(|v| v.capacity() * 4).sum();
        base + lex + self.lex_count.capacity() * 4
    }

    pub fn update_lexicon(&mut self, a: &[f32], _k_w: f32) -> (usize, bool) {
        // Compute zero-mean Deviation from running cortical baseline `m_w` to eliminate common-mode cone
        let da: Vec<f32> = a.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
        let na = l2_norm(&da) + 1e-8;
        if !self.lexicon.is_empty() {
            let total_cnt: u32 = self.lex_count.iter().sum::<u32>().max(1);
            let avg_share = 1.0 / (self.lexicon.len().max(1) as f32);
            let mut best_j = 0usize;
            let mut best_ov = f32::NEG_INFINITY;
            for (j, p) in self.lexicon.iter().enumerate() {
                let dp: Vec<f32> = p.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
                let np = l2_norm(&dp) + 1e-8;
                let raw_cos = dot(&da, &dp) / (na * np);
                // Homeostatic conscience: prevent any single prototype from monopolizing >2x its fair share
                let share = (self.lex_count[j] as f32) / (total_cnt as f32);
                let penalty = (share - avg_share).max(0.0) * 0.35;
                let ov = raw_cos - penalty;
                if ov > best_ov {
                    best_ov = ov;
                    best_j = j;
                }
            }
            if best_ov > self.overlap_thresh && self.lexicon.len() >= 18 || best_ov > (self.overlap_thresh + 0.18) {
                self.lex_count[best_j] += 1;
                let eta_p = (1.0 / (self.lex_count[best_j] as f32)).max(0.04);
                for (pv, &av) in self.lexicon[best_j].iter_mut().zip(a.iter()) {
                    *pv += eta_p * (av - *pv);
                }
                return (best_j, false);
            }
        }
        // Open-ended developmental lexicon growth (bounded only by physical RAM safety limit 4096)
        if self.lexicon.len() < 4096 {
            let j = self.lexicon.len();
            self.lexicon.push(a.to_vec());
            self.lex_count.push(1);
            return (j, true);
        }
        // If at physical RAM safety limit, merge into closest prototype
        let mut best_j = 0usize;
        let mut best_ov = f32::NEG_INFINITY;
        for (j, p) in self.lexicon.iter().enumerate() {
            let dp: Vec<f32> = p.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
            let ov = dot(&da, &dp) / (na * (l2_norm(&dp) + 1e-8));
            if ov > best_ov {
                best_ov = ov;
                best_j = j;
            }
        }
        self.lex_count[best_j] += 1;
        for (pv, &av) in self.lexicon[best_j].iter_mut().zip(a.iter()) {
            *pv += 0.05 * (av - *pv);
        }
        (best_j, false)
    }

    pub fn expectations(&self, a: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Lexical attractor pattern completion: blend incoming assembly `a` with best matching
        // consolidated prototype in `self.lexicon` if similarity is above threshold
        let mut a_eff = a.to_vec();
        if !self.lexicon.is_empty() {
            let da: Vec<f32> = a.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
            let na = l2_norm(&da) + 1e-8;
            let mut best_j = 0usize;
            let mut best_cos = f32::NEG_INFINITY;
            for (j, p) in self.lexicon.iter().enumerate() {
                let dp: Vec<f32> = p.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
                let c = dot(&da, &dp) / (na * (l2_norm(&dp) + 1e-8));
                if c > best_cos {
                    best_cos = c;
                    best_j = j;
                }
            }
            if best_cos > 0.20 {
                let proto = &self.lexicon[best_j];
                for (av, &pv) in a_eff.iter_mut().zip(proto.iter()) {
                    *av = 0.65 * (*av) + 0.35 * pv;
                }
            }
        }

        let mut es = vec![0.0f32; self.n_s];
        let mut ec = vec![0.0f32; self.n_c];
        for i in 0..self.n_w {
            // Pre-synaptic glutamate release occurs ONLY on active pre-synaptic neurons (`a_eff[i] > 0.05`),
            // weighted by synaptic specificity (habituating non-selective hub neurons with high `m_w[i]`).
            if a_eff[i] <= 0.05 {
                continue;
            }
            let spec = (1.0 - 2.2 * self.m_w[i]).max(0.05);
            let pre_drive = a_eff[i] * spec;
            let row_s = &self.bs[i * self.n_s..(i + 1) * self.n_s];
            for (s_val, &w_val) in es.iter_mut().zip(row_s.iter()) {
                *s_val += pre_drive * w_val;
            }
            let row_c = &self.bc[i * self.n_c..(i + 1) * self.n_c];
            for (c_val, &w_val) in ec.iter_mut().zip(row_c.iter()) {
                *c_val += pre_drive * w_val;
            }
        }
        (es, ec)
    }

    pub fn object_familiarity(&self, obj: &PerceivedObject) -> f32 {
        let ds: Vec<f32> = obj.s.iter().zip(self.m_s.iter()).map(|(&x, &m)| x - m).collect();
        let dc: Vec<f32> = obj.c.iter().zip(self.m_c.iter()).map(|(&x, &m)| x - m).collect();
        let mut best = 0.0f32;
        for (j, p) in self.lexicon.iter().enumerate() {
            if self.lex_count[j] < 2 {
                continue;
            }
            let (es, ec) = self.expectations(p);
            let ns = l2_norm(&es) + 1e-8;
            let nc = l2_norm(&ec) + 1e-8;
            let sc = (dot(&es, &ds) / ns).max(dot(&ec, &dc) / nc);
            if sc > best {
                best = sc;
            }
        }
        best
    }

    pub fn bind_experience(&mut self, a: &[f32], objs: &[PerceivedObject], order: &[usize], eta: f32, is_novel_word: bool) {
        let mut weights = vec![1.0f32; objs.len()];
        if is_novel_word && objs.len() > 1 && self.lexicon.len() >= 4 {
            let fams: Vec<f32> = objs.iter().map(|o| self.object_familiarity(o)).collect();
            let mut min_i = 0usize;
            let mut min_f = f32::INFINITY;
            for (i, &f) in fams.iter().enumerate() {
                if f < min_f {
                    min_f = f;
                    min_i = i;
                }
            }
            for (i, w) in weights.iter_mut().enumerate() {
                *w = if i == min_i { 3.5 } else { 0.15 };
            }
        }

        // Normalize per-scene learning rate by number of fixated objects so cluttered scenes
        // do not inject excess distractor variance into the active word's synapses
        let scene_scale = 1.0 / ((order.len().max(1) as f32).sqrt());

        for &idx in order {
            let o = &objs[idx];
            let eff_eta = eta * weights[idx] * scene_scale;
            let ds: Vec<f32> = o.s.iter().zip(self.m_s.iter()).map(|(&x, &m)| x - m).collect();
            let dc: Vec<f32> = o.c.iter().zip(self.m_c.iter()).map(|(&x, &m)| x - m).collect();
            for i in 0..self.n_w {
                // Pre-synaptically gated BCM rule: silent word neurons (`a[i] == 0`) release zero
                // glutamate and NEVER suffer cross-word interference when other words are spoken!
                if a[i] <= 0.05 {
                    continue;
                }
                let spec = (1.0 - 2.2 * self.m_w[i]).max(0.05);
                let pre_lr = a[i] * spec * eff_eta;
                let row_s = &mut self.bs[i * self.n_s..(i + 1) * self.n_s];
                for (w_val, &dsv) in row_s.iter_mut().zip(ds.iter()) {
                    *w_val += pre_lr * (dsv - 0.02 * (*w_val));
                }
                let row_c = &mut self.bc[i * self.n_c..(i + 1) * self.n_c];
                for (w_val, &dcv) in row_c.iter_mut().zip(dc.iter()) {
                    *w_val += pre_lr * (dcv - 0.02 * (*w_val));
                }
            }
            for (m, &x) in self.m_s.iter_mut().zip(o.s.iter()) {
                *m += 0.01 * (x - *m);
            }
            for (m, &x) in self.m_c.iter_mut().zip(o.c.iter()) {
                *m += 0.01 * (x - *m);
            }
        }
        for (m, &x) in self.m_w.iter_mut().zip(a.iter()) {
            *m += 0.01 * (x - *m);
        }
    }


    /// Recall the grounded word-form assembly `a` (dim `n_w`) from a visual shape assembly `s`
    /// (`stream_mode = 1`), colour assembly `c` (`stream_mode = 0`), or combined (`stream_mode = 2`).
    pub fn recall_word_assembly(&self, s: &[f32], c: &[f32], stream_mode: u8) -> Vec<f32> {
        let ds: Vec<f32> = s.iter().zip(self.m_s.iter()).map(|(&x, &m)| x - m).collect();
        let dc: Vec<f32> = c.iter().zip(self.m_c.iter()).map(|(&x, &m)| x - m).collect();
        let mut a_hat = vec![0.0f32; self.n_w];
        for i in 0..self.n_w {
            let spec = (1.0 - 2.2 * self.m_w[i]).max(0.05);
            let mut val = 0.0f32;
            if stream_mode == 0 || stream_mode == 2 {
                let row_c = &self.bc[i * self.n_c..(i + 1) * self.n_c];
                val += dot(row_c, &dc);
            }
            if stream_mode == 1 || stream_mode == 2 {
                let row_s = &self.bs[i * self.n_s..(i + 1) * self.n_s];
                val += dot(row_s, &ds);
            }
            a_hat[i] = (val * spec).max(0.0);
        }
        // Snap to nearest discovered lexicon prototype using zero-mean centered cosine
        if !self.lexicon.is_empty() {
            let da_hat: Vec<f32> = a_hat.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
            let na = l2_norm(&da_hat) + 1e-8;
            let mut best_j = 0usize;
            let mut best_sc = f32::NEG_INFINITY;
            for (j, proto) in self.lexicon.iter().enumerate() {
                let dp: Vec<f32> = proto.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
                let sc = dot(&da_hat, &dp) / (na * (l2_norm(&dp) + 1e-8));
                if sc > best_sc {
                    best_sc = sc;
                    best_j = j;
                }
            }
            return self.lexicon[best_j].clone();
        }
        a_hat
    }

    pub fn lesion(&mut self) {
        self.bs.fill(0.0);
        self.bc.fill(0.0);
    }
}
