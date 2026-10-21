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

    pub fn update_lexicon(&mut self, a: &[f32], k_w: f32) -> (usize, bool) {
        if !self.lexicon.is_empty() {
            let mut best_j = 0usize;
            let mut best_ov = f32::NEG_INFINITY;
            for (j, p) in self.lexicon.iter().enumerate() {
                let ov = dot(a, p) / k_w;
                if ov > best_ov {
                    best_ov = ov;
                    best_j = j;
                }
            }
            if best_ov > self.overlap_thresh {
                self.lex_count[best_j] += 1;
                let cnt = self.lex_count[best_j] as f32;
                for (pv, &av) in self.lexicon[best_j].iter_mut().zip(a.iter()) {
                    *pv += (av - *pv) / cnt;
                }
                return (best_j, false);
            }
        }
        let j = self.lexicon.len();
        self.lexicon.push(a.to_vec());
        self.lex_count.push(1);
        (j, true)
    }

    pub fn expectations(&self, a: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut es = vec![0.0f32; self.n_s];
        let mut ec = vec![0.0f32; self.n_c];
        for i in 0..self.n_w {
            let da = a[i] - self.m_w[i];
            if da.abs() < 1e-6 {
                continue;
            }
            let row_s = &self.bs[i * self.n_s..(i + 1) * self.n_s];
            for (s_val, &w_val) in es.iter_mut().zip(row_s.iter()) {
                *s_val += da * w_val;
            }
            let row_c = &self.bc[i * self.n_c..(i + 1) * self.n_c];
            for (c_val, &w_val) in ec.iter_mut().zip(row_c.iter()) {
                *c_val += da * w_val;
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
        let da: Vec<f32> = a.iter().zip(self.m_w.iter()).map(|(&x, &m)| x - m).collect();
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

        for &idx in order {
            let o = &objs[idx];
            let eff_eta = eta * weights[idx];
            let ds: Vec<f32> = o.s.iter().zip(self.m_s.iter()).map(|(&x, &m)| x - m).collect();
            let dc: Vec<f32> = o.c.iter().zip(self.m_c.iter()).map(|(&x, &m)| x - m).collect();
            for i in 0..self.n_w {
                let dai = da[i] * eff_eta;
                if dai.abs() < 1e-7 {
                    continue;
                }
                let row_s = &mut self.bs[i * self.n_s..(i + 1) * self.n_s];
                for (w_val, &dsv) in row_s.iter_mut().zip(ds.iter()) {
                    *w_val += dai * dsv;
                }
                let row_c = &mut self.bc[i * self.n_c..(i + 1) * self.n_c];
                for (w_val, &dcv) in row_c.iter_mut().zip(dc.iter()) {
                    *w_val += dai * dcv;
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

    pub fn lesion(&mut self) {
        self.bs.fill(0.0);
        self.bc.fill(0.0);
    }
}
