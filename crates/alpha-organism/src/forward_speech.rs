//! Novel Speech & Action Chosen by Predicted Effect on Inner State (Session Transcript Q6, Pages 9-10).

use crate::body::HomeostaticBody;
use crate::util::{dot, normalize_l2_in_place, Rng64};

#[derive(Clone, Debug)]
pub struct FeelingSpeechEngine {
    pub pat_dim: usize,
    pub w_context_to_utter: Vec<f32>,
    pub w_body_to_pat: Vec<f32>,
    pub w_forward_body: Vec<f32>,
}

impl FeelingSpeechEngine {
    pub fn new(pat_dim: usize, rng: &mut Rng64) -> Self {
        let mut w_context_to_utter = vec![0.0f32; pat_dim * pat_dim];
        for v in &mut w_context_to_utter {
            *v = 0.02 * rng.standard_normal();
        }
        let mut w_body_to_pat = vec![0.0f32; pat_dim * 4];
        for v in &mut w_body_to_pat {
            *v = 0.02 * rng.standard_normal();
        }
        let mut w_forward_body = vec![0.0f32; 4 * pat_dim];
        for v in &mut w_forward_body {
            *v = 0.01 * rng.standard_normal();
        }
        Self {
            pat_dim,
            w_context_to_utter,
            w_body_to_pat,
            w_forward_body,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.w_context_to_utter.capacity()
            + self.w_body_to_pat.capacity()
            + self.w_forward_body.capacity())
            * 4
    }

    pub fn predict_next_body(&self, body: &[f32; 4], utter_pat: &[f32]) -> [f32; 4] {
        let mut pred = *body;
        for b in 0..4 {
            let row = &self.w_forward_body[b * self.pat_dim..(b + 1) * self.pat_dim];
            let delta = dot(row, utter_pat).tanh() * 0.35;
            pred[b] = (pred[b] + delta).clamp(0.0, 1.2);
        }
        pred
    }

    pub fn choose_utterance(
        &self,
        body: &HomeostaticBody,
        context: &[f32],
        repertoire: &[Vec<f32>],
        n_cands: usize,
        temp: f32,
        rng: &mut Rng64,
    ) -> (usize, Vec<f32>, f32) {
        let need = [
            (body.set_point[0] - body.state[0]).max(0.0),
            (body.set_point[1] - body.state[1]).max(0.0),
            (body.set_point[2] - body.state[2]).max(0.0),
            (body.set_point[3] - body.state[3]).max(0.0),
        ];
        let mut base = vec![0.0f32; self.pat_dim];
        for i in 0..self.pat_dim {
            let row_c = &self.w_context_to_utter[i * self.pat_dim..(i + 1) * self.pat_dim];
            let row_b = &self.w_body_to_pat[i * 4..(i + 1) * 4];
            base[i] = dot(row_c, context) + dot(row_b, &need);
        }

        let mut cands: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n_cands + repertoire.len());
        for (idx, proto) in repertoire.iter().enumerate() {
            let mut c = proto.clone();
            for v in &mut c {
                *v += 0.08 * temp * rng.standard_normal();
            }
            normalize_l2_in_place(&mut c);
            cands.push((idx, c));
        }
        while cands.len() < n_cands.max(repertoire.len() + 1) {
            let mut c = base.clone();
            for v in &mut c {
                *v += temp * rng.standard_normal();
            }
            normalize_l2_in_place(&mut c);
            let idx = cands.len();
            cands.push((idx, c));
        }

        let d_now = body.distress;
        let mut gains = Vec::with_capacity(cands.len());
        let mut max_g = f32::NEG_INFINITY;
        for (_, c) in &cands {
            let pred_b = self.predict_next_body(&body.state, c);
            let d_pred = HomeostaticBody::compute_distress_for(&pred_b, &body.set_point);
            let g = d_now - d_pred;
            if g > max_g {
                max_g = g;
            }
            gains.push(g);
        }

        let eff_temp = temp.max(0.02);
        let mut probs = Vec::with_capacity(gains.len());
        let mut sum_p = 0.0f32;
        for &g in &gains {
            let p = ((g - max_g) / eff_temp).exp();
            probs.push(p);
            sum_p += p;
        }
        let r = rng.next_f32() * sum_p;
        let mut acc = 0.0f32;
        let mut chosen = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= r {
                chosen = i;
                break;
            }
        }
        (cands[chosen].0, cands[chosen].1.clone(), gains[chosen])
    }

    pub fn learn_from_outcome(
        &mut self,
        body_before: &[f32; 4],
        body_after: &[f32; 4],
        set_point: &[f32; 4],
        context: &[f32],
        utter_pat: &[f32],
        valence: f32,
        arousal: f32,
        base_lr: f32,
    ) {
        let pred_after = self.predict_next_body(body_before, utter_pat);
        let eta_fwd = 0.18;
        for b in 0..4 {
            let err = (body_after[b] - body_before[b]) - (pred_after[b] - body_before[b]);
            let row = &mut self.w_forward_body[b * self.pat_dim..(b + 1) * self.pat_dim];
            for (wv, &uv) in row.iter_mut().zip(utter_pat.iter()) {
                *wv += eta_fwd * err * uv;
            }
        }

        let lr = base_lr * (1.0 + 5.0 * arousal);
        let sign_v = if valence >= 0.0 { 1.0 } else { -0.5 };
        let need = [
            (set_point[0] - body_before[0]).max(0.0),
            (set_point[1] - body_before[1]).max(0.0),
            (set_point[2] - body_before[2]).max(0.0),
            (set_point[3] - body_before[3]).max(0.0),
        ];
        for i in 0..self.pat_dim {
            let ui = utter_pat[i] * lr * sign_v;
            let row_c = &mut self.w_context_to_utter[i * self.pat_dim..(i + 1) * self.pat_dim];
            for (wv, &cv) in row_c.iter_mut().zip(context.iter()) {
                *wv += ui * cv;
            }
            let row_b = &mut self.w_body_to_pat[i * 4..(i + 1) * 4];
            for (wv, &nv) in row_b.iter_mut().zip(need.iter()) {
                *wv += ui * nv;
            }
        }
    }
}
