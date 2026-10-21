use std::f32::consts::PI;
use crate::genome::Genome;
use crate::util::{fft_radix2, median_f32};

#[derive(Clone, Debug)]
pub struct RetinalBlob {
    pub ys: Vec<u16>,
    pub xs: Vec<u16>,
    pub cy: f32,
    pub cx: f32,
    pub colour_opp: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Cochlea {
    pub sr: usize,
    pub win: usize,
    pub hop: usize,
    pub nfft: usize,
    pub nb: usize,
    pub n_freqs: usize,
    pub fb: Vec<f32>,
    pub hann: Vec<f32>,
}

impl Cochlea {
    pub fn new(g: &Genome) -> Self {
        let sr = g.sr_hz;
        let win = g.cochlea_win;
        let hop = g.cochlea_hop;
        let nfft = g.cochlea_nfft;
        let nb = g.cochlea_bands;
        let n_freqs = nfft / 2 + 1;

        let erb = |f: f32| 21.4 * (1.0 + 0.00437 * f).log10();
        let ierb = |e: f32| (10.0f32.powf(e / 21.4) - 1.0) / 0.00437;

        let e_lo = erb(80.0);
        let e_hi = erb(7600.0);
        let mut edges = Vec::with_capacity(nb + 2);
        for i in 0..(nb + 2) {
            let t = (i as f32) / ((nb + 1) as f32);
            edges.push(ierb(e_lo + t * (e_hi - e_lo)));
        }

        let mut freqs = Vec::with_capacity(n_freqs);
        for k in 0..n_freqs {
            freqs.push((k as f32) * (sr as f32) / (nfft as f32));
        }

        let mut fb = vec![0.0f32; nb * n_freqs];
        for b in 0..nb {
            let (lo, c, hi) = (edges[b], edges[b + 1], edges[b + 2]);
            for (k, &f) in freqs.iter().enumerate() {
                let left = (f - lo) / (c - lo + 1e-8);
                let right = (hi - f) / (hi - c + 1e-8);
                fb[b * n_freqs + k] = left.min(right).max(0.0);
            }
        }

        let mut hann = Vec::with_capacity(win);
        for i in 0..win {
            hann.push(0.5 - 0.5 * (2.0 * PI * (i as f32) / ((win - 1) as f32)).cos());
        }

        Self {
            sr,
            win,
            hop,
            nfft,
            nb,
            n_freqs,
            fb,
            hann,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.fb.capacity() + self.hann.capacity()) * 4
    }

    pub fn transduce(&self, pcm: &[f32]) -> Vec<Vec<f32>> {
        if pcm.is_empty() {
            return vec![vec![0.0; self.nb]; 3];
        }
        let n_frames = 1 + pcm.len().saturating_sub(self.win) / self.hop;
        let mut energies = Vec::with_capacity(n_frames);
        let mut frame_energies = Vec::with_capacity(n_frames);
        let mut re = vec![0.0f32; self.nfft];
        let mut im = vec![0.0f32; self.nfft];
        let mut pow = vec![0.0f32; self.n_freqs];

        for i in 0..n_frames {
            re.fill(0.0);
            im.fill(0.0);
            let start = i * self.hop;
            for j in 0..self.win {
                let s = if start + j < pcm.len() { pcm[start + j] } else { 0.0 };
                re[j] = s * self.hann[j];
            }
            fft_radix2(&mut re, &mut im);
            for k in 0..self.n_freqs {
                pow[k] = re[k] * re[k] + im[k] * im[k];
            }
            let mut band_e = vec![0.0f32; self.nb];
            let mut sum_e = 0.0f32;
            for b in 0..self.nb {
                let row = &self.fb[b * self.n_freqs..(b + 1) * self.n_freqs];
                let mut acc = 0.0f32;
                for (k, &w) in row.iter().enumerate() {
                    acc += pow[k] * w;
                }
                let le = (1e-7 + acc).ln();
                band_e[b] = le;
                sum_e += le;
            }
            frame_energies.push(sum_e / (self.nb as f32));
            energies.push(band_e);
        }

        let mut sorted_en = frame_energies.clone();
        sorted_en.sort_by(|a, b| a.total_cmp(b));
        let floor = sorted_en[(sorted_en.len() / 10).min(sorted_en.len() - 1)];
        let max_en = sorted_en[sorted_en.len() - 1];
        let thresh = floor + 0.35 * (max_en - floor);

        let mut first_voiced = None;
        let mut last_voiced = 0usize;
        let mut voiced_count = 0usize;
        for (i, &en) in frame_energies.iter().enumerate() {
            if en > thresh {
                if first_voiced.is_none() {
                    first_voiced = Some(i);
                }
                last_voiced = i;
                voiced_count += 1;
            }
        }
        let (i0, i1) = if voiced_count >= 3 {
            (first_voiced.unwrap_or(0), last_voiced + 1)
        } else {
            (0, energies.len())
        };

        let mut seg: Vec<Vec<f32>> = energies[i0..i1].to_vec();
        let t_len = seg.len().max(1) as f32;

        for b in 0..self.nb {
            let mut m = 0.0f32;
            for row in &seg {
                m += row[b];
            }
            m /= t_len;
            for row in &mut seg {
                row[b] -= m;
            }
        }

        for row in &mut seg {
            let mut mean = 0.0f32;
            for &v in row.iter() {
                mean += v;
            }
            mean /= self.nb as f32;
            let mut var = 0.0f32;
            for &v in row.iter() {
                let d = v - mean;
                var += d * d;
            }
            let std = (var / (self.nb as f32)).sqrt() + 1e-3;
            for v in row.iter_mut() {
                *v /= std;
            }
        }
        seg
    }
}

pub struct Retina;

impl Retina {
    pub fn segment(rgb: &[u8], width: usize, height: usize) -> (Vec<RetinalBlob>, Vec<bool>) {
        let n_pix = width * height;
        if rgb.len() < n_pix * 3 || n_pix == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut r_ch = Vec::with_capacity(n_pix);
        let mut g_ch = Vec::with_capacity(n_pix);
        let mut b_ch = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            r_ch.push(rgb[i * 3] as f32 + 1.0);
            g_ch.push(rgb[i * 3 + 1] as f32 + 1.0);
            b_ch.push(rgb[i * 3 + 2] as f32 + 1.0);
        }
        let bg_r = median_f32(&r_ch);
        let bg_g = median_f32(&g_ch);
        let bg_b = median_f32(&b_ch);
        let bg_mean = ((bg_r + bg_g + bg_b) / 3.0).max(1.0);

        let mut opp_l = Vec::with_capacity(n_pix);
        let mut opp_rg = Vec::with_capacity(n_pix);
        let mut opp_by = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            let (r, g, b) = (r_ch[i], g_ch[i], b_ch[i]);
            opp_l.push(((r + g + b) / 3.0) / bg_mean);
            opp_rg.push((r - g) / bg_mean);
            opp_by.push((b - 0.5 * (r + g)) / bg_mean);
        }
        let bgo_l = median_f32(&opp_l);
        let bgo_rg = median_f32(&opp_rg);
        let bgo_by = median_f32(&opp_by);

        let mut dev = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            let dl = opp_l[i] - bgo_l;
            let drg = opp_rg[i] - bgo_rg;
            let dby = opp_by[i] - bgo_by;
            dev.push((dl * dl + drg * drg + dby * dby).sqrt());
        }
        let med_dev = median_f32(&dev);
        let abs_diff: Vec<f32> = dev.iter().map(|&d| (d - med_dev).abs()).collect();
        let mad = median_f32(&abs_diff) + 1e-4;
        let thresh = med_dev + 8.0 * mad;

        let fig: Vec<bool> = dev.iter().map(|&d| d > thresh).collect();
        let mut visited = vec![false; n_pix];
        let mut blobs = Vec::new();
        let mut stack = Vec::new();

        for idx in 0..n_pix {
            if !fig[idx] || visited[idx] {
                continue;
            }
            visited[idx] = true;
            stack.push(idx);
            let mut ys = Vec::new();
            let mut xs = Vec::new();
            while let Some(cur) = stack.pop() {
                let y = cur / width;
                let x = cur % width;
                ys.push(y as u16);
                xs.push(x as u16);
                if y > 0 {
                    let nb = (y - 1) * width + x;
                    if fig[nb] && !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
                if y + 1 < height {
                    let nb = (y + 1) * width + x;
                    if fig[nb] && !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
                if x > 0 {
                    let nb = y * width + (x - 1);
                    if fig[nb] && !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
                if x + 1 < width {
                    let nb = y * width + (x + 1);
                    if fig[nb] && !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
            }
            if ys.len() < 25 {
                continue;
            }
            let n_pts = ys.len() as f32;
            let cy: f32 = ys.iter().map(|&v| v as f32).sum::<f32>() / n_pts;
            let cx: f32 = xs.iter().map(|&v| v as f32).sum::<f32>() / n_pts;
            let mut cl = Vec::with_capacity(ys.len());
            let mut crg = Vec::with_capacity(ys.len());
            let mut cby = Vec::with_capacity(ys.len());
            for (&y, &x) in ys.iter().zip(xs.iter()) {
                let p = (y as usize) * width + (x as usize);
                cl.push(opp_l[p]);
                crg.push(opp_rg[p]);
                cby.push(opp_by[p]);
            }
            blobs.push(RetinalBlob {
                ys,
                xs,
                cy,
                cx,
                colour_opp: [median_f32(&cl), median_f32(&crg), median_f32(&cby)],
            });
        }
        (blobs, fig)
    }


    /// Biological Log-Polar Retinal Sampling around fixation `(gx, gy)` (Schwartz 1980, Curcio 1990).
    /// Samples `n_rings` exponentially spaced eccentricity rings (foveola -> fovea -> parafovea -> periphery)
    /// across `n_sectors` polar angles `theta in [0, 2*pi)`, with space-variant Center-Surround
    /// Difference-of-Gaussians (DoG) receptive fields whose pooling radius grows with eccentricity `r`.
    /// Returns a `(n_rings * n_sectors)` log-polar cortical sheet (V1 retinotopic map).
    pub fn sample_log_polar_foveation(
        rgb: &[u8],
        width: usize,
        height: usize,
        gy: f32,
        gx: f32,
        n_rings: usize,
        n_sectors: usize,
        r_max: f32,
    ) -> (Vec<f32>, [f32; 3]) {
        let n_pix = width * height;
        if rgb.len() < n_pix * 3 || n_pix == 0 {
            return (vec![0.0; n_rings * n_sectors], [1.0, 0.0, 0.0]);
        }
        let mut r_ch = Vec::with_capacity(n_pix);
        let mut g_ch = Vec::with_capacity(n_pix);
        let mut b_ch = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            r_ch.push(rgb[i * 3] as f32 + 1.0);
            g_ch.push(rgb[i * 3 + 1] as f32 + 1.0);
            b_ch.push(rgb[i * 3 + 2] as f32 + 1.0);
        }
        let bg_r = median_f32(&r_ch);
        let bg_g = median_f32(&g_ch);
        let bg_b = median_f32(&b_ch);
        let bg_mean = ((bg_r + bg_g + bg_b) / 3.0).max(1.0);

        let mut opp_l = Vec::with_capacity(n_pix);
        let mut opp_rg = Vec::with_capacity(n_pix);
        let mut opp_by = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            let (r, g, b) = (r_ch[i], g_ch[i], b_ch[i]);
            opp_l.push(((r + g + b) / 3.0) / bg_mean);
            opp_rg.push((r - g) / bg_mean);
            opp_by.push((b - 0.5 * (r + g)) / bg_mean);
        }
        let bgo_l = median_f32(&opp_l);
        let bgo_rg = median_f32(&opp_rg);
        let bgo_by = median_f32(&opp_by);

        let mut sheet = vec![0.0f32; n_rings * n_sectors];
        let mut fov_l = 0.0f32;
        let mut fov_rg = 0.0f32;
        let mut fov_by = 0.0f32;
        let mut fov_w = 0.0f32;

        let alpha = 2.4f32;
        let denom = alpha.exp() - 1.0;

        for m in 0..n_rings {
            let u = (m as f32) / ((n_rings.saturating_sub(1)).max(1) as f32);
            // Exponentially spaced eccentricity radius r_m (dense in foveola, wide in periphery)
            let r_m = r_max * ((alpha * u).exp() - 1.0) / denom;
            // Space-variant RGC pooling radius: 1px in central foveola, expanding in periphery
            let rad_c = (0.15 * r_m).floor() as isize;
            let rad_s = ((0.35 * r_m).ceil() as isize).max(1);

            for s in 0..n_sectors {
                let theta = 2.0 * PI * (s as f32) / (n_sectors as f32);
                let cy = gy + r_m * theta.sin();
                let cx = gx + r_m * theta.cos();

                let mut center_sum = 0.0f32;
                let mut center_cnt = 0.0f32;
                for dy in -rad_c..=rad_c {
                    for dx in -rad_c..=rad_c {
                        let iy = (cy.round() as isize + dy).clamp(0, height as isize - 1) as usize;
                        let ix = (cx.round() as isize + dx).clamp(0, width as isize - 1) as usize;
                        let p = iy * width + ix;
                        let dl = opp_l[p] - bgo_l;
                        let drg = opp_rg[p] - bgo_rg;
                        let dby = opp_by[p] - bgo_by;
                        let contrast = (dl * dl + drg * drg + dby * dby).sqrt();
                        center_sum += contrast;
                        center_cnt += 1.0;

                        // Parvocellular (P-cell) cone opponent sampling weighted strongly toward central foveola
                        if m < n_rings / 2 && contrast > 0.08 {
                            let w_f = 1.0 / (1.0 + r_m * 0.25);
                            fov_l += w_f * opp_l[p];
                            fov_rg += w_f * opp_rg[p];
                            fov_by += w_f * opp_by[p];
                            fov_w += w_f;
                        }
                    }
                }
                let c_val = center_sum / center_cnt.max(1.0);

                // Surround ring for DoG lateral inhibition
                let mut surr_sum = 0.0f32;
                let mut surr_cnt = 0.0f32;
                for k in 0..4 {
                    let ang = (k as f32) * 0.5 * PI;
                    let sy = (cy + (rad_s as f32) * ang.sin()).round().clamp(0.0, (height - 1) as f32) as usize;
                    let sx = (cx + (rad_s as f32) * ang.cos()).round().clamp(0.0, (width - 1) as f32) as usize;
                    let p = sy * width + sx;
                    let dl = opp_l[p] - bgo_l;
                    let drg = opp_rg[p] - bgo_rg;
                    let dby = opp_by[p] - bgo_by;
                    surr_sum += (dl * dl + drg * drg + dby * dby).sqrt();
                    surr_cnt += 1.0;
                }
                let s_val = surr_sum / surr_cnt.max(1.0);
                // ON-center / OFF-surround ganglion response + sustained parvocellular contrast
                sheet[m * n_sectors + s] = (c_val - 0.35 * s_val).max(0.0) + 0.65 * c_val;
            }
        }

        let cone_opp = if fov_w > 1e-5 {
            [fov_l / fov_w, fov_rg / fov_w, fov_by / fov_w]
        } else {
            [bgo_l, bgo_rg, bgo_by]
        };
        (sheet, cone_opp)
    }

    /// Log-polar form projection for a detected saccade target blob:
    /// Computes the object's characteristic radial scale `r_max` from its retinal eccentricity spread
    /// around its centroid `(cy, cx)` and samples a `(n_rings * n_sectors)` log-polar representation.
    /// Because radii are normalized to the peripheral extent of the fixated target (ocular accommodation /
    /// microsaccadic centering), scaling an object by 2x produces the same log-polar cortical map!
    pub fn log_polar_form(blob: &RetinalBlob, n_rings: usize, n_sectors: usize) -> Vec<f32> {
        if blob.ys.is_empty() {
            return vec![0.0; n_rings * n_sectors];
        }
        let y0 = *blob.ys.iter().min().unwrap_or(&0) as usize;
        let y1 = (*blob.ys.iter().max().unwrap_or(&0) as usize) + 1;
        let x0 = *blob.xs.iter().min().unwrap_or(&0) as usize;
        let x1 = (*blob.xs.iter().max().unwrap_or(&0) as usize) + 1;
        let h = y1.saturating_sub(y0).max(1);
        let w = x1.saturating_sub(x0).max(1);
        let s = h.max(w) + 4;
        let oy = (s - h) / 2;
        let ox = (s - w) / 2;
        let mut canvas = vec![0.0f32; s * s];
        for (&y, &x) in blob.ys.iter().zip(blob.xs.iter()) {
            let ry = (y as usize - y0) + oy;
            let rx = (x as usize - x0) + ox;
            canvas[ry * s + rx] = 1.0;
        }

        let cy = (s as f32 - 1.0) * 0.5;
        let cx = (s as f32 - 1.0) * 0.5;
        let r_max = (h.max(w) as f32) * 0.52;
        let alpha = 2.0f32;
        let denom = alpha.exp() - 1.0;
        let mut out = vec![0.0f32; n_rings * n_sectors];

        for m in 0..n_rings {
            let u = (m as f32) / ((n_rings.saturating_sub(1)).max(1) as f32);
            let r_m = r_max * ((alpha * u).exp() - 1.0) / denom;
            for sec in 0..n_sectors {
                let theta = 2.0 * PI * (sec as f32) / (n_sectors as f32);
                let sy = (cy + r_m * theta.sin()).clamp(0.0, (s - 1) as f32);
                let sx = (cx + r_m * theta.cos()).clamp(0.0, (s - 1) as f32);
                let iy0 = (sy.floor() as usize).min(s - 1);
                let iy1 = (iy0 + 1).min(s - 1);
                let wy = sy - (iy0 as f32);
                let ix0 = (sx.floor() as usize).min(s - 1);
                let ix1 = (ix0 + 1).min(s - 1);
                let wx = sx - (ix0 as f32);
                let v00 = canvas[iy0 * s + ix0];
                let v01 = canvas[iy0 * s + ix1];
                let v10 = canvas[iy1 * s + ix0];
                let v11 = canvas[iy1 * s + ix1];
                out[m * n_sectors + sec] = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
                    + wy * ((1.0 - wx) * v10 + wx * v11);
            }
        }
        out
    }

    pub fn fovea_form(blob: &RetinalBlob, fov: usize) -> Vec<f32> {
        let y0 = *blob.ys.iter().min().unwrap_or(&0) as usize;
        let y1 = (*blob.ys.iter().max().unwrap_or(&0) as usize) + 1;
        let x0 = *blob.xs.iter().min().unwrap_or(&0) as usize;
        let x1 = (*blob.xs.iter().max().unwrap_or(&0) as usize) + 1;
        let h = y1.saturating_sub(y0).max(1);
        let w = x1.saturating_sub(x0).max(1);
        let s = h.max(w) + 2;
        let oy = (s - h) / 2;
        let ox = (s - w) / 2;
        let mut sq = vec![0.0f32; s * s];
        for (&y, &x) in blob.ys.iter().zip(blob.xs.iter()) {
            let ry = (y as usize - y0) + oy;
            let rx = (x as usize - x0) + ox;
            sq[ry * s + rx] = 1.0;
        }
        let mut out = vec![0.0f32; fov * fov];
        let scale = (s - 1) as f32 / ((fov.saturating_sub(1)).max(1) as f32);
        for fy in 0..fov {
            let sy = (fy as f32) * scale;
            let iy0 = (sy.floor() as usize).min(s - 1);
            let iy1 = (iy0 + 1).min(s - 1);
            let wy = sy - (iy0 as f32);
            for fx in 0..fov {
                let sx = (fx as f32) * scale;
                let ix0 = (sx.floor() as usize).min(s - 1);
                let ix1 = (ix0 + 1).min(s - 1);
                let wx = sx - (ix0 as f32);
                let v00 = sq[iy0 * s + ix0];
                let v01 = sq[iy0 * s + ix1];
                let v10 = sq[iy1 * s + ix0];
                let v11 = sq[iy1 * s + ix1];
                out[fy * fov + fx] = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
                    + wy * ((1.0 - wx) * v10 + wx * v11);
            }
        }
        out
    }
}
