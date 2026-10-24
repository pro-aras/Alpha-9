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
        self.transduce_with_end(pcm).0
    }

    pub fn transduce_with_end(&self, pcm: &[f32]) -> (Vec<Vec<f32>>, usize, usize) {
        if pcm.is_empty() {
            return (vec![vec![0.0; self.nb]; 3], 0, 0);
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
            let start = first_voiced.unwrap_or(0);
            // Biophysical envelope-trough boundary detection:
            // Track smoothed envelope E_slow(t) from `start`. If the stream contains multiple
            // concatenated words/phrases, detect the inter-word amplitude trough after the primary
            // vocal nucleus (>= 16 frames) where smoothed energy dips below `floor + 0.22 * (peak - floor)`
            // for at least 3 consecutive frames, followed by a subsequent energy rise.
            let mut peak_en = thresh;
            let mut end_idx = last_voiced + 1;
            let mut low_run = 0usize;
            let mut trough_candidate = None;
            for t in start..=last_voiced {
                let e0 = frame_energies[t];
                let e1 = frame_energies[(t + 1).min(last_voiced)];
                let e2 = frame_energies[t.saturating_sub(1)];
                let e_smooth = 0.25 * e2 + 0.50 * e0 + 0.25 * e1;
                if e_smooth > peak_en {
                    peak_en = e_smooth;
                }
                let dur = t - start;
                let dip_thresh = floor + 0.22 * (peak_en - floor);
                if dur >= 16 && e_smooth < dip_thresh {
                    low_run += 1;
                    if low_run >= 2 && trough_candidate.is_none() {
                        trough_candidate = Some(t);
                    }
                } else {
                    if let Some(tc) = trough_candidate {
                        // A second distinct vocalization burst rose after the trough!
                        if e_smooth > floor + 0.45 * (peak_en - floor) && (last_voiced - t) >= 8 {
                            end_idx = tc;
                            break;
                        }
                    }
                    low_run = 0;
                }
            }
            (start, end_idx.max(start + 3).min(energies.len()))
        } else {
            (0, energies.len())
        };

        (
            Self::normalize_segment(&energies[i0..i1], self.nb),
            i1,
            voiced_count,
        )
    }

    fn normalize_segment(slice: &[Vec<f32>], nb: usize) -> Vec<Vec<f32>> {
        let mut seg: Vec<Vec<f32>> = slice.to_vec();
        let t_len = seg.len().max(1) as f32;

        for b in 0..nb {
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
            mean /= nb as f32;
            let mut var = 0.0f32;
            for &v in row.iter() {
                let d = v - mean;
                var += d * d;
            }
            let std = (var / (nb as f32)).sqrt() + 1e-3;
            for v in row.iter_mut() {
                *v /= std;
            }
        }
        seg
    }

    /// Segment a continuous audio stream (1 word, 2 words, or a full multi-word sentence)
    /// into a sequence of normalized cochlear segments via envelope-trough tracking.
    pub fn transduce_stream(&self, pcm: &[f32]) -> Vec<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        let mut offset = 0usize;
        let min_step = self.hop * 16;
        while offset + min_step <= pcm.len() {
            let slice = &pcm[offset..];
            let (seg, i1_end, voiced_cnt) = self.transduce_with_end(slice);
            if seg.is_empty() || voiced_cnt < 8 {
                break;
            }
            let consumed_frames = (i1_end + 3).max(16);
            out.push(seg);
            offset += consumed_frames * self.hop;
            if out.len() >= 16 {
                break;
            }
        }
        if out.is_empty() {
            out.push(self.transduce(pcm));
        }
        out
    }
}

pub struct Retina;

impl Retina {
    /// Biologically grounded retinal segmentation using a Bilinearly-Interpolated Local-Median
    /// Horizontal-Cell Syncytium (`6 x 4` macro-grid of wide-field `96 x 96` spatial medians).
    /// - Because the spatial median has a 50% breakdown point, compact foreground objects (`< 20%` of
    ///   a wide horizontal-cell field) contribute ZERO contamination to the background estimate
    ///   (no halos around objects, zero hollowing of object interiors).
    /// - Because the `6 x 4` grid of local medians is bilinearly interpolated across `(x, y)`, smooth
    ///   directional illumination/shadow gradients across the scene are tracked and cancelled out locally,
    ///   providing both shadow-invariant segmentation and Retinex opponent colour constancy.
    pub fn segment(rgb: &[u8], width: usize, height: usize) -> (Vec<RetinalBlob>, Vec<bool>) {
        let n_pix = width * height;
        if rgb.len() < n_pix * 3 || n_pix == 0 || width < 8 || height < 8 {
            return (Vec::new(), Vec::new());
        }
        let mut r_ch = vec![0.0f32; n_pix];
        let mut g_ch = vec![0.0f32; n_pix];
        let mut b_ch = vec![0.0f32; n_pix];
        for i in 0..n_pix {
            r_ch[i] = rgb[i * 3] as f32 + 1.0;
            g_ch[i] = rgb[i * 3 + 1] as f32 + 1.0;
            b_ch[i] = rgb[i * 3 + 2] as f32 + 1.0;
        }

        const GX: usize = 6;
        const GY: usize = 4;
        let mut grid_r = [0.0f32; GX * GY];
        let mut grid_g = [0.0f32; GX * GY];
        let mut grid_b = [0.0f32; GX * GY];
        let rad_x = (width / 4).max(24);
        let rad_y = (height / 3).max(24);
        let mut buf_r = Vec::with_capacity(1024);
        let mut buf_g = Vec::with_capacity(1024);
        let mut buf_b = Vec::with_capacity(1024);

        for gy in 0..GY {
            let cy = (gy * (height - 1)) / (GY - 1);
            let y0 = cy.saturating_sub(rad_y);
            let y1 = (cy + rad_y + 1).min(height);
            for gx in 0..GX {
                let cx = (gx * (width - 1)) / (GX - 1);
                let x0 = cx.saturating_sub(rad_x);
                let x1 = (cx + rad_x + 1).min(width);
                buf_r.clear();
                buf_g.clear();
                buf_b.clear();
                let mut y = y0;
                while y < y1 {
                    let mut x = x0;
                    while x < x1 {
                        let p = y * width + x;
                        buf_r.push(r_ch[p]);
                        buf_g.push(g_ch[p]);
                        buf_b.push(b_ch[p]);
                        x += 3;
                    }
                    y += 3;
                }
                grid_r[gy * GX + gx] = median_f32(&buf_r);
                grid_g[gy * GX + gx] = median_f32(&buf_g);
                grid_b[gy * GX + gx] = median_f32(&buf_b);
            }
        }

        let mut opp_l = vec![0.0f32; n_pix];
        let mut opp_rg = vec![0.0f32; n_pix];
        let mut opp_by = vec![0.0f32; n_pix];
        let mut dev = vec![0.0f32; n_pix];

        for y in 0..height {
            let fy = (y as f32) * ((GY - 1) as f32) / ((height - 1).max(1) as f32);
            let gy0 = (fy.floor() as usize).min(GY - 1);
            let gy1 = (gy0 + 1).min(GY - 1);
            let wy = fy - (gy0 as f32);

            for x in 0..width {
                let fx = (x as f32) * ((GX - 1) as f32) / ((width - 1).max(1) as f32);
                let gx0 = (fx.floor() as usize).min(GX - 1);
                let gx1 = (gx0 + 1).min(GX - 1);
                let wx = fx - (gx0 as f32);

                let i00 = gy0 * GX + gx0;
                let i01 = gy0 * GX + gx1;
                let i10 = gy1 * GX + gx0;
                let i11 = gy1 * GX + gx1;

                let bg_r = (1.0 - wy) * ((1.0 - wx) * grid_r[i00] + wx * grid_r[i01])
                    + wy * ((1.0 - wx) * grid_r[i10] + wx * grid_r[i11]);
                let bg_g = (1.0 - wy) * ((1.0 - wx) * grid_g[i00] + wx * grid_g[i01])
                    + wy * ((1.0 - wx) * grid_g[i10] + wx * grid_g[i11]);
                let bg_b = (1.0 - wy) * ((1.0 - wx) * grid_b[i00] + wx * grid_b[i01])
                    + wy * ((1.0 - wx) * grid_b[i10] + wx * grid_b[i11]);
                let bg_mean = ((bg_r + bg_g + bg_b) / 3.0).max(4.0);

                let p = y * width + x;
                let (r, g, b) = (r_ch[p], g_ch[p], b_ch[p]);
                let l_val = ((r + g + b) / 3.0) / bg_mean;
                let rg_val = (r - g) / bg_mean;
                let by_val = (b - 0.5 * (r + g)) / bg_mean;
                opp_l[p] = l_val;
                opp_rg[p] = rg_val;
                opp_by[p] = by_val;

                let bgo_rg = (bg_r - bg_g) / bg_mean;
                let bgo_by = (bg_b - 0.5 * (bg_r + bg_g)) / bg_mean;
                let dl = l_val - 1.0;
                let drg = rg_val - bgo_rg;
                let dby = by_val - bgo_by;
                dev[p] = (dl * dl + drg * drg + dby * dby).sqrt();
            }
        }

        let med_dev = median_f32(&dev);
        let abs_diff: Vec<f32> = dev.iter().map(|&d| (d - med_dev).abs()).collect();
        let mad = median_f32(&abs_diff) + 1e-4;
        let thresh = med_dev + 6.0 * mad;

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
            if ys.len() < 20 || ys.len() > (n_pix / 2) {
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
            let r_m = r_max * ((alpha * u).exp() - 1.0) / denom;
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

    /// Dual-Stream Retinotopic Silhouette + Radial Contour Harmonic Sheet (`n_rings * n_sectors`):
    /// Combines the anti-aliased 2D retinotopic silhouette (preserving aspect ratio, top-bottom polarity,
    /// and horizontal/vertical/diagonal edge orientations) with an interior angular contour-deviation wave
    /// `r_smooth(theta) - mean_r` so that interior V1 patches are never flat-zero and encode the number
    /// and angular phase of corners/lobes/concavities (3 for triangle, 4 for square/diamond/cross, 5 for star, 0 for circle).
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
        let mut mask = vec![0.0f32; s * s];
        for (&y, &x) in blob.ys.iter().zip(blob.xs.iter()) {
            let ry = (y as usize - y0) + oy;
            let rx = (x as usize - x0) + ox;
            mask[ry * s + rx] = 1.0;
        }

        // Smooth 3x3 retinal horizontal/bipolar receptive field to suppress pixelation jitter
        let mut smooth = mask.clone();
        for y in 1..(s - 1) {
            for x in 1..(s - 1) {
                smooth[y * s + x] = 0.40 * mask[y * s + x]
                    + 0.15 * (mask[(y - 1) * s + x] + mask[(y + 1) * s + x] + mask[y * s + x - 1] + mask[y * s + x + 1]);
            }
        }

        let cy = (s as f32 - 1.0) * 0.5;
        let cx = (s as f32 - 1.0) * 0.5;
        let r_max = (h.max(w) as f32) * 0.50;

        // Measure outer contour radius r(theta) across 36 polar angles and smooth circularly
        const N_ANG: usize = 36;
        let mut r_raw = [0.0f32; N_ANG];
        for a in 0..N_ANG {
            let theta = 2.0 * PI * (a as f32) / (N_ANG as f32);
            let st = theta.sin();
            let ct = theta.cos();
            let mut max_r = 0.0f32;
            for k in 1..=32 {
                let rk = r_max * (k as f32) / 32.0;
                let iy = (cy + rk * st).round().clamp(0.0, (s - 1) as f32) as usize;
                let ix = (cx + rk * ct).round().clamp(0.0, (s - 1) as f32) as usize;
                if smooth[iy * s + ix] > 0.35 {
                    max_r = rk / r_max.max(1.0);
                }
            }
            r_raw[a] = max_r;
        }
        let mut r_prof = [0.0f32; N_ANG];
        for a in 0..N_ANG {
            r_prof[a] = 0.25 * r_raw[(a + N_ANG - 1) % N_ANG] + 0.50 * r_raw[a] + 0.25 * r_raw[(a + 1) % N_ANG];
        }
        let mean_r: f32 = r_prof.iter().sum::<f32>() / (N_ANG as f32);

        let mut out = vec![0.0f32; n_rings * n_sectors];
        let scale_y = (s - 1) as f32 / ((n_rings.saturating_sub(1)).max(1) as f32);
        let scale_x = (s - 1) as f32 / ((n_sectors.saturating_sub(1)).max(1) as f32);

        for fy in 0..n_rings {
            let sy = (fy as f32) * scale_y;
            let iy0 = (sy.floor() as usize).min(s - 1);
            let iy1 = (iy0 + 1).min(s - 1);
            let wy = sy - (iy0 as f32);
            let dy_n = ((fy as f32) - ((n_rings - 1) as f32) * 0.5) / ((n_rings as f32) * 0.5);

            for fx in 0..n_sectors {
                let sx = (fx as f32) * scale_x;
                let ix0 = (sx.floor() as usize).min(s - 1);
                let ix1 = (ix0 + 1).min(s - 1);
                let wx = sx - (ix0 as f32);
                let dx_n = ((fx as f32) - ((n_sectors - 1) as f32) * 0.5) / ((n_sectors as f32) * 0.5);

                let v00 = smooth[iy0 * s + ix0];
                let v01 = smooth[iy0 * s + ix1];
                let v10 = smooth[iy1 * s + ix0];
                let v11 = smooth[iy1 * s + ix1];
                let sil = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
                    + wy * ((1.0 - wx) * v10 + wx * v11);

                // Polar angle & normalized radius of this receptor `(fy, fx)`
                let rad = (dy_n * dy_n + dx_n * dx_n).sqrt().min(1.2);
                let mut ang = dy_n.atan2(dx_n);
                if ang < 0.0 {
                    ang += 2.0 * PI;
                }
                let a_idx = ((ang / (2.0 * PI)) * (N_ANG as f32)).floor() as usize % N_ANG;
                let radial_dev = r_prof[a_idx] - mean_r;

                // Superimpose anti-aliased 2D silhouette with angular radial-lobe ridge field
                out[fy * n_sectors + fx] = sil * (1.0 + 1.8 * radial_dev - 0.35 * rad) + 0.65 * radial_dev * (-2.0 * (rad - r_prof[a_idx]).powi(2)).exp();
            }
        }
        out
    }

    pub fn fovea_form(blob: &RetinalBlob, fov: usize) -> Vec<f32> {
        Self::log_polar_form(blob, fov, fov)
    }
}
