use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }

    #[inline]
    pub fn standard_normal(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    pub fn shuffle_indices(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            idx.swap(i, j);
        }
        idx
    }
}

#[inline]
pub fn l2_norm(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

#[inline]
pub fn normalize_l2_in_place(x: &mut [f32]) {
    let n = l2_norm(x) + 1e-8;
    for v in x.iter_mut() {
        *v /= n;
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

pub fn kwta_in_place(x: &mut [f32], k: usize) {
    if k == 0 {
        x.fill(0.0);
        return;
    }
    if k >= x.len() {
        return;
    }
    let mut pairs: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
    pairs.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
    let mut keep = vec![false; x.len()];
    for &(idx, _) in &pairs[..k] {
        keep[idx] = true;
    }
    for (i, v) in x.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        }
    }
}

pub fn median_f32(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut c = vals.to_vec();
    let mid = c.len() / 2;
    c.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    c[mid]
}

pub fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert!(n.is_power_of_two() && n == im.len());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let ang = -2.0 * PI / (len as f32);
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        for i in (0..n).step_by(len) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..half {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * w_re - im[i + k + half] * w_im;
                let v_im = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
            }
        }
        len <<= 1;
    }
}
