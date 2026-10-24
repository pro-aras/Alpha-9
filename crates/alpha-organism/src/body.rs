//! Biophysical Homeostasis, Energy Budget, Feelings Engine (Session Transcript Q1, Q2, Q6)
//! and Quadripartite Human Insular Cortex Architecture (Quabs & Parvizi et al., Nature 2026).

/// Quadripartite functional architecture of the insular cortex (Quabs et al., Nature 2026):
/// - Region I (Dorsal-Posterior Granular): Somatotopic nociception (pain) & temperature.
/// - Region II (Ventral-Posterior Dysgranular + Area Id4): Somatosensory & oral-vocal proprioception.
/// - Region III (Mid-Anterior): Viscerosensory internal organ states (gut/ATP, chest pressure) & anxiety.
/// - Region IV (Anterior Polar Silent Insula): Salience detection, Action-Mode Network switch,
///   and asymmetric top-down cognitive-interoceptive modulation (shaping how the body is felt).
#[derive(Clone, Debug)]
pub struct QuadripartiteInsula {
    /// Region I: [nociceptive_pain, thermal_deviation, limb_noxia]
    pub region1_pain_temp: [f32; 3],
    /// Region II: [cutaneous_tingling, deep_pressure, id4_oral_cavity_reafference]
    pub region2_somatosensory: [f32; 3],
    /// Region III: [visceral_gut_nausea, cardiopulmonary_fullness, visceral_anxiety, cognitive_anxiety]
    pub region3_visceral_anxiety: [f32; 4],
    /// Region IV: [salience_burst, action_mode_switch, top_down_cognitive_prior]
    pub region4_polar_salience: [f32; 3],
    /// Asymmetric CCEP connectivity weights: Region IV <-> Region III/I (forward vs top-down)
    pub w_bottom_up_i_to_iii: f32,
    pub w_bottom_up_iii_to_iv: f32,
    pub w_top_down_iv_to_iii: f32,
    pub w_top_down_pfc_to_iv: f32,
}

impl Default for QuadripartiteInsula {
    fn default() -> Self {
        Self {
            region1_pain_temp: [0.0; 3],
            region2_somatosensory: [0.0; 3],
            region3_visceral_anxiety: [0.0; 4],
            region4_polar_salience: [0.0; 3],
            w_bottom_up_i_to_iii: 0.45,
            w_bottom_up_iii_to_iv: 0.50,
            w_top_down_iv_to_iii: 0.65,
            w_top_down_pfc_to_iv: 0.70,
        }
    }
}

impl QuadripartiteInsula {
    /// Step the 4 insular regions combining ascending bodily signals with top-down cognitive framing.
    /// Returns (felt_bodily_distress_modulation, action_mode_switch_signal).
    pub fn step_insula(
        &mut self,
        nociceptive_input: [f32; 3],
        somatic_oral_input: [f32; 3],
        visceral_organ_deficit: [f32; 2],
        surprise_salience: f32,
        top_down_pfc_reappraisal: f32,
    ) -> (f32, f32) {
        // Region I: Dorsal-posterior granular (pain & temperature)
        for i in 0..3 {
            self.region1_pain_temp[i] = (0.3 * self.region1_pain_temp[i] + 0.7 * nociceptive_input[i]).clamp(0.0, 1.5);
        }
        // Region II: Ventral-posterior dysgranular + Id4 oral cavity somatosensory
        for i in 0..3 {
            self.region2_somatosensory[i] = (0.3 * self.region2_somatosensory[i] + 0.7 * somatic_oral_input[i]).clamp(0.0, 1.5);
        }

        // Region IV: Anterior polar silent insula — integrates unexpected salience (independent of Go/Withhold sign)
        // and receives asymmetric top-down narrative/context from PFC.
        let top_down_story = self.w_top_down_pfc_to_iv * top_down_pfc_reappraisal.clamp(-1.0, 1.0);
        self.region4_polar_salience[2] = top_down_story;

        let r1_drive = (self.region1_pain_temp[0] + self.region1_pain_temp[1] + self.region1_pain_temp[2]) / 3.0;
        let raw_visceral = 0.5 * (visceral_organ_deficit[0] + visceral_organ_deficit[1])
            + self.w_bottom_up_i_to_iii * r1_drive;

        // Top-down cognitive prior from Region IV shapes how Region III feels the visceral state
        // (Parvizi 2026: the insula is not just reading the body, it influences the story of how the body is felt).
        let top_down_mod = self.w_top_down_iv_to_iii * top_down_story;
        let felt_visceral = (raw_visceral - top_down_mod).clamp(0.0, 1.5);

        self.region3_visceral_anxiety[0] = (0.25 * self.region3_visceral_anxiety[0] + 0.75 * visceral_organ_deficit[0]).clamp(0.0, 1.5);
        self.region3_visceral_anxiety[1] = (0.25 * self.region3_visceral_anxiety[1] + 0.75 * visceral_organ_deficit[1]).clamp(0.0, 1.5);
        // Visceral/somatic vs cognitive anxiety components (Quabs et al. Fig. 3 BAI dissociation)
        self.region3_visceral_anxiety[2] = felt_visceral;
        self.region3_visceral_anxiety[3] = (0.6 * felt_visceral - 0.4 * top_down_story).clamp(0.0, 1.5);

        let r3_to_r4 = self.w_bottom_up_iii_to_iv * self.region3_visceral_anxiety[2];
        let salience = (surprise_salience.abs() + 0.5 * r3_to_r4).clamp(0.0, 1.5);
        self.region4_polar_salience[0] = salience;
        // Action-mode switch triggers on salient events regardless of motor polarity (Go vs Withhold)
        self.region4_polar_salience[1] = if salience > 0.35 { salience } else { 0.1 * salience };

        (felt_visceral, self.region4_polar_salience[1])
    }
}

#[derive(Clone, Debug)]
pub struct HomeostaticBody {
    /// [energy_atp, thermal_integrity, social_contact, epistemic_comfort]
    pub state: [f32; 4],
    pub set_point: [f32; 4],
    pub distress_prev: f32,
    pub distress: f32,
    pub valence: f32,
    pub arousal: f32,
    pub basal_drain: f32,
    pub insula: QuadripartiteInsula,
}

impl Default for HomeostaticBody {
    fn default() -> Self {
        let set_point = [0.85, 1.00, 0.75, 0.80];
        let state = [0.65, 0.95, 0.40, 0.60];
        let mut d = 0.0f32;
        for i in 0..4 {
            let diff = state[i] - set_point[i];
            d += diff * diff;
        }
        Self {
            state,
            set_point,
            distress_prev: d,
            distress: d,
            valence: 0.0,
            arousal: 0.0,
            basal_drain: 0.004,
            insula: QuadripartiteInsula::default(),
        }
    }
}

impl HomeostaticBody {
    #[inline]
    pub fn compute_distress_for(state: &[f32; 4], set_point: &[f32; 4]) -> f32 {
        let mut d = 0.0f32;
        for i in 0..4 {
            let diff = (set_point[i] - state[i]).max(0.0);
            d += diff * diff;
        }
        d
    }

    pub fn step_homeostasis(&mut self, flux: &[f32; 4], spike_fraction: f32, surprise: f32) -> (f32, f32) {
        self.step_homeostasis_with_cognition(flux, spike_fraction, surprise, 0.0)
    }

    pub fn step_homeostasis_with_cognition(
        &mut self,
        flux: &[f32; 4],
        spike_fraction: f32,
        surprise: f32,
        top_down_reappraisal: f32,
    ) -> (f32, f32) {
        let spike_cost = 0.015 * spike_fraction;
        self.state[0] = (self.state[0] - self.basal_drain - spike_cost + flux[0]).clamp(0.0, self.set_point[0]);
        self.state[1] = (self.state[1] - 0.5 * self.basal_drain + flux[1]).clamp(0.0, self.set_point[1]);
        self.state[2] = (self.state[2] - 0.8 * self.basal_drain + flux[2]).clamp(0.0, self.set_point[2]);
        self.state[3] = (self.state[3] - 0.4 * self.basal_drain + flux[3]).clamp(0.0, self.set_point[3]);

        let thermal_err = (self.set_point[1] - self.state[1]).max(0.0);
        let energy_err = (self.set_point[0] - self.state[0]).max(0.0);
        let social_err = (self.set_point[2] - self.state[2]).max(0.0);

        let (felt_visceral, action_switch) = self.insula.step_insula(
            [thermal_err, thermal_err * 0.8, 0.0],
            [spike_fraction, 0.0, 0.0],
            [energy_err, social_err],
            surprise,
            top_down_reappraisal,
        );

        let raw_d = Self::compute_distress_for(&self.state, &self.set_point);
        let d = (0.75 * raw_d + 0.25 * felt_visceral * felt_visceral).max(0.0);
        let mean_flux = 0.25 * (flux[0] + flux[1] + flux[2] + flux[3]);
        let phasic = if mean_flux.abs() > 0.02 { 0.65 * mean_flux } else { 0.0 };
        let delta_d = if mean_flux.abs() <= 0.02 {
            (self.distress_prev - d).clamp(-0.02, 0.02)
        } else {
            self.distress_prev - d
        };
        self.valence = delta_d + phasic;
        self.arousal = self.valence.abs() + 0.35 * surprise + 0.15 * action_switch;
        self.distress_prev = d;
        self.distress = d;
        (self.valence, self.arousal)
    }

    #[inline]
    pub fn emotional_lr(&self, base_lr: f32) -> f32 {
        base_lr * (1.0 + 5.0 * self.arousal)
    }
}
