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

/// Living Visceral / Somatic Organ Cell inside `HomeostaticBody`:
/// Follows the same cellular laws as cortical cells (hunger/feeding on physiological flux,
/// fullness/satiety, variety tracking, allostatic adaptation, and inter-organ co-bonds).
#[derive(Clone, Copy, Debug)]
pub struct VisceralOrganCell {
    pub organ_role: u8,
    pub pattern: [f32; 4],
    pub energy: f32,
    pub fullness: f32,
    pub variety: f32,
    pub adaptation_shield: f32,
    pub fed_count: u32,
    pub co_bonds: [f32; 8],
}

impl Default for VisceralOrganCell {
    fn default() -> Self {
        Self {
            organ_role: 0,
            pattern: [0.25; 4],
            energy: 0.75,
            fullness: 0.0,
            variety: 0.0,
            adaptation_shield: 0.10,
            fed_count: 0,
            co_bonds: [0.0; 8],
        }
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
    /// CellML v2.0 Neurovascular / Astrocyte-Neuron Lactate Shuttle (ANLS) & Adenosine Sleep-Pressure Physiome:
    pub cbf_flow: f32,
    pub astrocyte_lactate_mm: f32,
    pub neuronal_atp_mm: f32,
    pub extracellular_adenosine_um: f32,
    pub insula: QuadripartiteInsula,
    /// 8 Living Visceral & Insular Organ Cells (4 homeostatic organ systems + 4 insular regions)
    pub organ_cells: [VisceralOrganCell; 8],
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
        let mut organ_cells = [VisceralOrganCell::default(); 8];
        for (idx, cell) in organ_cells.iter_mut().enumerate() {
            cell.organ_role = idx as u8;
            cell.pattern = [0.10; 4];
            cell.pattern[idx % 4] = 0.85;
        }
        Self {
            state,
            set_point,
            distress_prev: d,
            distress: d,
            valence: 0.0,
            arousal: 0.0,
            basal_drain: 0.004,
            cbf_flow: 1.0,
            astrocyte_lactate_mm: 1.40,
            neuronal_atp_mm: 2.35,
            extracellular_adenosine_um: 0.12,
            insula: QuadripartiteInsula::default(),
            organ_cells,
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

    /// Coupled CellML v2.0 Neurovascular CBF + Astrocyte-Neuron Lactate Shuttle (ANLS) + Adenosine ODE step:
    pub fn step_cellml_physiome(&mut self, spike_fraction: f32, nutrient_glucose_flux: f32) {
        let neural_work = spike_fraction.clamp(0.0, 1.0);
        // Neurovascular coupling: cortical spiking dilates arterioles via NO/prostaglandin signaling
        let target_cbf = (1.0 + 0.85 * neural_work).clamp(0.75, 2.20);
        self.cbf_flow = 0.82 * self.cbf_flow + 0.18 * target_cbf;

        // Astrocyte glycolysis driven by glutamate uptake + capillary glucose delivery
        let glucose_in = 0.035 * self.cbf_flow + 0.45 * nutrient_glucose_flux.max(0.0);
        let anls_shuttle = 0.042 * self.astrocyte_lactate_mm * (0.35 + neural_work);
        self.astrocyte_lactate_mm = (self.astrocyte_lactate_mm + glucose_in - anls_shuttle).clamp(0.25, 3.20);

        // Mitochondrial oxidative phosphorylation of shuttled lactate vs Na+/K+-ATPase spike pump cost
        let atp_synth = 0.055 * anls_shuttle * self.cbf_flow;
        let atp_pump_cost = 0.018 + 0.065 * neural_work;
        self.neuronal_atp_mm = (self.neuronal_atp_mm + atp_synth - atp_pump_cost).clamp(0.40, 3.00);

        // Basal forebrain extracellular adenosine rises with ATP hydrolysis, clears during quiet rest
        let ado_prod = 0.024 * neural_work + 0.015 * (2.20 - self.neuronal_atp_mm).max(0.0);
        let glymphatic_clear = if neural_work < 0.08 { 0.045 } else { 0.014 };
        self.extracellular_adenosine_um =
            (self.extracellular_adenosine_um * (1.0 - glymphatic_clear) + ado_prod).clamp(0.05, 2.50);
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
        self.step_cellml_physiome(spike_fraction, flux[0]);

        // Step living visceral organ cells via Carandini-Heeger Divisive Receptor Conductance
        let mut active_organs = [0.0f32; 8];
        let mut signal_vec = [0.0f32; 4];
        for k in 0..4 {
            signal_vec[k] = (self.set_point[k] - self.state[k]).max(0.0) + (-flux[k]).max(0.0);
        }
        let sig_energy_sq: f32 = signal_vec.iter().map(|x| x * x).sum();
        let sig_norm = sig_energy_sq.sqrt() + 1e-6;
        // Divisive normalization across visceral interneuron pool (sigma = 0.08)
        let div_denom = (0.0064 + sig_energy_sq).sqrt();
        let sig_div = [
            signal_vec[0] / div_denom,
            signal_vec[1] / div_denom,
            signal_vec[2] / div_denom,
            signal_vec[3] / div_denom,
        ];

        for idx in 0..8 {
            let cell = &mut self.organ_cells[idx];
            cell.fullness = (cell.fullness * 0.65 - 0.04).clamp(0.0, 1.0);
            cell.energy = (cell.energy - 0.002).clamp(0.10, 1.50);
            let p_energy_sq: f32 = cell.pattern.iter().map(|x| x * x).sum();
            let g_conductance = (cell.pattern[0].max(0.0) * sig_div[0]
                + cell.pattern[1].max(0.0) * sig_div[1]
                + cell.pattern[2].max(0.0) * sig_div[2]
                + cell.pattern[3].max(0.0) * sig_div[3])
                / (p_energy_sq.sqrt() + 1e-6);
            let hunger = (1.0 - 0.55 * cell.fullness).clamp(0.20, 1.0);
            if g_conductance * hunger > 0.40 && sig_norm > 0.02 {
                let lr = (0.12 * hunger).clamp(0.02, 0.20);
                let mut diff_sq = 0.0f32;
                for k in 0..4 {
                    let err = sig_div[k] - cell.pattern[k];
                    diff_sq += err * err;
                    cell.pattern[k] = (1.0 - lr) * cell.pattern[k] + lr * sig_div[k];
                }
                cell.variety = 0.90 * cell.variety + 0.10 * diff_sq.sqrt();
                cell.energy = (cell.energy + 0.08 * sig_norm).clamp(0.10, 1.50);
                cell.fullness = (cell.fullness + 0.35).clamp(0.0, 1.0);
                cell.adaptation_shield = (cell.adaptation_shield + 0.03 * cell.energy).clamp(0.05, 0.65);
                cell.fed_count = cell.fed_count.saturating_add(1);
                active_organs[idx] = g_conductance * cell.energy;
            } else {
                cell.adaptation_shield = (cell.adaptation_shield * 0.995).clamp(0.05, 0.65);
            }
        }
        // Update decaying co-bonds among simultaneously active visceral organ cells
        for i in 0..8 {
            for j in 0..8 {
                self.organ_cells[i].co_bonds[j] *= 0.992;
                if i != j && active_organs[i] > 0.10 && active_organs[j] > 0.10 {
                    let b = &mut self.organ_cells[i].co_bonds[j];
                    *b = (*b + 0.08 * active_organs[i] * active_organs[j]).clamp(0.0, 1.5);
                }
            }
        }

        let spike_cost = 0.015 * spike_fraction;
        for k in 0..4 {
            let drain_scale = match k {
                0 => 1.0,
                1 => 0.5,
                2 => 0.8,
                _ => 0.4,
            };
            let eff_flux = if flux[k] < 0.0 {
                let shield = 0.35 * self.organ_cells[k].adaptation_shield;
                flux[k] * (1.0 - shield).clamp(0.35, 1.0)
            } else {
                flux[k]
            };
            let extra_cost = if k == 0 { spike_cost } else { 0.0 };
            self.state[k] = (self.state[k] - drain_scale * self.basal_drain - extra_cost + eff_flux)
                .clamp(0.0, self.set_point[k]);
        }

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
