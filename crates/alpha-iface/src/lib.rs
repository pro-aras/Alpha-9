//! Alpha 9 Sensory-Motor Skin (`alpha-iface`).
//! Law 1: Only continuous physical numbers cross the skin. Zero String or &str.

#[derive(Clone, Copy, Debug)]
pub struct SensoryFrame<'a> {
    pub retina_rgb: &'a [u8],
    pub width: u16,
    pub height: u16,
    pub pcm_16khz: &'a [f32],
    /// Physical body inputs: [nourishment_atp_flux, ambient_warmth, social_touch_voice, physical_safety]
    pub body_flux: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct MotorFrame {
    pub vocal_tract: [f32; 8],
    pub pcm_out: Vec<f32>,
    pub gaze_xy: [f32; 2],
    pub fixation_blob: Option<usize>,
    /// Physical actions: [seek_nourishment, seek_warmth, seek_social, explore_novelty]
    pub action_drive: [f32; 4],
}

impl Default for MotorFrame {
    fn default() -> Self {
        Self {
            vocal_tract: [0.0; 8],
            pcm_out: Vec::new(),
            gaze_xy: [0.5, 0.5],
            fixation_blob: None,
            action_drive: [0.0; 4],
        }
    }
}
