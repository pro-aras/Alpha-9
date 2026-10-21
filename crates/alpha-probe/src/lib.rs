//! Law 4: Read-only instrument probes (`alpha-probe`).

use alpha_organism::Organism;

#[derive(Clone, Debug)]
pub struct OrganismTelemetry {
    pub step_count: u64,
    pub memory_bytes: usize,
    pub ram_pressure_ok: bool,
    pub lexicon_prototypes: usize,
    pub body_state: [f32; 4],
    pub distress: f32,
    pub valence: f32,
    pub arousal: f32,
    pub discovery_archive_size: usize,
}

pub fn probe_telemetry(org: &Organism) -> OrganismTelemetry {
    OrganismTelemetry {
        step_count: org.step_count,
        memory_bytes: org.memory_bytes(),
        ram_pressure_ok: org.ram_pressure_ok(),
        lexicon_prototypes: org.binding.lexicon.len(),
        body_state: org.body.state,
        distress: org.body.distress,
        valence: org.body.valence,
        arousal: org.body.arousal,
        discovery_archive_size: org.discovery.candidates.len(),
    }
}
