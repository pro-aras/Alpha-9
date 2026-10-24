//! Alpha 9 Native C-ABI Bridge (`libalpha9.so`) — Hosts Live Alpha-9 + Companion Organisms.

use std::ffi::CString;
use std::os::raw::c_char;
use std::slice;
use std::sync::{Mutex, OnceLock};
use alpha_iface::SensoryFrame;
use alpha_organism::genome::Genome;
use alpha_organism::Organism;
use alpha_probe::probe_telemetry;

static ORGANISM: OnceLock<Mutex<Organism>> = OnceLock::new();
static COMPANION: OnceLock<Mutex<Organism>> = OnceLock::new();
static LAST_ALPHA_PCM: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();
static LAST_COMP_PCM: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();

fn with_org<F, R>(f: F) -> R
where
    F: FnOnce(&mut Organism) -> R,
{
    let mtx = ORGANISM.get_or_init(|| Mutex::new(Organism::new(Genome::default())));
    let mut guard = mtx.lock().unwrap();
    f(&mut guard)
}

fn with_pair<F, R>(f: F) -> R
where
    F: FnOnce(&mut Organism, &mut Organism, &mut Vec<f32>, &mut Vec<f32>) -> R,
{
    let mtx_a = ORGANISM.get_or_init(|| Mutex::new(Organism::new(Genome::default())));
    let mtx_c = COMPANION.get_or_init(|| {
        Mutex::new(Organism::new(Genome {
            seed: 0xC049_A110_5566_7788,
            ..Genome::default()
        }))
    });
    let mtx_pa = LAST_ALPHA_PCM.get_or_init(|| Mutex::new(Vec::new()));
    let mtx_pc = LAST_COMP_PCM.get_or_init(|| Mutex::new(Vec::new()));
    let mut ga = mtx_a.lock().unwrap();
    let mut gc = mtx_c.lock().unwrap();
    let mut gpa = mtx_pa.lock().unwrap();
    let mut gpc = mtx_pc.lock().unwrap();
    f(&mut ga, &mut gc, &mut gpa, &mut gpc)
}

fn json_c_ptr(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn alpha9_bridge_init() -> *mut c_char {
    with_pair(|org, comp, _, _| {
        let t = probe_telemetry(org);
        let tc = probe_telemetry(comp);
        json_c_ptr(format!(
            "{{\"ok\":true,\"version\":\"9.0.0-biophysical\",\"alpha_ram\":{},\"companion_ram\":{},\"distress\":{:.4},\"valence\":{:.4}}}",
            t.memory_bytes, tc.memory_bytes, t.distress, t.valence
        ))
    })
}

/// Step both living organisms (`Alpha-9` and `Companion`) in a shared live world/game/camera frame.
/// External audio (`pcm_ptr`) plus the peer's previous vocalization reach each organism's cochlea;
/// neither organism's output is forced or scripted.
#[no_mangle]
pub unsafe extern "C" fn alpha9_bridge_step_pair(
    rgb_ptr: *const u8,
    rgb_len: usize,
    width: u16,
    height: u16,
    pcm_ptr: *const f32,
    pcm_len: usize,
    alpha_flux_ptr: *const f32,
    comp_flux_ptr: *const f32,
) -> *mut c_char {
    let rgb: &[u8] = if !rgb_ptr.is_null() && rgb_len > 0 {
        slice::from_raw_parts(rgb_ptr, rgb_len)
    } else {
        &[]
    };
    let ext_pcm: &[f32] = if !pcm_ptr.is_null() && pcm_len > 0 {
        slice::from_raw_parts(pcm_ptr, pcm_len)
    } else {
        &[]
    };
    let mut a_flux = [0.0f32; 4];
    if !alpha_flux_ptr.is_null() {
        a_flux.copy_from_slice(slice::from_raw_parts(alpha_flux_ptr, 4));
    }
    let mut c_flux = [0.0f32; 4];
    if !comp_flux_ptr.is_null() {
        c_flux.copy_from_slice(slice::from_raw_parts(comp_flux_ptr, 4));
    }

    with_pair(|alpha, comp, prev_a_pcm, prev_c_pcm| {
        let hear_a: &[f32] = if !ext_pcm.is_empty() { ext_pcm } else { prev_c_pcm.as_slice() };
        let out_a = alpha.step(&SensoryFrame {
            retina_rgb: rgb,
            width,
            height,
            pcm_16khz: hear_a,
            body_flux: a_flux,
        });

        let hear_c: &[f32] = if !ext_pcm.is_empty() { ext_pcm } else { out_a.pcm_out.as_slice() };
        let out_c = comp.step(&SensoryFrame {
            retina_rgb: rgb,
            width,
            height,
            pcm_16khz: hear_c,
            body_flux: c_flux,
        });

        *prev_a_pcm = out_a.pcm_out;
        *prev_c_pcm = out_c.pcm_out;

        let (stm_a, ltm_a, concepts_a, words_a, meanings_a) = alpha.mind.living_type_counts();
        let (stm_c, ltm_c, concepts_c, words_c, meanings_c) = comp.mind.living_type_counts();

        json_c_ptr(format!(
            "{{\"alpha\":{{\"step\":{},\"gaze\":[{:.3},{:.3}],\"action\":[{:.3},{:.3},{:.3},{:.3}],\"vocal\":[{:.3},{:.3},{:.3},{:.3}],\"body\":[{:.3},{:.3},{:.3},{:.3}],\"valence\":{:.4},\"distress\":{:.4},\"cells_total\":{},\"born\":{},\"split\":{},\"merged\":{},\"died\":{},\"stm\":{},\"ltm\":{},\"concepts\":{},\"words\":{},\"meanings\":{},\"cheap_steps\":{},\"waking_steps\":{}}},\"companion\":{{\"step\":{},\"gaze\":[{:.3},{:.3}],\"action\":[{:.3},{:.3},{:.3},{:.3}],\"vocal\":[{:.3},{:.3},{:.3},{:.3}],\"body\":[{:.3},{:.3},{:.3},{:.3}],\"valence\":{:.4},\"distress\":{:.4},\"cells_total\":{},\"born\":{},\"split\":{},\"stm\":{},\"ltm\":{},\"concepts\":{},\"words\":{},\"meanings\":{}}}}}",
            alpha.step_count,
            out_a.gaze_xy[0], out_a.gaze_xy[1],
            out_a.action_drive[0], out_a.action_drive[1], out_a.action_drive[2], out_a.action_drive[3],
            out_a.vocal_tract[0], out_a.vocal_tract[1], out_a.vocal_tract[2], out_a.vocal_tract[3],
            alpha.body.state[0], alpha.body.state[1], alpha.body.state[2], alpha.body.state[3],
            alpha.body.valence, alpha.body.distress,
            alpha.mind.cells.len(), alpha.mind.cells_born, alpha.mind.cells_split,
            alpha.mind.cells_merged, alpha.mind.cells_died_handoff,
            stm_a, ltm_a, concepts_a, words_a, meanings_a,
            alpha.mind.cheap_cell_steps, alpha.mind.waking_brain_steps,
            comp.step_count,
            out_c.gaze_xy[0], out_c.gaze_xy[1],
            out_c.action_drive[0], out_c.action_drive[1], out_c.action_drive[2], out_c.action_drive[3],
            out_c.vocal_tract[0], out_c.vocal_tract[1], out_c.vocal_tract[2], out_c.vocal_tract[3],
            comp.body.state[0], comp.body.state[1], comp.body.state[2], comp.body.state[3],
            comp.body.valence, comp.body.distress,
            comp.mind.cells.len(), comp.mind.cells_born, comp.mind.cells_split,
            stm_c, ltm_c, concepts_c, words_c, meanings_c
        ))
    })
}

#[no_mangle]
pub extern "C" fn alpha9_bridge_get_telemetry() -> *mut c_char {
    with_org(|org| {
        let t = probe_telemetry(org);
        json_c_ptr(format!(
            "{{\"version\":\"9.0.0-biophysical\",\"step_count\":{},\"memory_bytes\":{},\"ram_pressure_ok\":{},\"lexicon_prototypes\":{},\"body_state\":[{:.3},{:.3},{:.3},{:.3}],\"distress\":{:.4},\"valence\":{:.4},\"arousal\":{:.4},\"discovery_archive_size\":{}}}",
            t.step_count,
            t.memory_bytes,
            t.ram_pressure_ok,
            t.lexicon_prototypes,
            t.body_state[0],
            t.body_state[1],
            t.body_state[2],
            t.body_state[3],
            t.distress,
            t.valence,
            t.arousal,
            t.discovery_archive_size
        ))
    })
}

#[no_mangle]
pub unsafe extern "C" fn alpha9_bridge_free_str(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
