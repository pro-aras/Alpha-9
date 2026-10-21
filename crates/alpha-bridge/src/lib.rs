//! Alpha 9 Native C-ABI Bridge (`libalpha9.so`).

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};
use alpha_organism::genome::Genome;
use alpha_organism::Organism;
use alpha_probe::probe_telemetry;

static ORGANISM: OnceLock<Mutex<Organism>> = OnceLock::new();

fn with_org<F, R>(f: F) -> R
where
    F: FnOnce(&mut Organism) -> R,
{
    let mtx = ORGANISM.get_or_init(|| Mutex::new(Organism::new(Genome::default())));
    let mut guard = mtx.lock().unwrap();
    f(&mut guard)
}

fn json_c_ptr(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn alpha9_bridge_init() -> *mut c_char {
    with_org(|org| {
        let t = probe_telemetry(org);
        json_c_ptr(format!(
            "{{\"ok\":true,\"version\":\"9.0.0-biophysical\",\"memory_bytes\":{},\"distress\":{:.4},\"valence\":{:.4},\"arousal\":{:.4}}}",
            t.memory_bytes, t.distress, t.valence, t.arousal
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
