use std::path::Path;
use alpha_exams::run_all_gates;

fn main() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../alpha-organism/src");
    let r = run_all_gates(&src_dir);
    println!("==========================================================================");
    println!("ALPHA 9 BIOPHYSICAL, FEELING-DRIVEN & SENSORY EMERGENCE GATES");
    println!("==========================================================================");
    println!(
        "[G0] Anti-Cheat Zero-Strings in alpha-organism ({} files): {}",
        r.g0_files_scanned,
        if r.g0_anti_cheat_zero_strings { "PASS" } else { "FAIL" }
    );
    println!(
        "[B1] Biophysical Heat-Map (ATP limit={}, 30%-damaged cue settle={:.3}, bound word follow={:.3}): {}",
        r.b1_atp_energy_constrained,
        r.b1_corrupted_cue_recovery,
        r.b1_bound_word_recall,
        if r.b1_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[B2] Learning-Progress Curiosity (learnable={:+.4} vs noisy-TV={:+.4} vs static={:+.4}): {}",
        r.b2_curiosity_learnable_prog,
        r.b2_curiosity_noisy_tv_prog,
        r.b2_curiosity_static_prog,
        if r.b2_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[B3] Homeostatic Feelings (relief valence={:+.3}, harm valence={:+.3}, emotional lr gain={:.2}x): {}",
        r.b3_positive_valence_on_relief,
        r.b3_negative_valence_on_harm,
        r.b3_emotional_memory_gain_ratio,
        if r.b3_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[B4] Novel Speech by Predicted Effect on Inner State (need->utterance acc={:.3}, chance=0.250): {}",
        r.b4_feeling_speech_need_acc,
        if r.b4_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[B5] Quadripartite Insula (Quabs & Parvizi 2026: post_spec={:+.3}, top_down_delta={:+.4}, polar_switch={:+.3}): {}",
        r.b5_posterior_pain_specificity,
        r.b5_top_down_relief_delta,
        r.b5_polar_action_mode_invariance,
        if r.b5_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[B6] Biological Log-Polar Retina (scale_invariance={:.3}, fovea/periphery_acuity={:.2}x): {}",
        r.b6_logpolar_scale_invariance,
        r.b6_logpolar_fovea_vs_periph_ratio,
        if r.b6_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S0] Senses (V1 selectivity={:.3}, A1 held-out speaker ABX={:.3}): {}",
        r.s0_v1_selectivity,
        r.s0_a1_abx_acc,
        if r.s0_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S1] Saffran Segmentation (word={:.3} vs part-word={:.3}): {}",
        r.s1_saffran_word_score,
        r.s1_saffran_partword_score,
        if r.s1_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S2] Grounding C1 Held-Out Speakers: acc={:.3} (chance={:.3}) | C2 Voice Models: acc={:.3} | C3 Novel Combos: acc={:.3}",
        r.s2_c1_heldout_spk_acc, r.s2_c1_chance, r.s2_c2_heldout_voice_acc, r.s2_c3_novel_combos_acc
    );
    println!(
        "[S2] Controls C4 Lesion: acc={:.3} | C5 Shuffled: acc={:.3} | C6 Streams (col={:.3} vs shp={:.3}) | C7 Purity={:.3}: {}",
        r.s2_c4_lesion_acc,
        r.s2_c5_shuffled_acc,
        r.s2_c6_colour_stream_pref,
        r.s2_c6_shape_stream_pref,
        r.s2_c7_lexicon_purity,
        if r.s2_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S3] Babbling -> Vocal Imitation (imitated_cos={:.3} vs null={:.3}): {}",
        r.s3_babble_imitation_cos,
        r.s3_babble_null_cos,
        if r.s3_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S4] Visual->Spoken Object Naming (spoken_acc={:.3} vs chance=0.091): {}",
        r.s4_spoken_naming_acc,
        if r.s4_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[S5] Two-Word Compositional Speech [colour+shape] (phrase_acc={:.3} vs chance=0.009): {}",
        r.s5_two_word_phrase_acc,
        if r.s5_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[D1] Active Falsification & Discovery (Hopfield_40pat={:.3} vs Sparse_kWTA={:.3}, false_claims_falsified={:.0}%, discovered_law_cos={:.3}): {}",
        r.d1_hopfield_40pat_recall,
        r.d1_sparse_kwta_40pat_recall,
        r.d1_false_claims_falsified_rate * 100.0,
        r.d1_discovered_law_cos,
        if r.d1_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[L1] Longitudinal Self-Advancement (T0 naive={:.3} -> T1 early={:.3} -> T2 mature={:.3} -> T3 post-sleep={:.3}, pruned={} synapses): {}",
        r.l1_acc_t0_naive,
        r.l1_acc_t1_early,
        r.l1_acc_t2_late,
        r.l1_acc_t3_post_sleep,
        r.l1_pruned_synapses,
        if r.l1_passed { "PASS" } else { "FAIL" }
    );
    println!(
        "[HW] Organism Heap Footprint: {:.2} MB (Budget <= 32.0 MB): {}",
        r.organism_ram_mb,
        if r.ram_budget_ok { "PASS" } else { "FAIL" }
    );
    println!("==========================================================================");
    assert!(r.g0_anti_cheat_zero_strings, "G0 anti-cheat failed");
    assert!(r.b1_passed, "B1 biophysical heatmap gate failed");
    assert!(r.b2_passed, "B2 curiosity learning-progress gate failed");
    assert!(r.b3_passed, "B3 homeostatic feelings gate failed");
    assert!(r.b4_passed, "B4 feeling-driven speech gate failed");
    assert!(r.b5_passed, "B5 quadripartite insula gate failed");
    assert!(r.b6_passed, "B6 biological log-polar retina gate failed");
    assert!(r.s0_passed, "S0 senses gate failed");
    assert!(r.s1_passed, "S1 Saffran segmentation gate failed");
    assert!(r.s2_passed, "S2 cross-situational grounding gate failed");
    assert!(r.s3_passed, "S3 vocal babbling imitation gate failed");
    assert!(r.s4_passed, "S4 spoken naming gate failed");
    assert!(r.s5_passed, "S5 two-word compositional speech gate failed");
    assert!(r.d1_passed, "D1 causal falsification and discovery gate failed");
    assert!(r.l1_passed, "L1 longitudinal self-advancement gate failed");
    assert!(r.ram_budget_ok, "Hardware RAM budget gate failed");
}
