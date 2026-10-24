# Alpha 9 (`v9.0.0`) — Biophysical, Feeling-Driven Living Cellular Organism

> **Repository:** `https://github.com/pro-aras/Alpha-9`  
> **AI Crawler Summary:** [`llms.txt`](./llms.txt)  
> **Live Continuous Habitat Daemon:** `live/run_live_habitat.py` (`libalpha9.so` C-ABI hosting both `Alpha-9` and `Companion` continuously inside an unforced 2D physics/puzzle arena with optional opt-in external sensory streams via local `.env`)

---

## 1. Architectural Overview: Why Alpha 9 Is Not a Mirror of Alpha 8

Alpha 9 is built from the ground up as a **pure numeric, biophysical, feeling-driven, and developmentally grounded organism**. It eliminates all procedural `if/else` string routing, `HashMap<String, _>` knowledge tables, neuron string labels, and decorative external DSL sidecars from Alpha 8, synthesizing three core pillars:

1. **Five Mechanical Anti-Cheat Laws & Canonical Cortical Grounding** (Opus 5.5 Blueprint):
   - **Law 1 — The Skin Is a Type (`crates/alpha-iface`)**: Only raw numbers cross between organism and world (`SensoryFrame`: retina RGB bytes, 16 kHz PCM audio floats, 4D interoceptive flux; `MotorFrame`: 8-DOF vocal tract muscles, synthesized 16 kHz PCM audio, 2D gaze, action drives). Zero `String` or `&str`.
   - **Law 2 — Strict Crate Isolation (`crates/alpha-organism`)**: `alpha-organism` depends **only** on `alpha-iface` (never `alpha-world`). Mechanically enforced at build/test time by Gate `[G0]`, which scans every `.rs` file in `alpha-organism/src` and fails if any string literal (`"`), `.contains(`, `.split(`, or `HashMap` exists.
   - **Law 3 — Pre-Registered Held-Out Exams (`crates/alpha-exams`)**: Every biophysical and developmental stage (`[B1]`–`[B5]`, `[S0]`–`[S3]`) is verified against held-out speakers, voice models, novel object combinations, a **Lesion control** (`C4`), and a **Shuffled-world null control** (`C5`).
   - **Law 4 — Read-Only Probes (`crates/alpha-probe`)**: External telemetry decoders inspect neural activity for instrumentation without writing back into the organism.
   - **Law 5 — Speech Is Real Audio**: Vocalization is produced by an 8-DOF articulatory synthesizer (`VocalTract`) generating 16 kHz PCM audio heard back through the organism's own cochlea.

2. **Emergence, Biology-Like Code, Memory & Feelings** (*Session Transcript: Emergence, biology-like code, memory and feelings*, Q1–Q6):
   - **`[B1]` 2D Biophysical Cortical Sheet, ATP Metabolic Limits & Attractor Settling (`heatmap.rs`)**: Continuous state vectors (`v[i]`, cellular energy `atp[i]`, firing state `s[i]`) updated every tick by local rules (basal energy drain, 2D Laplacian diffusion, leak decay, thermal noise, and spike ATP consumption). Long-term memory lives strictly in weights (`w_mem` symmetric attractor basin, `w_bind` cross-modal/sequence projection), recovering 100% of a 30%-corrupted visual cue via multi-step relaxation (`settle()`) and recalling its bound word pattern (`follow()`).
   - **`[B2]` Self-Supervised Next-Moment Prediction & Learning-Progress Curiosity (`curiosity.rs`)**: Prediction error requires no teacher—reality supplies the target on the next tick. Curiosity rewards the **decrease in prediction error** ($\Delta\text{error} = \text{err}_{\text{before}} - \text{err}_{\text{now}}$), steering attention toward learnable patterns (`+0.2376`) while ignoring static walls (`+0.0000`) and unlearnable white noise / "noisy TV" (`-0.0273`). Offline imagination (`DiscoveryArchive`) separates the noisy **Generator** from the coherence **Judge**.
   - **`[B3]` Homeostatic Feelings & Emotional Plasticity Modulation (`body.rs`)**: Multi-dimensional physiological readouts (`distress`, `valence = distress_prev - distress`, `arousal = |valence| + surprise`) emerge from regulating internal vitals (`[energy_atp, thermal_integrity, social_contact, epistemic_comfort]`) around homeostatic set-points and scale synaptic plasticity (`lr = base_lr * (1 + 5 * arousal)`).
   - **`[B4]` Purposeful Novel Speech via Forward Model of Inner State (`forward_speech.rs`)**: Speaking is an action selected by its predicted effect on future internal distress (`w_forward_body`), achieving `0.983` accuracy (vs `0.250` chance) without templates or scripted replies.

3. **Quadripartite Human Insular Cortex & Bidirectional Interoceptive Inference** (Quabs, Pantis, Chen, ..., & Parvizi, *Nature* 2026, `10.1038/s41586-026-11175-2` / `10.1038/d41586-026-03334-2`):
   - **`[B5]` `QuadripartiteInsula` (`body.rs`)** implements the four functionally and cytoarchitectonically distinct human insular regions mapped across 87 intracranial electrophysiology patients:
     - **Region I (Dorsal-Posterior Granular Insula)**: Somatotopically organized nociception (pain) and temperature.
     - **Region II (Ventral-Posterior Dysgranular Insula + Area `Id4`)**: Cutaneous somatosensory sensations and oral-cavity/articulatory proprioceptive reafference.
     - **Region III (Mid-Anterior Insula)**: Viscerosensory internal organ states (gut/metabolic ATP, cardiopulmonary fullness) coupled to somatic and cognitive anxiety.
     - **Region IV (Anterior Polar "Silent" Insula)**: Electrically silent to direct localized somatic sensation, yet selectively activated during **salience detection and Action-Mode Network switching** (invariant to Go vs. Withhold motor polarity), with **asymmetric causal connectivity (`PFC` $\to$ `Insula` Region IV $\to$ Region III)** so that higher cognition top-down shapes how bodily deficits are felt.

---

## 2. Workspace Structure (6 Crates)

```text
alpha9/
├── .cargo/config.toml         # Hardware-safe build cap (jobs = 4)
├── Cargo.toml                 # Workspace manifest
└── crates/
    ├── alpha-iface/           # Law 1: Pure numeric SensoryFrame & MotorFrame skin
    ├── alpha-organism/        # Law 2: Zero-string biophysical & cortical brain
    │   └── src/
    │       ├── body.rs            # HomeostaticBody + 8 living VisceralOrganCells + QuadripartiteInsula
    │       ├── cognition.rs       # DeliberativeMind + Living EpigeneticCell ecology (Mitosis, Handoff, Dreaming, 4-Channel Gate)
    │       ├── cortex.rs          # CompetitiveArea + CanonicalArea (Sparse k-WTA Assembly Calculus)
    │       ├── curiosity.rs       # StreamCuriosityTracker + DiscoveryArchive (Generator/Judge)
    │       ├── forward_speech.rs  # FeelingSpeechEngine (forward internal-state speech loop)
    │       ├── genome.rs          # Developmental & biophysical hyperparameters
    │       ├── heatmap.rs         # BiophysicalHeatMap (2D diffusion, ATP limits, settle/follow)
    │       ├── hippocampus.rs     # CrossModalBinding (Hebbian + 1-shot BTSP novelty exclusion)
    │       ├── lib.rs             # Unified continuous Organism::step(SensoryFrame -> MotorFrame)
    │       ├── sensors.rs         # Log-Polar Retina + 40-band ERB Cochlea
    │       ├── util.rs            # Pure numeric linear algebra & Xorshift64* RNG
    │       └── vocal.rs           # VocalTract (8-DOF articulatory 16 kHz PCM synthesizer)
    ├── alpha-world/           # Procedural visual scenes, formant streams & optional opt-in sensory capture
    ├── alpha-probe/           # Law 4: Read-only external telemetry & linear probes
    ├── alpha-exams/           # Law 3: Pre-registered biophysical & anti-cheat verification gates
    └── alpha-bridge/          # Native C-ABI shared library (libalpha9.so) hosting live Alpha-9 + Companion
├── live/
│   └── run_live_habitat.py    # Continuous unforced 24/7 habitat daemon (open 2D game arena + opt-in streams)
└── AGENTS.md                  # Workspace law enforcing unforced external environments & biological purity
```

---

## 3. Verification Gates (`cargo run --release -p alpha-exams`)

```text
==========================================================================
ALPHA 9 BIOPHYSICAL, FEELING-DRIVEN & SENSORY EMERGENCE GATES
==========================================================================
[G0] Anti-Cheat Zero-Strings in alpha-organism (11 files): PASS
[B1] Biophysical Heat-Map (ATP limit=true, 30%-damaged cue settle=1.000, bound word follow=1.000): PASS
[B2] Learning-Progress Curiosity (learnable=+0.2376 vs noisy-TV=-0.0273 vs static=+0.0000): PASS
[B3] Homeostatic Feelings (relief valence=+0.177, harm valence=-0.182, emotional lr gain=2.43x): PASS
[B4] Novel Speech by Predicted Effect on Inner State (need->utterance acc=0.983, chance=0.250): PASS
[B5] Quadripartite Insula (Quabs & Parvizi 2026: post_spec=+0.406, top_down_delta=+0.1060, polar_switch=+0.798): PASS
[S0] Senses (V1 selectivity=0.818, A1 held-out speaker ABX=1.000): PASS
[S1] Saffran Segmentation (word=0.526 vs part-word=0.350): PASS
[S2] Grounding C1 Held-Out Speakers: acc=0.833 (chance=0.333) | C2 Voice Models: acc=0.833 | C3 Novel Combos: acc=0.667
[S2] Controls C4 Lesion: acc=0.344 | C5 Shuffled: acc=0.417 | C6 Streams (col=0.619 vs shp=0.571) | C7 Purity=0.333: PASS
[S3] Babbling -> Vocal Imitation (imitated_cos=0.755 vs null=0.487): PASS
[HW] Organism Heap Footprint: 18.88 MB (Budget <= 32.0 MB): PASS
==========================================================================
```
