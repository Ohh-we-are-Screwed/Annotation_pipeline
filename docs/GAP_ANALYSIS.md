# Literature Scan & Gap Analysis — What to Look Out For, What to Aim At
### DhakaScenes companion to `comprehensive.md`

**Scan date:** 2026-08-07 (web-verified on this date; re-run the §11 monitoring queries before every milestone — this document decays).
**Purpose:** for every claimed contribution, establish (a) what the literature already contains, (b) the precise gap that remains, (c) the measurable aim that fills it, and (d) the threats — scooping risks, methodological pitfalls, and items that must be re-verified.
**Rule:** a gap is only claimable in the paper if the "What exists" row below is cited *in the paper* and the differentiation survives it. Reviewers check whether you know the nearest prior work better than they do.

---

## 0. Summary — gap status board

| # | Gap | Status | Our aim (claim it feeds) | Biggest threat |
|---|---|---|---|---|
| G1 | 3D boxes/tracks + LiDAR in Bangladesh / Dhaka-class density | **OPEN** — all Bangladesh datasets are 2D camera-only | First 3D multimodal benchmark for Bangladeshi traffic with indigenous taxonomy (C1) | IDD-ecosystem extension or a PHE3D-style regional release landing first |
| G2 | Density-stratified evaluation as a shipped protocol | **PARTIAL** — per-paper crowdedness breakdowns exist; no dataset ships it as a first-class benchmark axis | ρ-binned reporting + Degradation Δ for every headline metric (C2) | Effect may be flat (our own kill-gate G5); someone formalizes difficulty-stratified eval first |
| G3 | Open-vocabulary recognition of indigenous 3-wheeler subtypes | **OPEN** — foundation-model labeling validated only on Western taxonomies | S1 separability study + working per-class pipeline thresholds (C1, C5) | SAM 3-era models may already separate them (good for us) or fail badly (plan B exists) |
| G4 | Night + tropical-monsoon coverage with LiDAR in dense traffic | **OPEN** — adverse-weather LiDAR datasets are snow/temperate, structured, sparse | ≥ 25 % night share; monsoon sessions; illumination-stratified metrics (C2) | Weather-robustness fatigue in reviews — must pair coverage with *measured* degradation numbers |
| G5 | cm-class ground truth **plus** perception labels in a GNSS-hostile dense city | **OPEN** — cm-GT datasets have no perception labels; perception datasets have m-class GT | PPK+RTS trajectory with published per-frame σ, feeding odometry *and* label quality (C3, C5) | PPK fixed-rate in Dhaka canyons unproven — our gate G3 exists precisely for this |
| G6 | Auto-labels used to *bootstrap a released benchmark* with audited tiers | **PARTIAL** — auto-labeling is a hot method area; no benchmark ships tiered, audited machine labels | Label-tier policy + published pseudo-label audit (C5) | Circularity critique; SAM 3-based auto-label papers accelerating since 2025-11 |
| G7 | Bidirectional "value of data" cross-dataset protocol with confound controls | **PARTIAL** — domain-adaptation literature is mature; value-of-data framing with label-provenance control is not | 4-cell protocol + sensor & provenance controls (C4) | Sensor-geometry confound; a controlled negative must remain publishable |
| G8 | Detection benchmarking on non-repetitive solid-state LiDAR | **PARTIAL** — Livox is common in SLAM datasets, rare in detection benchmarks | Documented baseline adaptations + input-variant reporting (C5) | Baselines underperform → "broken dataset" optics if adaptation is not documented |
| G9 | Crowd-honest 3D evaluation semantics (group/ignore boxes) | **OPEN in 3D** — established in 2D (CrowdHuman) and as no-label zones (Waymo) | Devkit-coded 3D group-box semantics (C5) | Metric gaming via group boxes; semantics must be airtight in code |
| G10 | Trustworthy yaw GT in stop-and-go + heading-aware metrics for symmetric vehicles | **PARTIAL** — APH exists (Waymo); dual-antenna practice exists; the combination applied to 3-wheelers does not | Moving-base heading + published yaw σ + APH as first-class metric (C1, C3) | Overclaiming heading accuracy; publish measured, not datasheet, values |

---

## 1. G1 — Geographic & modality gap: no 3D perception benchmark for Bangladesh

**What exists (verified this scan):**
- **Bangladesh, all 2D camera-only:** RSUD20K (Zunair et al., ~20.3K driver-view images, ~130K 2D boxes, 13 classes, narrow streets and crowded scenes; also evaluates large vision models as zero-shot annotators — cite that angle, it anticipates our pipeline story). BadODD (Baig et al., 2024: smartphone-camera 2D detection across 9 districts incl. Dhaka; 13 classes incl. auto-rickshaw and generic "three-wheeler"; day+night; ~9.8K images). DhakaAI (~3.9K images, 21 classes; known label-quality issues per RSUD20K's related-work). Poribohon-BD (~9K images, 15 vehicle classes, includes non-driving classes).
- **India:** IDD (camera, 2D/segmentation, unstructured roads). **IDD-3D (WACV 2023)** — the nearest prior work: Hyderabad, multi-camera + LiDAR, ~12K annotated LiDAR frames (paper also mentions 15.5K annotated frames — resolve this discrepancy from the primary PDF before citing a number), ~223K 3D boxes, 17 categories (10 primary) including auto-rickshaw, hand-carts, animals; standard 3D detection + tracking benchmarks. DriveIndia (Kumar et al., 2025) — RGB 2D only `[VERIFY scope]`.
- **Region-adjacent:** PHE3D — a Vietnamese-streets LiDAR 3D detection/tracking dataset surfaced in citation trails `[VERIFY venue, sensors, availability — appears in secondary sources only]`.

**The gap, stated precisely:** there is **no dataset anywhere with 3D boxes + tracks + LiDAR collected in Bangladesh**, and no dataset in any country that first-classes the *full* indigenous set {CNG auto-rickshaw, battery-rickshaw, cycle-rickshaw, tempo, human-hauler, thela} as separate 3D categories with attributes. IDD-3D covers auto-rickshaw-class objects but with a coarser taxonomy, metre-class pose, and no density/illumination protocol.

**What we aim at:** the C1 package — 16-class taxonomy with the 6 indigenous classes as separate categories, ≥ 1k instances per headline indigenous class, per-class AP + confusion matrix, published mapping tables to nuScenes/Waymo/KITTI/IDD-3D. The paper's positioning table must place IDD-3D in the adjacent column and differentiate on: PPK pose, night share, density protocol, taxonomy depth, attribute annotations — *never* on "unstructured traffic" alone.

**Look out for:**
- The IDD group (IIIT-Hyderabad CVIT) ships follow-ups regularly (IDD-AW, IDD-X, DriveIndia); an "IDD-3D v2" with denser taxonomy or better pose would compress G1 substantially. Monitor their pages and WACV/ICCV/CVPR dataset tracks.
- Regional efforts of the PHE3D type (Vietnam, Indonesia, Pakistan, Nigeria) — the "dense developing-world 3D dataset" slot is visibly filling.
- Bangladesh 2D groups (RSUD20K's Concordia team, BadODD's community) upgrading to LiDAR — cheapest path for them is exactly our rig class.
- Pitfall: claiming "first South Asian 3D dataset" (false — IDD-3D). Claim "first for Bangladesh" and "first with this taxonomy depth + pose class", both defensible.

---

## 2. G2 — Density-stratified evaluation as a benchmark protocol

**What exists:**
- Per-paper crowdedness breakdowns: FSDv2 (Fan et al.) defines crowded objects via nearest-neighbor distance < 2 m and reports a crowdedness-conditioned table; FSF reports recall by points-per-object strata. These are *method-paper ablations*, defined ad hoc, not dataset protocols.
- Dataset-level difficulty axes that exist: KITTI easy/moderate/hard (occlusion/truncation/height), Waymo LEVEL_1/2 (points-in-box), nuScenes visibility bins. None is a *scene-density* axis.
- H3D (Honda, 2019) markets itself as full-surround 3D detection/tracking in crowded urban scenes (~1.07 M boxes) — crowded as a *setting*, with no density-binned reporting protocol.
- 2D crowd literature: CrowdHuman established crowd-specific evaluation practice; pedestrian-detection work reports occlusion-stratified miss rates.

**The gap:** no driving benchmark defines a measured per-frame density statistic, bins its entire evaluation by it, and reports degradation deltas as the release's headline protocol. "Averaged-away difficulty" is a known criticism without an instituted fix.

**What we aim at:** the C2 package — ρ defined over the eval region with area normalization, quantile bins with published edges, every headline metric per bin, Degradation Δ, and the pilot ablation (gate G5) run **before** the paper is written. Aim to make "report per-density" a protocol others adopt — that is what makes a dataset paper cited beyond its data.

**Look out for:**
- Any 2025–26 paper proposing standardized difficulty- or density-conditioned evaluation for 3D detection (watch CVPR/NeurIPS D&B track). If one appears, adopt their formalism and be the first *dataset* to instantiate it — the contribution shifts from "protocol" to "instrument", still strong.
- The self-threat: on our own data the curve may be flat (detectors may saturate on near-field dense objects because density correlates with low speed and short range). This is why ρ bins must be crossed with range bins in the analysis, and why G5 runs first.
- Pitfall: density confounded with range and with class mix. Report ρ-binned metrics per class and control range; otherwise a reviewer will attribute the degradation curve to composition shift, not density.

---

## 3. G3 — Do open-vocabulary models even see our classes?

**What exists:**
- Foundation-model labeling of LiDAR is validated on Western taxonomies: SAL (ECCV 2024) lifts SAM masks + CLIP tokens to LiDAR and reaches 91 % class-agnostic / ~44–54 % zero-shot LPS relative to supervised SOTA; SAL-4D (CVPR 2025) extends to 4D with VOS pseudo-labeling; SAM3D does BEV-image zero-shot detection on Waymo; shelf-supervised zero-shot 3D boxes from RGB+LiDAR (Khurana et al. line) target common classes.
- RSUD20K already probed large vision models as zero-shot annotators on Bangladeshi imagery (2D) — read their findings closely; they are the closest existing evidence on whether VLM-era detectors recognize local vehicle types.
- SAM 3 (Meta, released 2025-11, open weights + SA-Co benchmark) accepts short noun-phrase concept prompts and segments/tracks *all* instances — precisely the capability our taxonomy prompting needs; crowded-scene identity switches remain a stated limitation.

**The gap:** nobody has measured whether current open-vocabulary detectors/segmenters separate {battery-rickshaw vs. cycle-rickshaw}, {tempo vs. human-hauler}, {CNG vs. generic auto-rickshaw} — the pairs our benchmark lives on. The S1 study is therefore not just a gate; it is a reportable result about foundation-model coverage of Global-South categories.

**What we aim at:** S1 on the S0 seed set with pass criteria (recall ≥ 0.7, pairwise confusion ≤ 30 % per indigenous class), reported in the paper either way; per-class prompt engineering documented (synonyms, exemplar prompts via SAM 3 image exemplars); fallback merged-pair pre-labelling if S1 fails.

**Look out for:**
- SA-Co's 200K+ concept vocabulary may include rickshaw subtypes — check the released benchmark vocabulary before designing prompts `[VERIFY]`.
- Exemplar prompting (SAM 3) likely outperforms text for visually-defined local subtypes; budget S1 to test text-only vs. text+exemplar arms — that comparison alone is a nice supplementary table.
- Pitfall: tuning prompts on S0 and evaluating on S0. Split S0 for prompt-tuning vs. S1 measurement.

---

## 4. G4 — Night + monsoon LiDAR data in dense traffic

**What exists:**
- Adverse-weather LiDAR datasets are temperate/snow-centric and structured-traffic: Boreas (Toronto, 128-beam, repeated route over a year, rain/snow/night, cm-accurate post-processed poses; leaderboards for odometry, localization, detection); K-Radar (Korea, 35K frames, 4D radar + LiDAR + RTK, fog/rain/snow, day/night distribution published); Ithaca365 (repeated route, snow/rain/night, amodal 2D/3D annotations); WADS (point-wise snow labels); CADC (snow); SeeingThroughFog/DENSE (controlled fog/rain, day/night).
- Camera-domain night/adverse: ACDC, Dark Zurich, nuScenes-Night subsets — 2D or camera-3D, and night is a minority share in the majors (verify exact nuScenes/Waymo night fractions from their papers before the positioning table).

**The gap:** no dataset combines LiDAR + tropical monsoon rain + substantial night share + extreme-density unstructured traffic. Monsoon is not Toronto rain: warm heavy precipitation, flooded road surfaces, umbrella-carrying crowds, rickshaws with rain covers — appearance shifts no existing set contains.

**What we aim at:** the §4.4 coverage contract (night+dark ≥ 25 %, planned monsoon sessions), luminance-measured illumination bins, Night Degradation Δ per task, day/night depth evaluation split — coverage always paired with a measured consequence, never presented as scenery.

**Look out for:**
- Reviewer fatigue with "adverse weather" claims: the differentiator is the *conjunction* (monsoon × density × LiDAR), argue it as such with the table above.
- LiDAR-in-rain physics: heavy rain suppresses returns and adds noise; report per-condition points-per-box statistics so weather effects on the *sensor* are separable from effects on the *models* — Boreas/WADS papers give the vocabulary for this.
- Rig survival (risk register): monsoon capability is a claim about engineering first.

---

## 5. G5 — Centimetre-class ground truth *with* perception labels in a GNSS-hostile megacity

**What exists:**
- cm-class GT, **no perception labels**: UrbanNav (Hong Kong urban canyons; multi-frequency GNSS, IMU, multiple LiDARs, cameras; cm-level ground truth post-processed from a NovAtel SPAN RTK/INS system; explicitly a *positioning* benchmark) and UrbanLoco (HK + San Francisco, same family). These prove cm-class GT is achievable in Asian canyons with survey-grade equipment — and define the honest-reporting bar.
- cm-class GT + (some) labels, **structured sparse traffic**: Boreas — post-processed cm-accurate poses plus a detection benchmark, suburban Toronto, with a correction-service subscription in the loop.
- Perception majors: nuScenes/Waymo/IDD-3D ship metre-class GNSS/INS pose — sufficient for boxes, unusable as odometry truth.

**The gap:** no dataset offers survey-grade post-processed pose **and** a full 3D perception benchmark **in a dense developing-world megacity**, and almost none publishes per-frame pose uncertainty at all. That last point is our cheapest credibility win: UrbanNav reports conditions; we ship σ per frame.

**What we aim at:** C3 — PPK+RTS trajectory from a ≤ 20 km base, dual-antenna heading, per-frame σ_pos/σ_yaw + PPK-fix flags in `frame_quality`, odometry/localization benchmark gated by G3 (≥ 40 % fixed-ambiguity or the task demotes). Explicitly position against UrbanNav: they benchmark *positioning algorithms*; we use positioning quality to *certify a perception benchmark* — complementary, cite generously.

**Look out for:**
- Our low-cost F9P pair vs. their SPAN-class hardware: expect reviewers who know UrbanNav to ask whether F9P PPK can reach cm-class in Dhaka. The answer is measured, not asserted — the pilot's PPK-fixed percentage and σ distributions are the response. If they are poor, C3 degrades gracefully by design.
- Dhaka canyons are lower-rise than Hong Kong's but with heavy tree cover and overpasses — different multipath profile; do not import Hong Kong numbers as expectations in either direction.
- Any 2025–26 release of a labeled perception dataset with PPP/PPK-grade poses (watch Boreas-style groups and the ION/NAVIGATION venue) — would compress the conjunction claim.

---

## 6. G6 — Auto-labels as *release-grade* ground truth with audited tiers

**What exists (fast-moving — the hottest adjacent area):**
- **VESPA** — now verified real and published: arXiv 2507.20397 (2025-07) and accepted at **CVPR 2026** ("VESPA: Open-World Auto-Labeling for 3D Object Detection in Autonomous Driving"). It is a *VLM-based* multimodal auto-labeling pipeline (ground removal → zero-shot segmentation → 2D-3D distillation → multimodal refinement) reporting ~52.95 % AP object discovery and ~46.5 % multiclass detection on nuScenes, with additional TruckScenes results. **Correction to our internal drafts:** quote only the camera-ready numbers — some figures circulating in our annotation doc (e.g., "48.12 NDS") do not match the arXiv abstract; re-derive every number from the final PDF.
- **AIDE** (CVPR 2024) — automatic data engine with taxonomy expansion; **DetZero** (ICCV 2023) — offboard long-sequence track refinement (our tracking design reference); Waymo's offboard "auto labeling" line (Qi et al.) preceding it.
- **SAL** (ECCV 2024) / **SAL-4D** (CVPR 2025) — pseudo-label engines distilling 2D foundation models into LiDAR panoptic models; **SAM3D**; shelf-supervised zero-shot 3D boxes; **UNION**, MODEST/OYSTER (motion-based unsupervised discovery).
- **OpenAD** (NeurIPS 2025) — open-world 3D detection benchmark built with an MLLM-assisted corner-case annotation pipeline across 5 existing datasets: proof that reviewers now accept machine-assisted annotation *when its role is explicit*.
- **SAM 3 / SAM 3.1** (Meta, 2025-11 onward): open-weights concept-promptable detect+segment+track, with an official 3.1 iteration advertising faster multi-object video tracking. Its own data engine used LLM annotators+verifiers — a design precedent for human-in-the-loop auditing at scale.

**The gap:** all of the above are *method papers evaluated against existing human GT* (or benchmarks of corner cases). No driving benchmark has shipped its own labels as an explicit tier system — machine-accepted train labels with published audited error rates, human-verified val/test — as a first-class, documented property of the release. The norm is silence about label provenance.

**What we aim at:** C5's label story: tier policy (§7.4 of `comprehensive.md`), stratified pseudo-label audit table, IAA, and the framing "we tell you exactly what the machine labeled and how well." Additionally, our pipeline delta vs. VESPA is now sharper and honest: **VESPA requires VLM calls; we constrain the hot path to open-weights vision models (SAM 3-class + DINOv2 + geometric refinement) and add the S0-bootstrapped indigenous-class priors** — an ablation table (VESPA-style stages, our data) belongs in the supplementary.

**Look out for:**
- **Pipeline-stack revision (action item):** the earlier internal audit flagged "SAM 3.1" as unverifiable; this scan **confirms SAM 3 and a 3.1 iteration exist with open weights**. `comprehensive.md` §7.1 should be revised: prefer SAM 3/3.1 as the unified proposal+mask+track engine (text + exemplar prompts), with Grounding DINO/MM-GDINO retained as a comparison arm; verify the SAM 3 license permits dataset production `[VERIFY license terms]`. DINO-X remains API-first (check whether an open Edge checkpoint suffices) — the "fully offline" claim stands only on the open-weights stack.
- Circularity critique to pre-empt in writing: benchmark labels seeded by foundation models partially encode those models' biases (our F3/control-6 answer; also report S0-human-only results beside pipeline-assisted ones).
- Velocity of the area: expect several "SAM 3 for 3D auto-labeling" papers by CVPR/ICCV 2026–27 cycles. Our defensible position is the *dataset + audit protocol*, not the pipeline novelty — write the paper accordingly.

---

## 7. G7 — Cross-dataset "value of data" protocol

**What exists:**
- The canonical cross-dataset 3D study: Wang et al., *Train in Germany, Test in the USA* (CVPR 2020) — established that 3D detectors transfer poorly across datasets and that object-size statistics drive much of the gap `[VERIFY exact findings before citing]`. Domain-adaptation methods followed: ST3D/ST3D++, MS3D++ (multi-source ensembles), plus the domain-generalization axis formalized in OpenAD.
- Standard practice in dataset papers: a one-direction "models trained elsewhere do worse here" table — widely (and correctly) criticized as conflating difficulty with label noise.

**The gap:** dataset papers rarely run the *bidirectional* 4-cell matrix with matched budgets, and none we found adds a **label-provenance control** separating what the data teaches from what the auto-label pipeline injected. The sensor-representation control (common BEV grid) is likewise usually absent.

**What we aim at:** C4 exactly as specified (§8.3.5): 4 cells × ≥ 3 seeds, matched size curves, common BEV representation, class-mapping table, provenance control, and the honest decision rule — a controlled negative is reported as such.

**Look out for:**
- The compute reality: this experiment dominates GPU budget (§8.5); if it slips, the paper survives on C1+C2+C5 — do not let C4 hold the submission hostage.
- Non-repetitive-LiDAR-to-32-beam conversion is where reviewers will probe; publish the grid spec and per-representation sanity checks (in-domain performance before/after conversion).

---

## 8. G8 — Benchmarking on non-repetitive solid-state LiDAR

**What exists:** Livox-pattern sensors are pervasive in SLAM/odometry datasets and in the LIO literature (FAST-LIO2 lineage), and appear in niche perception sets (e.g., dusty off-road LiDARDustX; various Livox-provided datasets). Mainstream detection benchmarks (KITTI/nuScenes/Waymo/AV2/ONCE/ZOD) are spinning-LiDAR; most detector implementations assume ring/range-image structure implicitly (input encodings, ground-truth sampling stats, anchor tuning).

**The gap:** no widely-used 3D *detection/tracking* benchmark on a non-repetitive rosette sensor with documented, fair baseline adaptations — so the community cannot currently answer "how well do standard detectors do on the LiDAR class that low-cost deployments actually use?"

**What we aim at:** §8.5's adaptation policy as a contribution in itself: per-baseline adaptation notes, single-vs-accumulated-sweep input variants, sensor-specific LEVEL thresholds, and the positioning sentence that the sensor choice is deliberate deployability realism.

**Look out for:**
- Optics risk: unadapted baselines scoring low will be read as "broken dataset" unless every number sits next to its adaptation note. Never publish an un-annotated low baseline.
- Watch for 2026 detector releases with native support for solid-state patterns (sparse-voxel/point-transformer lines are pattern-agnostic — prefer them in the baseline set).

---

## 9. G9 — Crowd-honest evaluation semantics in 3D

**What exists:** CrowdHuman institutionalized 2D crowd evaluation (visible/full-body boxes, ignore regions, crowd-aware metrics); Waymo ships No-Label Zones; nuScenes has per-box visibility but no group construct; 3D benchmarks otherwise assume every instance is individually resolvable.

**The gap:** no 3D driving benchmark defines group/ignore boxes with exact matcher semantics for scenes where instances are genuinely unresolvable — precisely the frames a gridlock dataset is about.

**What we aim at:** §6.3 + devkit-coded semantics (predictions matched to a group box are neither FP nor FN; `n_min` feeds ρ). Publish the fraction of group-boxed area per density bin — it is itself a finding about annotatability limits.

**Look out for:** gaming (a model that dumps predictions into group zones loses nothing) — cap the ignore area statistic per frame in reporting, and consider a secondary count-based metric (predicted-count vs. `n_min`) as a supplementary probe.

---

## 10. G10 — Heading truth in stop-and-go + heading-aware metrics for symmetric vehicles

**What exists:** Waymo's APH is standard for heading-aware detection; dual-antenna/moving-base heading is standard practice in navigation (u-blox moving-base, VN-300, SPAN-class systems in UrbanNav/Boreas); yaw-drift-in-stationarity is a known single-antenna INS failure mode in the navigation literature.

**The gap:** perception datasets do not publish yaw ground-truth uncertainty at all, and no benchmark centers APH on near-symmetric small vehicles where 180° flips are the dominant orientation error — our three-wheelers are the ideal instrument for it.

**What we aim at:** moving-base heading hardware (comprehensive.md F10), published yaw σ split by stationary flag, APH and mAOE as first-class per-class results for the indigenous classes, and a small analysis of flip-rate vs. class symmetry (quotable finding).

**Look out for:** publishing datasheet heading specs instead of measured values; and note SAM 3-era trackers reduce ID switches but not box orientation — orientation quality still comes from our geometric/track refinement, so ablate it.

---

## 11. Monitoring protocol — how this document stays alive

**Cadence:** re-run before: taxonomy freeze, scaled collection start, experiment freeze, submission, camera-ready.

**arXiv / Scholar alert queries (verbatim):**
1. `Bangladesh driving dataset` · 2. `Dhaka traffic detection` · 3. `unstructured traffic 3D detection` · 4. `rickshaw detection dataset` · 5. `IDD-3D` (citing-papers alert) · 6. `auto-rickshaw 3D` · 7. `density stratified evaluation detection` · 8. `LiDAR auto-labeling foundation model` · 9. `SAM 3 3D detection` · 10. `Livox object detection benchmark` · 11. `PPK ground truth driving dataset` · 12. `monsoon autonomous driving`

**Venue sweeps:** CVPR/ICCV/ECCV + WACV (datasets in the Global-South niche keep landing at WACV), NeurIPS Datasets & Benchmarks, ICRA/IROS, RA-L; ION/NAVIGATION for pose-GT practice.

**Named watchlist (nearest neighbors, what a scoop looks like):**
| Who | Signal that compresses our gap |
|---|---|
| IIIT-Hyderabad CVIT (IDD family) | IDD-3D v2 / denser taxonomy / RTK-grade pose |
| RSUD20K team (Concordia) & BadODD community | any LiDAR upgrade announcement for Bangladesh |
| VESPA authors (TUM-affiliated) | applying their engine to build a *released dataset* |
| SAL / SAL-4D line (NVIDIA-adjacent) | zero-shot labeling packaged as dataset production |
| Boreas group (UToronto) | dense-urban labeled release with post-processed GT |
| PHE3D-type regional efforts | any Global-South LiDAR release with 3D boxes |

**Verification backlog carried from this scan:** IDD-3D 12K-vs-15.5K frame count; DriveIndia scope; PHE3D primary source; SAM 3/3.1 license terms for dataset production; DINO-X Edge open availability; VESPA camera-ready numbers; nuScenes/Waymo night fractions; Wang et al. 2020 exact findings; SA-Co vocabulary coverage of rickshaw subtypes.

---

## 12. Primary references surfaced in this scan (for the paper's related-work file)

IDD-3D — Dokania et al., WACV 2023, arXiv:2210.12878 · RSUD20K — Zunair et al., arXiv:2401.07322 · BadODD — Baig et al., arXiv:2401.10659 · IDD — Varma et al., WACV 2019 · H3D — Patil et al., ICRA 2019, arXiv:1903.01568 · FSDv2 crowdedness breakdown — arXiv:2308.03755 · CrowdHuman — Shao et al., 2018 · Boreas — Burnett et al., IJRR 2023, arXiv:2203.10168 · K-Radar — Paek et al., NeurIPS 2022 D&B, arXiv:2206.08171 · Ithaca365 — Diaz-Ruiz et al., CVPR 2022 · UrbanNav — Hsu et al., NAVIGATION 70(4), 2023 · UrbanLoco — Wen et al., ICRA 2020 · VESPA — Tempfli/Rivera et al., CVPR 2026, arXiv:2507.20397 · AIDE — CVPR 2024 · DetZero — ICCV 2023 · SAL — ECCV 2024, arXiv:2403.13129 · SAL-4D — CVPR 2025 · SAM3D — arXiv:2306.02245 · OpenAD — NeurIPS 2025, arXiv:2411.17761 · SAM 3 — Meta, 2025-11 (paper + facebookresearch/sam3; SA-Co benchmark; 3.1 blog update) · SAM 2 — ICLR 2025-era, Meta 2024 · Grounding DINO — ECCV 2024 · DINOv2 — Oquab et al. · Train-in-Germany-Test-in-USA — Wang et al., CVPR 2020 · ST3D — CVPR 2021 · MS3D++ — arXiv 2023 · ZOD — ICCV 2023 · nuScenes — CVPR 2020 · Waymo Open — CVPR 2020 · KITTI — CVPR 2012 · Argoverse 2 — NeurIPS 2021 D&B · ONCE — NeurIPS 2021 · GOOSE — ICRA-adjacent off-road (contrast only) · LiDARDustX — arXiv:2505.21914 (contrast only).

**Bottom line:** every load-bearing claim in `comprehensive.md` still has an open lane as of 2026-08-07 — but G6's lane (foundation-model auto-labeling) is crowding fastest, and G1's lane is the one where a single regional release would hurt most. The two moves that protect the paper regardless of scooping are the ones only we can do: collect the Dhaka data with measured quality, and ship the density-stratified protocol with the audit tables.
