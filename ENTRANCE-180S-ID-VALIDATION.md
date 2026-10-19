# Entrance Room — Full 180-Second ID Stability Validation

**Verdict: ADDITIONAL_ID_ISSUES**

The preserved full-duration A/B records were reviewed without rerunning DeepStream. The complete experimental B replay is open in the integrated PySide6 staging UI on `DISPLAY=:0`. B has fewer identity-change and overlap *proxies* than A, but neither run has enough identity ground truth or tracker-lineage provenance to certify identity stability, false-merge rate, or true person counts.

## Evidence and provenance

| Item | A — baseline, ID correction on | B — experimental, ID correction off |
|---|---|---|
| Full event log | `entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/full-events.log`; SHA-256 `cfb1b0d21a84e333b55a0671998a00112a26aaeb814fc754dd93b9063a84b3af` | `entrance-mv3dt-visual-test/id-stability-20261018-B/B-no-id-correction-events.log`; SHA-256 `3046c56da870b14084eff0ee8fc9981986fd0aef657d6894ecf3635a579a61f7` |
| Full three-panel MP4 | `entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/full/entrance-three-panel-full-v2.mp4`; SHA-256 `53119c91128912a550276523a68a3f81aae2e9a3cdc6876d01cb4ee6f12d115f` | `entrance-mv3dt-visual-test/id-stability-20261018-B/entrance-three-panel-id-stability-B.mp4`; SHA-256 `9fdf3a8b3907ee3c3c37ee0ea2605b0c5f5f489bdf8668fe6b93d9830e0f583e` |
| Video properties | H.264, 1920×620, 30 FPS, 5,400 frames, 180 s; full independent decode passed | H.264, 1920×620, 30 FPS, 5,400 frames, 180 s; full independent decode passed |
| Replay stats | `entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/full/three-panel-stats-v2.json` | B replay imported from its exact event log into a private read-only staging SQLite DB under `/tmp/entrance-180s-review.9WdL9e/`; 10,790 messages, 34,539 observations |

Both runs use camera 1/source 0 = CAM-02 and camera 2/source 1 = CAM-05. For each, both video panels contain 5,400 frames over 180 seconds. Event metadata covers frames 5–5,399 (5,395 per camera); frames 0–4 have no corresponding event messages and were not filled or fabricated. Full replay reached frame 5,399. A/B source media are the existing rectified inputs: CAM-02 SHA-256 `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160`, CAM-05 SHA-256 `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813`.

## Full-timeline comparison

| Measured from preserved records | A | B |
|---|---:|---:|
| Raw camera observations | 34,639 | 34,539 |
| Distinct output/global labels (not people) | 18 | 16 |
| Adjacent-frame same-camera box matches with raw-ID change, IoU ≥ 0.4 | 13 | 7 |
| Same, IoU ≥ 0.8 | 11 | 6 |
| Same-camera same-frame distinct-ID overlap candidates, IoU ≥ 0.5 | 6 | 2 |
| Same-camera repeated raw ID in one frame | 0 | 0 |
| Frames with an equal numeric ID in both cameras | 5,393 | 5,394 |
| Projected observations outside the map | 0 | 0 |

These are reproducible continuity/overlap proxies, not ground-truth ID-switch, duplicate-person, false-merge, or unique-person rates. Equal numeric IDs across cameras and map containment do not prove identity or geometric accuracy. The tracker configuration change was limited to `MultiViewAssociator.enableIDCorrection` 1→0; thresholds and late peer reassociation were unchanged. B is experimental and is not promoted.

The records include ID transitions throughout the timeline, so **FULL_REPLAY_STABLE is not supported**. Examples of high-IoU adjacent-frame candidates are:

- A CAM-02: frames 2,233→2,234 (`8→7`, IoU .555), 2,234→2,235 (`7→8`, .995), 2,326→2,327 (`8→7`, .849), 2,778→2,779 (`10→9`, .977), 4,172→4,173 (`15→6`, .975), 4,223→4,224 (`6→16`, .887), 4,501→4,502 (`14→17`, .992).
- A CAM-05 startup frame 5→6: `5→1`, `3→2`, `4→0` with IoUs 1.000, .998, .998; later changes include frames 3,023→3,024 (`12→9`, .964) and 3,024→3,025 (`9→12`, .989).
- B CAM-02 startup frame 5→6: `3→1`, `4→2`, `5→0` with IoUs .993, .991, .989; later changes include frames 2,778→2,779 (`9→8`, .988), 4,172→4,173 (`14→6`, .980), and 4,433→4,434 (`15→13`, .972).
- B CAM-05 has a later candidate at frames 2,843→2,844 (`10→8`, IoU .493).

The existing inspected frame 2,234→2,235 sequence visually supports a continuously visible CAM-02 white-shirt walker while A changes its label and B keeps label 7. This is a narrow continuity example, not proof that B's label is globally correct. The available full event records do not preserve local tracker identity lineage or an explicit cause for each reassignment, so the remaining transitions cannot be reliably classified as fragmentation, cross-camera reassociation, correction, or true identity switches. False merges and the number of distinct real people remain unverified.

## Startup aliases

The three confirmed aliases `1→2`, `3→5`, `4→0` come from diagnostic run03 only. Its source Kafka log SHA-256 is `2c95349ce8bf86758152f55ee16fc2e22c6af645c484daaa93b723654e43afe0`; the alias manifest SHA-256 is `c28ed08b945bbba18356f467bea51e3cc2d4233acdaddcc44bb909d1a251b8f4`. Neither full event log matches that source hash. The staging alias validator rejects applying the manifest to both A and B. Their startup transitions also differ: A changes CAM-05 identities while B changes CAM-02 identities. Accordingly, no alias was applied; the replay displays **Raw ID and Display ID separately with Display ID equal to Raw ID**, and preserves all observations. No same-camera same-frame raw-ID collision was found in either imported full run.

## Review UI and uncertainty

The integrated PySide6 staging review window is open on `DISPLAY=:0`, using the full B three-panel MP4 (CAM-02, CAM-05, Entrance map), playback/seek controls, and a raw/display-ID observation table. The full event replay completed: last frame 5,399; 5,395 frame batches; 34,539 observations. The read-only replay API is on `127.0.0.1:18122`; the existing staging services on ports 18120/18121 were not changed. The A full three-panel video and A/B transition contact sheets are retained alongside the evidence above.

There is no independently verified source capture clock/PTS alignment; frame-number alignment is only the recording/replay timeline. The previously noted synchronization uncertainty (approximately ±150 ms) remains. VGGT pairwise geometry was not validated, and the map scale is unverified, so map paths/positions are for visual review only. No inference, calibration, camera access, production write, or tracker threshold change was performed in this validation.

## Conclusion

B reduces the measured transition and overlap proxies relative to A and has one visually supported continuity improvement, but raw transitions remain and evidence cannot establish that B avoids false merges. Keep B as an experimental review candidate only. To resolve causality, the next useful dataset must preserve per-frame local tracker IDs, raw emitted global IDs, reassociation/adoption events, source timestamps, and independently annotated person identity intervals; evaluate those records before changing tracker settings or promoting aliases.
