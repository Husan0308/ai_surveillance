# Entrance Room YOLO26m — Gate 3.1 Missed-Detection Diagnosis

**Verdict: `GATE3_1_PARTIAL`**

The saved same-frame outputs show a substantial proposal-count gap, especially
on CAM-02. One reviewed CAM-02 frame (120) visibly contains three seated people;
PeopleNet has three overlapping proposals and YOLO26m has none. This is strong
evidence of a candidate YOLO miss in that frame, but it is not a human-labeled
benchmark. The other PeopleNet-only proposals may include misses, false
positives, or both. No detection accuracy score or winning-detector decision is
justified yet. I found **no confirmed YOLO parser, class-ID, coordinate-mapping,
or threshold-configuration bug** in the available implementation and run
evidence.

## Scope and preserved evidence

- Experimental worktree: `/home/apsidal/ai_surveillance-yolo-gate1`
- Branch: `experiment/entrance-yolo26m-offline-gate1`
- Starting Gate 3 commit: `b9e135707eee4edad67fc33541935416dd3e3350`
- Analysis source: existing CPU-readable `mux-smoke-events.jsonl` from the
  Gate 3 A/B run; no detector, TensorRT, DeepStream or GPU process was started
  for Gate 3.1.
- Compared camera mapping: source 0 = CAM-02, source 1 = CAM-05.
- Both event logs contain one PGIE record for each of frames 0–299 per camera.
  The same 300 source frames were used for both detectors. This is the first
  10 seconds of each 30 FPS clip, not a full-clip accuracy sample.
- Original rectified videos were decoded with CPU OpenCV only to create private
  review images; both opened as 1920×1080, 30 FPS, 5,400-frame files. Gate 3
  recorded their hashes as CAM-02 `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160`
  and CAM-05 `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813`.
- Gate 3’s `source_pts_ns` values are appsrc sidecar values carried from
  OpenCV `CAP_PROP_POS_MSEC`; they are not independently extracted packet PTS
  and do not prove camera exposure synchronization.

## What the saved detections show

| Camera | YOLO26m boxes | PeopleNet boxes | Frames with zero YOLO / ≥1 PeopleNet | Boxes in PeopleNet not matched to YOLO at IoU ≥0.50* | Median box area, YOLO / PeopleNet (px²) | Confidence p50 / p95, YOLO | Confidence p50 / p95, PeopleNet |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM-02 | 191 | 903 | 171 / 300 | 712 | 72,387 / 64,622 | 0.4321 / 0.8730 | 0.9625 / 0.9753 |
| CAM-05 | 537 | 901 | 1 / 300 | 364 | 29,478 / 20,164 | 0.6875 / 0.8887 | 0.9637 / 0.9813 |

\*Proposal-level one-to-one matching, not ground-truth matching. Every YOLO
box found a PeopleNet proposal at IoU ≥0.50 in the previous Gate 3 comparison
(191/191 CAM-02; 537/537 CAM-05). At IoU ≥0.75, this was 162/191 and 536/537.
The remaining PeopleNet proposals are not automatically extra people; duplicate
or false detections have not been adjudicated.

Within frames 0–299, the longest CAM-02 intervals with zero YOLO boxes but at
least one PeopleNet proposal are frames **36–85**, **90–140**, and **163–203**.
Other shorter intervals are 87–88, 144–155, 159–161, 216, 252, 264–266,
276–278, 280–282 and 288. CAM-05 has one such frame: **216**. The balanced
15-frames-per-camera contact sample (indices 0, 20, …, 280) includes ten
CAM-02 zero-YOLO frames (40, 60, 80, 100, 120, 140, 160, 180, 200, 280) and
no CAM-05 zero-YOLO frame.

Largest per-frame count gaps include CAM-02 frames 278–282, 288, 291 and 293
(up to four PeopleNet proposals versus zero or one YOLO box), and CAM-05 frames
156 and 216 (four or three versus one or zero), followed by 224–227. See the
priority comparison images in the private review package.

At CAM-02 frame 120, the unboxed frame visually shows three seated people;
PeopleNet emits three boxes overlapping those subjects, while YOLO emits zero.
That frame deserves immediate human annotation. It is a qualitative visual
observation, not a verified label. In the retained outputs, YOLO confidence
values have a lower median, and the minimum retained scores sit just above its
0.25 threshold (0.2502); PeopleNet uses a different 0.30 threshold and a
different model/parser. Confidence values across these architectures are not
calibrated against each other. The run saved only post-threshold detections, so
it cannot show whether additional YOLO candidates were just below threshold.

Among PeopleNet boxes left unmatched by YOLO at IoU ≥0.50, median proposal
area/confidence was 64,419 px²/0.9599 on CAM-02 and 19,393 px²/0.7434 on
CAM-05. For PeopleNet boxes matched to YOLO, the corresponding medians were
82,759 px²/0.9700 and 29,148 px²/0.9707. This makes smaller/lower-confidence
CAM-05 proposals useful annotation targets, but does not establish that they
are small people rather than false positives or box-localization differences.

## Implementation review — proven versus unknown

Reviewed the frozen YOLO config, batch-2 render contract, tracked parser source,
Gate 1/2 reports and saved PGIE outputs:

- **Class mapping: no mismatch found.** YOLO outputs class 0 as `person`; the
  tracked parser intentionally retains only class 0. PeopleNet uses class 1
  as `Person` and the config filters non-person classes. The actual Gate 3
  metadata uses those respective IDs.
- **Input geometry/box transform: no gross error evidenced.** The YOLO config
  uses a 640×640 input with `maintain-aspect-ratio=1` and
  `symmetric-padding=1`; the parser consumes per-frame FLOAT `(84,8400)`
  output, reads center-x/center-y/width/height, and emits clamped network-space
  boxes. DeepStream performs configured scaling/padding inversion. The
  returned boxes are in the same 1920×1080 event space and all overlap
  PeopleNet proposals at IoU ≥0.50. This is not independent validation against
  ground truth, so a subtler coordinate error is not mathematically excluded.
- **Parser/output contract: no confirmed bug.** The parser checks the expected
  tensor shape/network dimensions, selects the maximum of 80 class scores,
  emits only class 0, and leaves NMS to DeepStream. The config uses
  `cluster-mode=2`, `nms-iou-threshold=0.45`, `topk=300`, threshold 0.25.
  These settings match the recorded Gate 3 candidate. No parser warnings or
  invalid class/coordinate evidence are reported in the saved successful run.
- **Threshold effect: possible, not proven.** The YOLO threshold is 0.25 and
  the event log has no below-threshold/raw tensor candidates. Lowering it now
  would be an unmeasured configuration change, not a diagnosis.
- **Model/task behavior: leading hypothesis, not proven.** The persistent
  CAM-02 zero-box spans and frame-120 seated-person example are consistent with
  lower YOLO26m sensitivity to this scene/posture/scale than PeopleNet, but
  PeopleNet may also over-detect. Human boxes are required to distinguish
  detector limitations from extra/duplicate PeopleNet proposals. The cameras
  differ materially: YOLO’s median proposal area is larger on CAM-02 and
  CAM-05, but box-size differences alone do not establish correctness.
- **Architecture/runtime comparability remains limited.** The prior Gate 3
  report records distinct input dimensions, thresholds, parser families and
  TensorRT 10.16.1.11 vs 10.16.0.72 images. Those are real model-path
  differences; this investigation did not rerun either arm or attribute the
  count gap to runtime. PeopleNet’s higher proposal confidences are not a
  cross-model quality measure.

No model threshold, preprocessing, parser, engine, NvDCF setting or production
file was changed. No definite detector implementation bug was found.

## Private human-review package

Owner-only package (directory mode 0700; generated files mode 0600):

`/tmp/entrance-yolo26m-gate3-20261019T1248Z/review-gate3-1/`

- `annotations.json`: balanced 30-frame template, with 15 frames per camera
  (0, 20, …, 280); all `annotation_complete` flags are false and every
  `human_verified_annotations` list is empty. It keeps YOLO and PeopleNet
  proposals in separate namespaces.
- `CAM-02/` and `CAM-05/`: for each of those 30 frames, a clean original
  image, YOLO-only overlay, PeopleNet-only overlay, and three-panel comparison.
- `CAM-02/cam-02-30-sample-comparison-grid.jpg` and
  `CAM-05/cam-05-30-sample-comparison-grid.jpg`: compact sample overviews.
- `zero-yolo-candidates/cam-02-all-300frames-zero-yolo-pnt-positive.jpg`:
  overview of all 171 CAM-02 frames with no YOLO boxes and ≥1 PeopleNet
  proposal. Corresponding CAM-05 sheet contains its one candidate frame.
- `priority-largest-gaps/`: full comparison panels for the largest per-frame
  proposal-count gaps, including frame 120 through the balanced sample.
- `README.txt`: annotation guidance and timestamp caveat.

Recommended label record for each visible person is a manually drawn
`bbox_xyxy` plus `posture` (`seated`, `standing`, `unknown`), `occlusion`
(`none`, `partial`, `heavy`, `unknown`) and `size`
(`small`, `not_small`, `unknown`). Mark a frame complete only after reviewing
the whole image, including persons missed by both models. Do not copy model
boxes directly into the verified annotation list.

The CPU-only, one-to-one IoU ≥0.50 evaluator is
`scripts/evaluate_detector_annotations.py`. It first maximizes the number of
qualifying matches and then total IoU, avoiding the greedy assignment failure
mode. It ignores incomplete frames and errors instead of computing metrics if
none are marked complete. Example after labels are independently supplied:

```bash
/home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B \
  scripts/evaluate_detector_annotations.py \
  --annotations /tmp/entrance-yolo26m-gate3-20261019T1248Z/review-gate3-1/annotations.json \
  --detector YOLO26m --iou-threshold 0.50
```

## Tests and changes

- Existing Gate 1/live/Gate 3 focused suites plus new evaluator tests:
  **32 passed, 0 failed** using the repository’s verified full-stack Python
  environment. `py_compile` and `git diff --check` passed.
- New tracked files: `scripts/evaluate_detector_annotations.py`,
  `tests/test_evaluate_detector_annotations.py`, and this report.
- No production detector code or configuration changed. Review images and
  annotation template remain under `/tmp`; none are committed.
- GPU inference: **NOT RUN**. Live RTSP: **NOT RUN**. Production: untouched.

## What is required before choosing a detector

1. A reviewer labels all visible people on the 30 balanced review frames,
   prioritizing CAM-02 frame 120, the zero-YOLO spans and the large-gap frames.
   Include empty frames and cases where either model proposes an extra box.
2. Run the evaluator independently for both models at IoU ≥0.50 and inspect
   matched/unmatched boxes; do not conflate proposal overlap with correctness.
3. If the decision will cover the full Entrance Room clip, add a balanced,
   human-annotated sample across later portions of the 180-second videos and
   relevant seated/standing, occluded and small-person cases. Thirty frames
   from the first ten seconds alone may not represent the complete footage.
4. Assess tracker identity quality only with separate verified identity
   annotations; this box-review tool does not measure ID switches or IDF1.

Until those annotations exist, the cause of the observed count difference is
**not conclusively attributable** to YOLO misses versus PeopleNet false or
duplicate proposals. No detector is recommended as the winner.
