# Real face validation procedure

Do not tune the production threshold from synthetic embeddings or one person. Enroll each consenting test person with at least ten good images. Then collect labelled live attempts covering frontal, slight left/right, up/down, glasses where applicable, different expression, moderate distance, partial face, weak lighting, and non-enrolled impostors.

Record one CSV row per independent attempt:

```csv
person_id,expected_match,predicted_person,similarity,face_width,face_height,quality,camera,distance_m,condition,timestamp
```

Keep identities pseudonymous in calibration exports. `expected_match=true` means the expected enrolled identity should match. Impostor attempts use `false`.

Evaluate the existing threshold without changing configuration:

```bash
python scripts/calibrate_face_threshold.py data/validation/face-attempts.csv --threshold 0.55
```

Report genuine and impostor sample counts, TAR/recall, false accepts, and false rejects. Use multiple people and cameras. If either distribution has fewer than 50 independent attempts, treat the threshold as provisional. Any threshold change requires a reviewed dataset result and a regression run.
