# Manual GUI release acceptance

Run the API and ML services, then `python -m services.frontend.main`. Record PASS/FAIL and evidence for every item; none are considered passed until a human observes them.

- [ ] Six-camera grid shows only real feeds; unavailable feeds show OFFLINE / NO SIGNAL.
- [ ] Enlarge one feed and restore it without stale overlays or layout damage.
- [ ] Take one camera offline and restore it; status and video recover.
- [ ] Confirm boxes, IDs, names, and confidence clear when metadata becomes stale.
- [ ] Create a valid temporary camera, edit metadata, disable, enable, and delete it; confirm the tile and runtime state follow each action.
- [ ] Open the empty Persons page with zero enrolled people.
- [ ] Enroll one consenting real person using the documented real-image flow.
- [ ] Edit that person's metadata and confirm the UI refreshes.
- [ ] Delete that person and confirm gallery/UI removal.
- [ ] Observe a real unknown-person card from live metadata.
- [ ] Verify an unknown identity upgrades to the enrolled name after valid real recognition.
- [ ] Restart the frontend while API and ML remain running; confirm cameras, people, events, and overlays rehydrate without duplicates.

Face matching is human validation with consenting real subjects. Do not use synthetic detections or claim biometric accuracy from this checklist.
