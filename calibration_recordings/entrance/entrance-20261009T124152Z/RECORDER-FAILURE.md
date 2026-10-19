# Interrupted Entrance recorder attempt — invalid recording evidence

This session is preserved for provenance only and must not be used as a calibration or human-validation pass.

- Session: `entrance-20261009T124152Z`
- Operator pressed Ctrl+C at approximately 10 seconds. The host controller exited, while the independently owned native recorder continued to its 180-second deadline.
- Native metadata reports 180.04 seconds, but its status is `FAILED` and the MP4s are not playable (`ffprobe`: `moov atom not found`).
- Native log reports `branch EOS/finalization timed out` eight seconds after duration-stop was requested.
- Root cause: the stop path marked each branch's EOS as sent before queuing the EOS callback; the callback then saw the flag and returned without sending EOS. Consequently `qtmux` never finalized the MP4 index.
- CAM-02 input/encoded counts were 3580/3577; CAM-05 counts were 3572/3572. No quality or calibration claim is made from these files.
- Annotated human-evidence recorders exited with code 255 after Ctrl+C; their clips are also incomplete.
- Video SHA256: CAM-02 `1a6b885befeebf8334051060b17c07c313241e09cd38c588c6cd79b14ec7f315`; CAM-05 `e1a0a07847c9311bfeb9a20f50d3738cb27090c01de6ba58cba12763d18f0254`.
- Metadata SHA256: `c084f9cb8e142659e8a50322abac77cd21f78792da848812ec0a213dbd3907f0`.

The original partial files are intentionally retained unchanged. The fix is isolated in the r6 EOS candidate; it has not been deployed to the active r5 owner.
