# Optional label resmoothing

Use this maintenance helper when existing prediction CSVs need their smoothed
labels regenerated without rerunning the neural model. Ordinary inference already
writes smoothed labels.

```text
python -m autobehaver.resmooth --input-dir <predictions> --output-dir <new-folder>
```

The input contains `*_inference.csv` files. Use a separate, empty output folder.
Keep the original predictions. The helper preserves raw predictions, probabilities and frame identifiers,
and changes only the smoothed label columns. By default it uses the two-stage,
five-frame smoothing rule and protects Jump. Before smoothing, internal missing
runs of up to four frames are filled only when the known labels on both sides
match. Apply the gap rule again after smoothing, because smoothing can make the
flanks match. Longer gaps, differing final flanks and recording edges remain missing.
MMGen applies this same gap rule when reading existing smoothed predictions, so
the per-fly summary update does not require another neural-model inference run.

Source-frame alignment is automatic when a CSV has `Source Frame Index` or its
processing metadata records the legacy offset. For unidentified legacy files only,
add `--source-frame-offset N` with the verified number of omitted leading frames
(`0` for untrimmed inputs). Do not assume every file needs an offset of five, or
manually shift its frame identifiers. Modern source indices must be used unchanged.

Each output has a provenance receipt containing source/output hashes, smoothing
settings and frame alignment. Keep these receipts with the CSVs for MMGen and
InfMovie. `resmoothing.complete.json` is written only after every selected file
passes verification. Use fresh MMGen and FeatureImp output directories when
recalculating analyses from the resmoothed labels.
