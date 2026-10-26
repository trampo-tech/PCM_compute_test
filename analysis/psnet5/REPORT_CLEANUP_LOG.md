# Completed report cleanup

Cleanup authorized by the user and completed October 21, 2026. Removed all seven listed candidates, including the historical report bundle and old ranking artifacts. Training logs, checkpoints, prediction exports and underlying evaluation metrics were retained.

| ID | Report or bundle | Reason removed | Historical scope / dependency |
|---|---|---|---|
| S1 | `PCM_LOADER_PORT_PLAN.md` | Proposal for a loader that is now implemented. | Original design rationale; retain implementation and tests. |
| S2 | `LOSS_EXPERIMENT.md` | Cancelled no-smoothing experiment, stopped during epoch 1; includes obsolete progress instructions. | Record of a cancelled hypothesis; no completed result. |
| S3 | `WEIGHTED_CE_EXPERIMENT.md` | Experiment launch notes still say training started, followed by completion notes. Current findings and the weighted visual report cover the successful change. | Exact launch/run/weight notes, also substantially preserved in the visual report. |
| S4 | `WEIGHTED_CE_AREA3_REPORT.md` | Compact result duplicates the weighted visual report and current summary. | Convenient old entry point; The referring launch notes were also removed. Preserve underlying evaluation metrics. |
| S5 | `ERROR_ANALYSIS_1_2.md` | Older epoch-85 voting diagnosis predates the current corrected-evaluation/weighted-loss baseline. | Historical evidence on voting; not a diagnosis of the latest model. Underlying prediction/metric artifacts are separate. |
| S6 | `reports/batchnorm_40_prior/` (report + 3 PNGs, about 1.2 MB) | Historical 60.5% result selected with legacy stored-BatchNorm validation; not the matched baseline for the weighted-loss comparison. | Historical visual comparison. The baseline report now links to retained evaluation metrics. |
| S7 | `results/run_best_val_miou.csv`, `.png`, `.svg` | October 4 ranking predates the successful weighted and latest runs. | Old comparison table/plot; regenerate if a current ranking is needed. |

## Retain as current evidence or implementation references

- `CURRENT_FINDINGS.md`: short current summary.
- `weighted_ce_40_report/`: successful weighted-loss result and figures.
- `batchnorm_40_report/`: matched unweighted baseline. Its title now identifies the October 5 baseline.
- `REPORT.md`: dataset characterization; not an experiment-status report.
- `PADDING_INVESTIGATION.md`: implementation rationale and limitations for padding compaction.
- `REFERENCE_EVALUATION_AUDIT.md`: historical protocol audit; useful reference, with run-specific settings interpreted in their original context.
- `results/classmix_40_training/`, `results/classmix_error_analysis/`, `results/classmix_spatial_analysis/`, `results/height_sensitivity_dev/`, `results/training_exposure_analysis/`: recent evidence supporting the current summary. Some summaries overlap, but they are not stale experiments.

Incoming documentation links were repaired. New summary figures were generated from retained evaluation JSON files; the old ranking artifacts were not reused.
