# Physical baseline at 32012, lead 48 h

N = 105 start times (2017-09-16 to 2017-12-29, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.0698 | 0.4466 | 0.0171 | 0.5169 | -0.0544 | 0.1504 | 0.2816 | 0.7838 | 0.3595 | 0.4051 | 4.1238 | 2.3619 |
| shape_v13 | 3.3383 | 0.1075 | 0.0216 | 0.9504 | -0.1185 | 0.4822 | 0.8448 | 0.9923 | 0.3995 | 0.4732 | 4.1238 | 7.2476 |
| ridge_AR | 2.8604 | 0.2352 | 0.0213 | 0.9390 | 0.1580 | -0.0111 | 0.1839 | 0.6409 | 0.4251 | 0.4219 | 4.1238 | 1.8952 |
| persistence | 3.7403 | 0.0000 | 0.0251 | 1.2074 | 0.0313 | 0.1955 | 0.6034 | 0.7954 | 0.4743 | 0.5335 | 4.1238 | 4.0667 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 1.2684 | [0.9901, 1.5781] | 0.0000 |
| shape_v13 | Shape_Wasserstein | 0.0045 | [0.0024, 0.0066] | 0.0000 |
| shape_v13 | Tm02_RMSE | 0.4336 | [0.3097, 0.5456] | 0.0000 |
| shape_v13 | peak_fidelity_SS | 0.3318 | [0.2409, 0.4125] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | 0.5632 | [0.4729, 0.6429] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_swell | 0.2085 | [0.1515, 0.2614] | 1.0000 |
| shape_v13 | Peak_Height_RelError_windsea | 0.0400 | [-0.0072, 0.0862] | 0.0410 |
| shape_v13 | Peak_Height_RelError_swell | 0.0682 | [-0.0165, 0.1592] | 0.0510 |
| ridge_AR | Shape_RMSE | 0.7906 | [0.5235, 1.0644] | 0.0000 |
| ridge_AR | Shape_Wasserstein | 0.0042 | [0.0021, 0.0063] | 0.0000 |
| ridge_AR | Tm02_RMSE | 0.4222 | [0.3071, 0.5212] | 0.0000 |
| ridge_AR | peak_fidelity_SS | -0.1615 | [-0.2393, -0.0906] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_windsea | -0.0977 | [-0.1728, -0.0276] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_swell | -0.1429 | [-0.2209, -0.0752] | 0.0000 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0656 | [0.0215, 0.1020] | 0.0020 |
| ridge_AR | Peak_Height_RelError_swell | 0.0168 | [-0.0530, 0.0875] | 0.3040 |
| persistence | Shape_RMSE | 1.6705 | [1.2607, 2.0989] | 0.0000 |
| persistence | Shape_Wasserstein | 0.0080 | [0.0050, 0.0112] | 0.0000 |
| persistence | Tm02_RMSE | 0.6905 | [0.5389, 0.8450] | 0.0000 |
| persistence | peak_fidelity_SS | 0.0451 | [-0.0674, 0.1454] | 0.7660 |
| persistence | Peak_Separation_Recall_windsea | 0.3218 | [0.2098, 0.4265] | 1.0000 |
| persistence | Peak_Separation_Recall_swell | 0.0116 | [-0.0738, 0.0865] | 0.5600 |
| persistence | Peak_Height_RelError_windsea | 0.1148 | [0.0519, 0.1733] | 0.0000 |
| persistence | Peak_Height_RelError_swell | 0.1285 | [0.0378, 0.2237] | 0.0040 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0119, Tm02_RMSE 0.5542, PF 0.1671
- Band Hs: GEFS RMSE 0.2169 m, bias -0.0426 m; persistence RMSE 0.6473 m
- shape_v13 with evaluate()'s shape-space labels: PF 0.4832
- Ridge AR forecasts with negative bins (clipped): 0 of 105
- True significant peaks: 434 on the full grid, 433 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_48h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_48h/linear_baseline_final.pt'}
