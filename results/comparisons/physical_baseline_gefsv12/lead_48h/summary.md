# Physical baseline at 32012, lead 48 h

N = 105 start times (2017-09-16 to 2017-12-29, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.0698 | 0.4466 | 0.0171 | 0.5169 | -0.0544 | 0.1395 | 0.2736 | 0.7738 | 0.3882 | 0.3802 | 2.6095 | 1.9619 |
| shape_v13 | 3.3383 | 0.1075 | 0.0216 | 0.9504 | -0.1185 | -0.2352 | 0.0189 | 0.4583 | 0.4251 | 0.5226 | 2.6095 | 1.6476 |
| ridge_AR | 2.8604 | 0.2352 | 0.0213 | 0.9390 | 0.1580 | -0.1542 | 0.0566 | 0.5179 | 0.4260 | 0.4568 | 2.6095 | 1.2952 |
| persistence | 3.7403 | 0.0000 | 0.0251 | 1.2074 | 0.0313 | -0.0554 | 0.3113 | 0.6250 | 0.5020 | 0.5452 | 2.6095 | 2.6095 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 1.2684 | [0.9901, 1.5781] | 0.0000 |
| shape_v13 | Shape_Wasserstein | 0.0045 | [0.0024, 0.0066] | 0.0000 |
| shape_v13 | Tm02_RMSE | 0.4336 | [0.3097, 0.5456] | 0.0000 |
| shape_v13 | peak_fidelity_SS | -0.3748 | [-0.4647, -0.2951] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | -0.2547 | [-0.3300, -0.1836] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_swell | -0.3155 | [-0.4044, -0.2275] | 0.0000 |
| shape_v13 | Peak_Height_RelError_windsea | 0.0369 | [-0.0263, 0.0951] | 0.1340 |
| shape_v13 | Peak_Height_RelError_swell | 0.1424 | [0.0600, 0.2250] | 0.0000 |
| ridge_AR | Shape_RMSE | 0.7906 | [0.5235, 1.0644] | 0.0000 |
| ridge_AR | Shape_Wasserstein | 0.0042 | [0.0021, 0.0063] | 0.0000 |
| ridge_AR | Tm02_RMSE | 0.4222 | [0.3071, 0.5212] | 0.0000 |
| ridge_AR | peak_fidelity_SS | -0.2937 | [-0.3674, -0.2209] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_windsea | -0.2170 | [-0.2844, -0.1429] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_swell | -0.2560 | [-0.3519, -0.1697] | 0.0000 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0378 | [-0.0121, 0.0837] | 0.0810 |
| ridge_AR | Peak_Height_RelError_swell | 0.0766 | [0.0114, 0.1427] | 0.0120 |
| persistence | Shape_RMSE | 1.6705 | [1.2607, 2.0989] | 0.0000 |
| persistence | Shape_Wasserstein | 0.0080 | [0.0050, 0.0112] | 0.0000 |
| persistence | Tm02_RMSE | 0.6905 | [0.5389, 0.8450] | 0.0000 |
| persistence | peak_fidelity_SS | -0.1949 | [-0.2868, -0.1054] | 0.0000 |
| persistence | Peak_Separation_Recall_windsea | 0.0377 | [-0.0606, 0.1285] | 0.7420 |
| persistence | Peak_Separation_Recall_swell | -0.1488 | [-0.2452, -0.0552] | 0.0010 |
| persistence | Peak_Height_RelError_windsea | 0.1138 | [0.0499, 0.1747] | 0.0000 |
| persistence | Peak_Height_RelError_swell | 0.1650 | [0.0889, 0.2549] | 0.0000 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0119, Tm02_RMSE 0.5542, PF 0.1560
- Band Hs: GEFS RMSE 0.2169 m, bias -0.0426 m; persistence RMSE 0.6473 m
- shape_v13 with the old unit-area-shape labels: PF -0.1587
- Ridge AR forecasts with negative bins (clipped): 0 of 105
- True significant peaks: 275 on the full grid, 274 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_48h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_48h/linear_baseline_final.pt'}
