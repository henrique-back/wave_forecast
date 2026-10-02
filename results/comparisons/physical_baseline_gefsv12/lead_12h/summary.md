# Physical baseline at 32012, lead 12 h

N = 107 start times (2017-09-16 to 2017-12-31, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.1716 | 0.1785 | 0.0182 | 0.5667 | -0.0058 | 0.4076 | 0.4286 | 0.7500 | 0.3753 | 0.4283 | 2.5140 | 1.9439 |
| lossablation_peak_lossablation_v3 | 2.1778 | 0.1761 | 0.0159 | 0.8391 | 0.3127 | 0.3893 | 0.3238 | 0.7073 | 0.3925 | 0.3697 | 2.5140 | 1.7383 |
| ridge_AR | 2.2444 | 0.1509 | 0.0155 | 0.6753 | 0.2021 | 0.3871 | 0.2952 | 0.7500 | 0.4215 | 0.3457 | 2.5140 | 1.8411 |
| persistence | 2.6434 | 0.0000 | 0.0163 | 0.8051 | 0.2055 | 0.3422 | 0.4952 | 0.7012 | 0.4280 | 0.4337 | 2.5140 | 2.5981 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| lossablation_peak_lossablation_v3 | Shape_RMSE | 0.0062 | [-0.4342, 0.3972] | 0.5050 |
| lossablation_peak_lossablation_v3 | Shape_Wasserstein | -0.0023 | [-0.0049, 0.0003] | 0.9560 |
| lossablation_peak_lossablation_v3 | Tm02_RMSE | 0.2724 | [0.0776, 0.4461] | 0.0030 |
| lossablation_peak_lossablation_v3 | peak_fidelity | -0.0183 | [-0.0671, 0.0339] | 0.2240 |
| lossablation_peak_lossablation_v3 | Peak_Separation_Recall_windsea | -0.1048 | [-0.1650, -0.0481] | 0.0000 |
| lossablation_peak_lossablation_v3 | Peak_Separation_Recall_swell | -0.0427 | [-0.1133, 0.0270] | 0.0950 |
| lossablation_peak_lossablation_v3 | Peak_Height_RelError_windsea | 0.0172 | [-0.0283, 0.0630] | 0.2520 |
| lossablation_peak_lossablation_v3 | Peak_Height_RelError_swell | -0.0586 | [-0.1678, 0.0350] | 0.8900 |
| ridge_AR | Shape_RMSE | 0.0728 | [-0.3884, 0.4926] | 0.4070 |
| ridge_AR | Shape_Wasserstein | -0.0026 | [-0.0049, -0.0003] | 0.9870 |
| ridge_AR | Tm02_RMSE | 0.1086 | [-0.0845, 0.2778] | 0.1590 |
| ridge_AR | peak_fidelity | -0.0205 | [-0.0723, 0.0280] | 0.2060 |
| ridge_AR | Peak_Separation_Recall_windsea | -0.1333 | [-0.2095, -0.0693] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_swell | 0.0000 | [-0.0659, 0.0614] | 0.4500 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0463 | [0.0054, 0.0904] | 0.0100 |
| ridge_AR | Peak_Height_RelError_swell | -0.0826 | [-0.1892, 0.0136] | 0.9500 |
| persistence | Shape_RMSE | 0.4718 | [0.0321, 0.8826] | 0.0210 |
| persistence | Shape_Wasserstein | -0.0019 | [-0.0044, 0.0007] | 0.9140 |
| persistence | Tm02_RMSE | 0.2384 | [0.0360, 0.4162] | 0.0060 |
| persistence | peak_fidelity | -0.0654 | [-0.1186, -0.0081] | 0.0120 |
| persistence | Peak_Separation_Recall_windsea | 0.0667 | [-0.0368, 0.1699] | 0.8650 |
| persistence | Peak_Separation_Recall_swell | -0.0488 | [-0.1243, 0.0132] | 0.0700 |
| persistence | Peak_Height_RelError_windsea | 0.0527 | [0.0005, 0.1112] | 0.0250 |
| persistence | Peak_Height_RelError_swell | 0.0054 | [-0.1049, 0.1058] | 0.4720 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0363, Tm02_RMSE 0.5526, PF 0.3799
- Band Hs: GEFS RMSE 0.2425 m, bias -0.0475 m; persistence RMSE 0.3529 m
- lossablation_peak_lossablation_v3 with the old unit-area-shape labels: PF 0.4177
- Ridge AR forecasts with negative bins (clipped): 8 of 107
- True significant peaks: 269 on the full grid, 269 on the band
- Checkpoints: {'lossablation_peak_lossablation_v3': 'results/lossablation_peak_lossablation_v3/shape/lead_12h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_12h/linear_baseline_final.pt'}
