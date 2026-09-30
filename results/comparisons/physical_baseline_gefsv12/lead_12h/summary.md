# Physical baseline at 32012, lead 12 h

N = 107 start times (2017-09-16 to 2017-12-31, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.1716 | 0.1785 | 0.0182 | 0.5667 | -0.0058 | 0.1875 | 0.4286 | 0.7500 | 0.3753 | 0.4283 | 2.5140 | 1.9439 |
| shape_v13 | 2.7684 | -0.0473 | 0.0173 | 0.7802 | 0.2785 | -0.1156 | 0.1714 | 0.4390 | 0.4274 | 0.4143 | 2.5140 | 1.2336 |
| ridge_AR | 2.2444 | 0.1509 | 0.0155 | 0.6753 | 0.2021 | 0.1390 | 0.2952 | 0.7500 | 0.4215 | 0.3457 | 2.5140 | 1.8411 |
| persistence | 2.6434 | 0.0000 | 0.0163 | 0.8051 | 0.2055 | 0.1674 | 0.4952 | 0.7012 | 0.4280 | 0.4337 | 2.5140 | 2.5981 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 0.5968 | [0.2434, 0.9451] | 0.0010 |
| shape_v13 | Shape_Wasserstein | -0.0009 | [-0.0027, 0.0011] | 0.7960 |
| shape_v13 | Tm02_RMSE | 0.2135 | [0.0737, 0.3460] | 0.0020 |
| shape_v13 | peak_fidelity_SS | -0.3031 | [-0.3805, -0.2188] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | -0.2571 | [-0.3696, -0.1456] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_swell | -0.3110 | [-0.3765, -0.2547] | 0.0000 |
| shape_v13 | Peak_Height_RelError_windsea | 0.0521 | [0.0091, 0.0981] | 0.0070 |
| shape_v13 | Peak_Height_RelError_swell | -0.0140 | [-0.1167, 0.0763] | 0.6070 |
| ridge_AR | Shape_RMSE | 0.0728 | [-0.3884, 0.4926] | 0.4070 |
| ridge_AR | Shape_Wasserstein | -0.0026 | [-0.0049, -0.0003] | 0.9870 |
| ridge_AR | Tm02_RMSE | 0.1086 | [-0.0845, 0.2778] | 0.1590 |
| ridge_AR | peak_fidelity_SS | -0.0485 | [-0.1298, 0.0333] | 0.1210 |
| ridge_AR | Peak_Separation_Recall_windsea | -0.1333 | [-0.2095, -0.0693] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_swell | 0.0000 | [-0.0659, 0.0614] | 0.4500 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0463 | [0.0054, 0.0904] | 0.0100 |
| ridge_AR | Peak_Height_RelError_swell | -0.0826 | [-0.1892, 0.0136] | 0.9500 |
| persistence | Shape_RMSE | 0.4718 | [0.0321, 0.8826] | 0.0210 |
| persistence | Shape_Wasserstein | -0.0019 | [-0.0044, 0.0007] | 0.9140 |
| persistence | Tm02_RMSE | 0.2384 | [0.0360, 0.4162] | 0.0060 |
| persistence | peak_fidelity_SS | -0.0201 | [-0.1116, 0.0784] | 0.3460 |
| persistence | Peak_Separation_Recall_windsea | 0.0667 | [-0.0368, 0.1699] | 0.8650 |
| persistence | Peak_Separation_Recall_swell | -0.0488 | [-0.1243, 0.0132] | 0.0700 |
| persistence | Peak_Height_RelError_windsea | 0.0527 | [0.0005, 0.1112] | 0.0250 |
| persistence | Peak_Height_RelError_swell | 0.0054 | [-0.1049, 0.1058] | 0.4720 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0363, Tm02_RMSE 0.5526, PF 0.1432
- Band Hs: GEFS RMSE 0.2425 m, bias -0.0475 m; persistence RMSE 0.3529 m
- shape_v13 with the old unit-area-shape labels: PF -0.0642
- Ridge AR forecasts with negative bins (clipped): 8 of 107
- True significant peaks: 269 on the full grid, 269 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_12h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_12h/linear_baseline_final.pt'}
