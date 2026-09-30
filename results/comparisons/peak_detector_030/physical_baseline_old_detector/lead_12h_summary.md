# Physical baseline at 32012, lead 12 h

N = 107 start times (2017-09-16 to 2017-12-31, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.1716 | 0.1785 | 0.0182 | 0.5667 | -0.0058 | 0.1519 | 0.3284 | 0.7608 | 0.3416 | 0.4438 | 4.2897 | 2.3458 |
| shape_v13 | 2.7684 | -0.0473 | 0.0173 | 0.7802 | 0.2785 | 0.5527 | 0.9167 | 0.9647 | 0.3960 | 0.3800 | 4.2897 | 7.6262 |
| ridge_AR | 2.2444 | 0.1509 | 0.0155 | 0.6753 | 0.2021 | 0.3555 | 0.5735 | 0.8392 | 0.3883 | 0.3135 | 4.2897 | 3.5140 |
| persistence | 2.6434 | 0.0000 | 0.0163 | 0.8051 | 0.2055 | 0.3801 | 0.7108 | 0.8431 | 0.3936 | 0.4002 | 4.2897 | 4.1028 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 0.5968 | [0.2434, 0.9451] | 0.0010 |
| shape_v13 | Shape_Wasserstein | -0.0009 | [-0.0027, 0.0011] | 0.7960 |
| shape_v13 | Tm02_RMSE | 0.2135 | [0.0737, 0.3460] | 0.0020 |
| shape_v13 | peak_fidelity_SS | 0.4008 | [0.3336, 0.4673] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | 0.5882 | [0.5125, 0.6557] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_swell | 0.2039 | [0.1536, 0.2510] | 1.0000 |
| shape_v13 | Peak_Height_RelError_windsea | 0.0544 | [0.0254, 0.0856] | 0.0000 |
| shape_v13 | Peak_Height_RelError_swell | -0.0638 | [-0.1545, 0.0143] | 0.9470 |
| ridge_AR | Shape_RMSE | 0.0728 | [-0.3884, 0.4926] | 0.4070 |
| ridge_AR | Shape_Wasserstein | -0.0026 | [-0.0049, -0.0003] | 0.9870 |
| ridge_AR | Tm02_RMSE | 0.1086 | [-0.0845, 0.2778] | 0.1590 |
| ridge_AR | peak_fidelity_SS | 0.2036 | [0.1421, 0.2694] | 1.0000 |
| ridge_AR | Peak_Separation_Recall_windsea | 0.2451 | [0.1722, 0.3250] | 1.0000 |
| ridge_AR | Peak_Separation_Recall_swell | 0.0784 | [0.0269, 0.1336] | 1.0000 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0467 | [0.0171, 0.0811] | 0.0030 |
| ridge_AR | Peak_Height_RelError_swell | -0.1304 | [-0.2205, -0.0559] | 1.0000 |
| persistence | Shape_RMSE | 0.4718 | [0.0321, 0.8826] | 0.0210 |
| persistence | Shape_Wasserstein | -0.0019 | [-0.0044, 0.0007] | 0.9140 |
| persistence | Tm02_RMSE | 0.2384 | [0.0360, 0.4162] | 0.0060 |
| persistence | peak_fidelity_SS | 0.2281 | [0.1567, 0.2977] | 1.0000 |
| persistence | Peak_Separation_Recall_windsea | 0.3824 | [0.3163, 0.4420] | 1.0000 |
| persistence | Peak_Separation_Recall_swell | 0.0824 | [0.0198, 0.1423] | 0.9940 |
| persistence | Peak_Height_RelError_windsea | 0.0520 | [0.0113, 0.0967] | 0.0080 |
| persistence | Peak_Height_RelError_swell | -0.0436 | [-0.1297, 0.0341] | 0.8560 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0363, Tm02_RMSE 0.5526, PF 0.1524
- Band Hs: GEFS RMSE 0.2425 m, bias -0.0475 m; persistence RMSE 0.3529 m
- shape_v13 with evaluate()'s shape-space labels: PF 0.5548
- Ridge AR forecasts with negative bins (clipped): 8 of 107
- True significant peaks: 459 on the full grid, 459 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_12h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_12h/linear_baseline_final.pt'}
