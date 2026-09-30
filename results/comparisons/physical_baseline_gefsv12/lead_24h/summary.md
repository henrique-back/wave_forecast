# Physical baseline at 32012, lead 24 h

N = 106 start times (2017-09-16 to 2017-12-30, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.0744 | 0.3880 | 0.0171 | 0.4972 | -0.0374 | 0.1307 | 0.2736 | 0.7692 | 0.3975 | 0.3839 | 2.5943 | 1.9623 |
| shape_v13 | 2.9419 | 0.1321 | 0.0217 | 0.9606 | -0.3136 | -0.2304 | 0.0000 | 0.4438 | 0.4095 | 0.4952 | 2.5943 | 1.0377 |
| ridge_AR | 2.7196 | 0.1977 | 0.0186 | 0.8304 | 0.0881 | -0.0242 | 0.1415 | 0.6213 | 0.4184 | 0.3928 | 2.5943 | 1.6415 |
| persistence | 3.3896 | 0.0000 | 0.0197 | 0.9838 | 0.0136 | 0.0231 | 0.3585 | 0.6746 | 0.4918 | 0.4951 | 2.5943 | 2.6038 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 0.8675 | [0.6010, 1.1770] | 0.0000 |
| shape_v13 | Shape_Wasserstein | 0.0046 | [0.0025, 0.0070] | 0.0000 |
| shape_v13 | Tm02_RMSE | 0.4634 | [0.3168, 0.5979] | 0.0000 |
| shape_v13 | peak_fidelity_SS | -0.3612 | [-0.4427, -0.2800] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | -0.2736 | [-0.3496, -0.2037] | 0.0000 |
| shape_v13 | Peak_Separation_Recall_swell | -0.3254 | [-0.4138, -0.2394] | 0.0000 |
| shape_v13 | Peak_Height_RelError_windsea | 0.0120 | [-0.0469, 0.0706] | 0.3440 |
| shape_v13 | Peak_Height_RelError_swell | 0.1113 | [0.0253, 0.1898] | 0.0070 |
| ridge_AR | Shape_RMSE | 0.6451 | [0.3760, 0.9517] | 0.0000 |
| ridge_AR | Shape_Wasserstein | 0.0015 | [-0.0004, 0.0033] | 0.0590 |
| ridge_AR | Tm02_RMSE | 0.3333 | [0.2270, 0.4363] | 0.0000 |
| ridge_AR | peak_fidelity_SS | -0.1549 | [-0.2356, -0.0818] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_windsea | -0.1321 | [-0.2024, -0.0594] | 0.0000 |
| ridge_AR | Peak_Separation_Recall_swell | -0.1479 | [-0.2184, -0.0838] | 0.0000 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0209 | [-0.0343, 0.0700] | 0.2060 |
| ridge_AR | Peak_Height_RelError_swell | 0.0089 | [-0.0587, 0.0784] | 0.3710 |
| persistence | Shape_RMSE | 1.3152 | [0.9237, 1.7606] | 0.0000 |
| persistence | Shape_Wasserstein | 0.0026 | [-0.0001, 0.0055] | 0.0320 |
| persistence | Tm02_RMSE | 0.4866 | [0.3432, 0.6250] | 0.0000 |
| persistence | peak_fidelity_SS | -0.1077 | [-0.1978, -0.0184] | 0.0080 |
| persistence | Peak_Separation_Recall_windsea | 0.0849 | [-0.0105, 0.1770] | 0.9440 |
| persistence | Peak_Separation_Recall_swell | -0.0947 | [-0.1868, -0.0176] | 0.0060 |
| persistence | Peak_Height_RelError_windsea | 0.0943 | [0.0267, 0.1576] | 0.0070 |
| persistence | Peak_Height_RelError_swell | 0.1112 | [0.0256, 0.2008] | 0.0030 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0246, Tm02_RMSE 0.5528, PF 0.1523
- Band Hs: GEFS RMSE 0.2037 m, bias -0.0429 m; persistence RMSE 0.4408 m
- shape_v13 with the old unit-area-shape labels: PF -0.1348
- Ridge AR forecasts with negative bins (clipped): 6 of 106
- True significant peaks: 276 on the full grid, 275 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_24h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_24h/linear_baseline_final.pt'}
