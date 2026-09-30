# Physical baseline at 32012, lead 24 h

N = 106 start times (2017-09-16 to 2017-12-30, 00Z); band 0.0375-0.4850 Hz (45 bins); labels on the physical density.

| Model | Shape_RMSE | Shape_SS | Shape_Wasserstein | Tm02_RMSE | Tm02_Bias | peak_fidelity_SS | Peak_Separation_Recall_windsea | Peak_Separation_Recall_swell | Peak_Height_RelError_windsea | Peak_Height_RelError_swell | Peak_Count_True_Mean | Peak_Count_Pred_Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEFSv12_c00 | 2.0744 | 0.3880 | 0.0171 | 0.4972 | -0.0374 | 0.1596 | 0.2971 | 0.7893 | 0.3619 | 0.4054 | 4.1132 | 2.3585 |
| shape_v13 | 2.9419 | 0.1321 | 0.0217 | 0.9606 | -0.3136 | 0.5348 | 0.8743 | 0.9923 | 0.3609 | 0.4362 | 4.1132 | 7.8113 |
| ridge_AR | 2.7196 | 0.1977 | 0.0186 | 0.8304 | 0.0881 | 0.2176 | 0.4343 | 0.7739 | 0.4036 | 0.3693 | 4.1132 | 3.1509 |
| persistence | 3.3896 | 0.0000 | 0.0197 | 0.9838 | 0.0136 | 0.2778 | 0.6800 | 0.8199 | 0.4538 | 0.4906 | 4.1132 | 4.0849 |

Difference from GEFSv12_c00 (model − GEFS), 95% block-bootstrap interval, P(model better); B = 1000, blocks of 3 days.

| Model | Metric | Diff | 95% CI | P(better) |
|---|---|---|---|---|
| shape_v13 | Shape_RMSE | 0.8675 | [0.6010, 1.1770] | 0.0000 |
| shape_v13 | Shape_Wasserstein | 0.0046 | [0.0025, 0.0070] | 0.0000 |
| shape_v13 | Tm02_RMSE | 0.4634 | [0.3168, 0.5979] | 0.0000 |
| shape_v13 | peak_fidelity_SS | 0.3752 | [0.3039, 0.4366] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_windsea | 0.5771 | [0.5050, 0.6527] | 1.0000 |
| shape_v13 | Peak_Separation_Recall_swell | 0.2031 | [0.1417, 0.2536] | 1.0000 |
| shape_v13 | Peak_Height_RelError_windsea | -0.0010 | [-0.0461, 0.0407] | 0.5300 |
| shape_v13 | Peak_Height_RelError_swell | 0.0308 | [-0.0462, 0.1121] | 0.1940 |
| ridge_AR | Shape_RMSE | 0.6451 | [0.3760, 0.9517] | 0.0000 |
| ridge_AR | Shape_Wasserstein | 0.0015 | [-0.0004, 0.0033] | 0.0590 |
| ridge_AR | Tm02_RMSE | 0.3333 | [0.2270, 0.4363] | 0.0000 |
| ridge_AR | peak_fidelity_SS | 0.0581 | [-0.0236, 0.1256] | 0.9210 |
| ridge_AR | Peak_Separation_Recall_windsea | 0.1371 | [0.0519, 0.2111] | 1.0000 |
| ridge_AR | Peak_Separation_Recall_swell | -0.0153 | [-0.0814, 0.0424] | 0.2890 |
| ridge_AR | Peak_Height_RelError_windsea | 0.0417 | [0.0026, 0.0805] | 0.0220 |
| ridge_AR | Peak_Height_RelError_swell | -0.0361 | [-0.1100, 0.0391] | 0.8030 |
| persistence | Shape_RMSE | 1.3152 | [0.9237, 1.7606] | 0.0000 |
| persistence | Shape_Wasserstein | 0.0026 | [-0.0001, 0.0055] | 0.0320 |
| persistence | Tm02_RMSE | 0.4866 | [0.3432, 0.6250] | 0.0000 |
| persistence | peak_fidelity_SS | 0.1182 | [-0.0043, 0.2108] | 0.9700 |
| persistence | Peak_Separation_Recall_windsea | 0.3829 | [0.2727, 0.4742] | 1.0000 |
| persistence | Peak_Separation_Recall_swell | 0.0307 | [-0.0520, 0.1062] | 0.7420 |
| persistence | Peak_Height_RelError_windsea | 0.0918 | [0.0398, 0.1443] | 0.0000 |
| persistence | Peak_Height_RelError_swell | 0.0852 | [0.0038, 0.1809] | 0.0200 |

Context:

- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE 2.0246, Tm02_RMSE 0.5528, PF 0.1616
- Band Hs: GEFS RMSE 0.2037 m, bias -0.0429 m; persistence RMSE 0.4408 m
- shape_v13 with evaluate()'s shape-space labels: PF 0.5305
- Ridge AR forecasts with negative bins (clipped): 6 of 106
- True significant peaks: 437 on the full grid, 436 on the band
- Checkpoints: {'shape_v13': 'results/shape_v13/shape/lead_24h/best_model.pt', 'ridge_AR': 'results/linear_baseline/shape/lead_24h/linear_baseline_final.pt'}
