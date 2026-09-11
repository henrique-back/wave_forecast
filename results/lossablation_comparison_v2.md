# Loss-ablation comparison — target=shape, lead=12h, study_version=lossablation_v2

## Winning weight per phase

| phase | winning weight |
|---|---|
| baseline [test] | - |
| kl [test] | kl_loss_weight=45.87 |
| wasserstein_only [test] | wasserstein_loss_weight=9.462 |
| peak_only [test] | peak_loss_weight=0.01079 |
| wasserstein [test] | wasserstein_loss_weight=154 |
| peak [test] | peak_loss_weight=17.65 |
| combined [test] | wasserstein_loss_weight=76.56, peak_loss_weight=8.802 |

## Peak_Height_RelError (lower better)

| phase | windsea | swell | avg |
|---|---|---|---|
| baseline [test] | 0.3558 | 0.4724 | 0.4141 |
| kl [test] | 0.2954 | 0.3842 | 0.3398 |
| wasserstein_only [test] | 0.3742 | 0.4145 | 0.3944 |
| peak_only [test] | 0.5485 | 0.7602 | 0.6543 |
| wasserstein [test] | 0.3197 | 0.3878 | 0.3537 |
| peak [test] | 0.3689 | 0.4599 | 0.4144 |
| combined [test] | 0.3242 | 0.4438 | 0.3840 |

## Peak_Separation_Recall (higher better)

| phase | windsea | swell | avg |
|---|---|---|---|
| baseline [test] | 0.6543 | 0.6713 | 0.6628 |
| kl [test] | 0.4244 | 0.7907 | 0.6075 |
| wasserstein_only [test] | 0.7701 | 0.8623 | 0.8162 |
| peak_only [test] | 0.9365 | 1.0000 | 0.9682 |
| wasserstein [test] | 0.4530 | 0.7671 | 0.6100 |
| peak [test] | 0.9515 | 0.9961 | 0.9738 |
| combined [test] | 0.9159 | 0.9692 | 0.9425 |

## Tm02_RMSE (lower better)

| phase | windsea | swell | avg |
|---|---|---|---|
| baseline [test] | 0.1600 | 0.8116 | 0.4858 |
| kl [test] | 0.1163 | 0.4531 | 0.2847 |
| wasserstein_only [test] | 0.1269 | 0.6036 | 0.3653 |
| peak_only [test] | 0.2501 | 1.3349 | 0.7925 |
| wasserstein [test] | 0.1262 | 0.4520 | 0.2891 |
| peak [test] | 0.1518 | 0.8336 | 0.4927 |
| combined [test] | 0.1485 | 0.6768 | 0.4126 |

## Tm02_Bias (closer to 0 better)

| phase | windsea | swell | avg |
|---|---|---|---|
| baseline [test] | 0.0054 | -0.3220 | -0.1583 |
| kl [test] | -0.0087 | -0.0979 | -0.0533 |
| wasserstein_only [test] | -0.0148 | -0.1065 | -0.0606 |
| peak_only [test] | 0.0186 | 0.6978 | 0.3582 |
| wasserstein [test] | -0.0231 | -0.0073 | -0.0152 |
| peak [test] | 0.0106 | 0.3034 | 0.1570 |
| combined [test] | 0.0074 | 0.0329 | 0.0201 |
