---
key: xiong2020prenorm
title: On Layer Normalization in the Transformer Architecture
authors: Xiong, R.; Yang, Y.; He, D.; Zheng, K.; Zheng, S.; Xing, C.; Zhang, H.; Lan, Y.; Wang, L.; Liu, T.-Y.
year: 2020
venue: Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, 10524-10533
relevance: medium
source_file: literature/xiong2020prenorm.pdf — published PMLR version (10 pp., main text only); proofs and full experimental details are deferred to the long arXiv version, arXiv:2002.04745 v2 (17 pp.), which the paper cites as "Xiong et al. (2019)"
---

# On Layer Normalization in the Transformer Architecture

**Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L.,
Liu, T.-Y. (2020).** In *Proceedings of the 37th International Conference on Machine Learning*,
PMLR vol. 119, pp. 10524-10533. No DOI (PMLR does not issue them). Metadata confirmed
2026-09-29 against the PMLR landing page (proceedings.mlr.press/v119/xiong20b.html) and the
arXiv API record (journal_ref "Published on ICML 2020").

## Summary
Asks why the original Post-LN Transformer (layer normalisation *between* residual blocks)
needs a learning-rate warm-up stage. It analyses gradients at initialisation with mean-field
theory, in a simplified setting: single-head attention, zero-initialised query/key matrices,
Xavier-initialised weights. For Post-LN, it proves that the gradient of the last FFN layer's
parameters does not depend on depth L and is large near the output. For the Pre-LN variant
(layer normalisation *inside* each residual branch, plus a final LayerNorm before the output),
the input to the final LayerNorm grows linearly with L, so gradients are normalised down by
roughly √L and stay nearly uniform across layers. Gradient-norm measurements on 6-6 to 14-14
layer models at initialisation agree with the theory.

It then shows empirically that warm-up is essential for Post-LN. On IWSLT14 De-En with Adam,
BLEU reaches only 8.45 without warm-up, against about 34 with it, and results are sensitive to
the warm-up length. Pre-LN can be trained *without* warm-up and reaches comparable final
BLEU/loss while converging faster (IWSLT14, WMT14 En-De, and BERT pre-training, where the paper
reports a roughly 40% speed-up). Moving the LayerNorm had a larger effect than switching from
Adam to RAdam.

**It does not originate Pre-LN.** The paper credits the architecture to earlier work
(Baevski & Auli 2018; Child et al. 2019; Wang et al. 2019; the tensor2tensor implementation,
Vaswani et al. 2018). Its contribution is the analysis of *why* the placement matters. The
claim that "Pre-LN outperforms Post-LN as the number of layers increases" is Wang et al.
(2019), reported in Xiong's related work. It is not Xiong's own finding. None of those earlier
papers are in this corpus.

All experiments are NLP: machine translation and masked-language-model pre-training. Nothing
is shown for regression, time-series, or geophysical data.

## Relevance to this manuscript
Medium. It is the standard method citation for the model's pre-normalisation arrangement
(`nn.Transformer(norm_first=True)`). Checked 2026-09-29 against the installed PyTorch
(2.12.1): with `norm_first=True`, `nn.Transformer` also applies a final LayerNorm on both the
encoder and decoder stacks. That exactly matches this paper's Pre-LN definition (Fig. 1b,
Table 1), so the citation describes the implemented architecture, not just a close analogue.

Two points of friction with the current text, both worth handling before submission:

1. **The stated rationale is not quite what this paper shows.** `02_methods.tex` (§ Model
   architecture, second copy) and the `nn/transformer.py` code comment justify pre-norm as
   "more stable to train at the deeper end of the search space". This paper supports
   *well-behaved gradients at initialisation and reduced sensitivity to the learning-rate
   schedule*. The depth argument comes from Wang et al. (2019), which is not in the corpus. The
   search space here is also shallow: 1-4 layers per side, against the 6-14 layers analysed
   in the paper. There is no `decisions/` entry for `norm_first`; the rationale exists only in
   the code comment.
2. **The model still uses a learning-rate warm-up.** Methods § training uses a 5-epoch
   linear warm-up, per `decisions/log/022-lr-linear-warmup.md`. This paper's headline result
   is that Pre-LN makes warm-up *removable*. That is not a contradiction: the paper shows
   warm-up *can* be dropped at its fixed learning rates, not that it must be. Decision 022 was
   also motivated by instability across a wide sampled range (10⁻³ to 1.5 × 10⁻²), not a single
   tuned value. Still, a reviewer who knows this paper may ask why warm-up is needed at all
   under pre-norm. Do not cite Xiong in the warm-up paragraph as if it supported warm-up.

## Suggested use
Cite once, where Methods introduces the pre-normalisation arrangement. `02_methods.tex` already
does this (`\citep{xiong2020prenorm}`, § Model architecture). If a one-clause rationale is
kept, word it to match what the paper actually shows, e.g. "which places layer normalisation
inside each residual branch, keeping gradient magnitudes well-behaved across layers at
initialisation and making optimisation less sensitive to the learning-rate schedule than the
original post-norm arrangement". Do not use it to back a depth-stability claim. Use it for
architecture only, not as evidence about wave or time-series forecasting.
