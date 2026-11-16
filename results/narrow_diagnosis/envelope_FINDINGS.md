# Envelope / overlap diagnosis

Read-only study: existing development8 and one prior check case now regression. Truth is used for attribution and fixed-known-phase counterfactuals only; no new test image is read.

| Width | Correctly fitted dirty-intersecting tiles | Input dirty MSE | Protected dirty MSE | Unprotected dirty MSE | Protected adjacent MSE | Unprotected adjacent MSE | Known-phase peak above cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge8 | 3/120 | 0.002944338 | 0.002944338 | 0.003234846 | 0 | 0.0002895236 | 100.0% |
| edge16 | 31/120 | 0.002922659 | 0.002858841 | 0.002732527 | 2.938459e-07 | 0.0002552405 | 100.0% |
| edge32 | 94/240 | 0.002817584 | 0.00221152 | 0.001168895 | 7.980002e-09 | 0.0002645086 | 4.2% |

The numerator/denominator use dirty-intersecting tiles, not block centers. Full-image outputs include clipping and normalized overlap reconstruction. Correct/wrong-frequency component energies do not add because cross terms and cancellations exist.

A 3x3 grid on a32 tile has15.5px node spacing. With a Hann-weighted constant amplitude, an8px boundary strip can be diluted before the2.5x cap is applied. Enlarging the cap in a known-phase diagnostic can reveal this numerical limitation, but must not be described as an input-only solution.

CSV files retain per-tile raw leakage, capped controls, and fixed-known-phase comparisons. Do not infer texture safety from reduced fitting SSE.

Matched-frequency fitting does not imply processing: positive-gain matched fits are edge8=0/3, edge16=4/31, edge32=44/94. The three matched edge8 fits all selected the grid alternative, but production gate weights are zero.

Known prior failure48017/edge32: protected adjacentMSE=0.00111306308; correct-frequency protected contribution energy=0; wrong-frequency protected contribution energy=0.00111306308. This damage is erroneous background/texture frequency subtraction, not a correctly identified stripe envelope alone.
