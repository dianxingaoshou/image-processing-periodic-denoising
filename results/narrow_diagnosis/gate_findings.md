# 门控逐层定位（开发集，只读诊断）

Already-used dev8, amplitude .12, supportall32/grid. All dirty-intersecting tiles included. Fractions weighted by Hann mass of dirty overlap; correlated and not pixel recall. Counterfactuals reuse existing residuals; local oracle is not a full-image optimum. Frequency recovered by continuous fitting proceeds to gate attribution even if raw pool missed.

| 条件 | 模式 | 平均含噪区MSE | 平均相邻干净区MSE |
|---|---|---:|---:|
| edge8 | input | 0.00294433809 | 0 |
| edge8 | normal | 0.00294433809 | 0 |
| edge8 | bypass_candidate_gate | 0.00299704292 | 7.35463285e-05 |
| edge8 | all_gains_one | 0.00323484605 | 0.000289523603 |
| edge8 | local_oracle_existing_residuals | 0.00289988869 | 1.89891354e-07 |
| edge16 | input | 0.00292265888 | 0 |
| edge16 | normal | 0.00285884057 | 2.9384585e-07 |
| edge16 | bypass_candidate_gate | 0.00265542387 | 4.18711608e-05 |
| edge16 | all_gains_one | 0.00273252707 | 0.000255240511 |
| edge16 | local_oracle_existing_residuals | 0.00242051931 | 3.62372221e-06 |
| edge32 | input | 0.00281758401 | 0 |
| edge32 | normal | 0.00221152013 | 7.98000156e-09 |
| edge32 | bypass_candidate_gate | 0.00173243619 | 5.15005823e-08 |
| edge32 | all_gains_one | 0.00116889485 | 0.00026450863 |
| edge32 | local_oracle_existing_residuals | 0.000934102905 | 3.17357716e-07 |

## 主要互斥失败阶段（按含噪重叠Hann质量加权）

| 条件 | 阶段 | 质量占比 |
|---|---|---:|
| edge8 | no_correct_candidate | 97.50% |
| edge8 | candidate_gate_rejected | 2.50% |
| edge16 | no_correct_candidate | 74.17% |
| edge16 | candidate_gate_rejected | 22.50% |
| edge16 | processed_correct_candidate | 3.33% |
| edge32 | no_correct_candidate | 45.75% |
| edge32 | candidate_gate_rejected | 27.84% |
| edge32 | processed_correct_candidate | 21.32% |
| edge32 | wrong_selected_or_fitted_frequency | 4.25% |
| edge32 | reliability_zero | 0.83% |

## 解释

8像素主要缺少正确候选；放开门控对同一组残差反而同时增加含噪区与邻近损伤。16像素存在正确候选被资格门控拒绝，但直接解除门控会带来明显相邻区损伤；不能据此全局降低阈值。

被门控拒绝的16像素正确频率候选，峰值分数中位数约3.89，低于4.5；频谱证据约0.714、邻块一致性约0.953，但跨窗支撑中位数为0，无法进入要求跨窗>=0.8的回退。

同残差的孤立块参考最优增益仍不能完全恢复8像素条纹，说明仅改门控不足。参考增益包含真值，仅用于离线诊断；不作为算法输入。
