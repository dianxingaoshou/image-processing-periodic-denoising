# 拒绝案例：对比上下文与减除资格分离

保持候选、系数、0.02幅度／0.8稳定性／0.08收益门槛和至少3个连续资格行不变，仅修改邻带取样上下文。完整方案不填资格孔、不合并输出段、不放宽0.35比值；还约束幅度上下文宽度≤32。

ratio_relaxed只将比值放宽至0.9，用作负面对照；context_no_width用于检查缺少宽幅度区域保护的后果。所有主表强度1，方法共用同一候选池与系数。

## development

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS/clean | baseline | 8 | — | 0 | — | 0 |
| BSDS/clean | context_full | 8 | — | 0 | — | 0 |
| BSDS/clean | context_no_width | 8 | — | 1.4199072e-05 | — | 0.0005196569 |
| BSDS/clean | ratio_relaxed | 8 | — | 0 | — | 0 |
| BSDS/edge16 | baseline | 8 | 0.0017441889 | 0 | 0 | 0.00253656 |
| BSDS/edge16 | context_full | 8 | 0.0015548403 | 0 | 0 | 0.002535895 |
| BSDS/edge16 | context_no_width | 8 | 0.0015548403 | 1.5145677e-05 | 0 | 0.003052597 |
| BSDS/edge16 | ratio_relaxed | 8 | 0.0017441889 | 0 | 0 | 0.00253656 |
| BSDS/edge32 | baseline | 8 | 0.0018967434 | 0 | 0 | 0.004922092 |
| BSDS/edge32 | context_full | 8 | 0.0014207805 | 0 | 0 | 0.004920167 |
| BSDS/edge32 | context_no_width | 8 | 0.0014207805 | 1.6227511e-05 | 0 | 0.004947993 |
| BSDS/edge32 | ratio_relaxed | 8 | 0.0017423606 | 0 | 0 | 0.004922092 |
| BSDS/edge8 | baseline | 8 | 0.0017190189 | 0 | 0 | 0.001343205 |
| BSDS/edge8 | context_full | 8 | 0.0017190189 | 0 | 0 | 0.001343205 |
| BSDS/edge8 | context_no_width | 8 | 0.0017190189 | 1.4657107e-05 | 0 | 0.001861377 |
| BSDS/edge8 | ratio_relaxed | 8 | 0.0017190189 | 0 | 0 | 0.001343205 |
| BSDS/global | baseline | 8 | 0.006904087 | — | — | 0.07614058 |
| BSDS/global | context_full | 8 | 0.006904087 | — | — | 0.07614058 |
| BSDS/global | context_no_width | 8 | 0.0071108939 | — | — | 0.0774273 |
| BSDS/global | ratio_relaxed | 8 | 0.0064435539 | — | — | 0.07437242 |
| BSDS/interior16 | baseline | 8 | 0.0020729268 | 0 | 0 | 0.00238609 |
| BSDS/interior16 | context_full | 8 | 0.0018367053 | 0 | 0 | 0.002381683 |
| BSDS/interior16 | context_no_width | 8 | 0.0018367053 | 1.5145677e-05 | 0.00011359258 | 0.002381683 |
| BSDS/interior16 | ratio_relaxed | 8 | 0.0018367053 | 0 | 0 | 0.002381683 |
| BSDS/patch32 | baseline | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |
| BSDS/patch32 | context_full | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |
| BSDS/patch32 | context_no_width | 8 | 0.0012180134 | 1.4424206e-05 | 7.7384103e-05 | 0.0008184331 |
| BSDS/patch32 | ratio_relaxed | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |

| 方法 | 局部输入 | 严格净改善 | 新恢复旧拒绝 | 旧成功退步/丢失 | 新增干净/相邻损伤 |
|---|---:|---:|---:|---:|---:|
| ratio_relaxed | 40 | 2 | 2 | 0/0 | 0/0 |
| context_no_width | 40 | 4 | 4 | 1/1 | 5/2 |
| context_full | 40 | 5 | 5 | 0/0 | 0/0 |

## controls

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| analytic/broad_amplitude_slope_clean | baseline | 1 | — | 0 | — | 0 |
| analytic/broad_amplitude_slope_clean | context_full | 1 | — | 0 | — | 0 |
| analytic/broad_amplitude_slope_clean | context_no_width | 1 | — | 0 | — | 0 |
| analytic/broad_amplitude_slope_clean | ratio_relaxed | 1 | — | 0 | — | 0 |
| analytic/broad_phase_banded_clean | baseline | 1 | — | 0 | — | 0 |
| analytic/broad_phase_banded_clean | context_full | 1 | — | 0 | — | 0 |
| analytic/broad_phase_banded_clean | context_no_width | 1 | — | 0.000390625 | — | 0.004981516 |
| analytic/broad_phase_banded_clean | ratio_relaxed | 1 | — | 0.000390625 | — | 0.004981516 |
| analytic/broad_texture_edge16 | baseline | 1 | 1.4542505e-05 | 0 | 0 | 9.768058e-05 |
| analytic/broad_texture_edge16 | context_full | 1 | 1.4542505e-05 | 0 | 0 | 9.768058e-05 |
| analytic/broad_texture_edge16 | context_no_width | 1 | 1.4542505e-05 | 0 | 0 | 9.768058e-05 |
| analytic/broad_texture_edge16 | ratio_relaxed | 1 | 1.4542505e-05 | 0 | 0 | 9.768058e-05 |
| analytic/broad_texture_edge8 | baseline | 1 | 2.0789014e-05 | 0 | 0 | 6.365122e-05 |
| analytic/broad_texture_edge8 | context_full | 1 | 2.0789014e-05 | 0 | 0 | 6.365122e-05 |
| analytic/broad_texture_edge8 | context_no_width | 1 | 2.0789014e-05 | 0 | 0 | 6.365122e-05 |
| analytic/broad_texture_edge8 | ratio_relaxed | 1 | 2.0789014e-05 | 0 | 0 | 6.365122e-05 |
| analytic/compact_legitimate_stripe_clean | baseline | 1 | — | 0.000225 | — | 0.002383952 |
| analytic/compact_legitimate_stripe_clean | context_full | 1 | — | 0.000225 | — | 0.002383952 |
| analytic/compact_legitimate_stripe_clean | context_no_width | 1 | — | 0.000225 | — | 0.002383952 |
| analytic/compact_legitimate_stripe_clean | ratio_relaxed | 1 | — | 0.000225 | — | 0.002383952 |
| analytic/fragmented_coherence_edge16 | baseline | 1 | 0.0031199557 | 0 | 0 | 0.002540216 |
| analytic/fragmented_coherence_edge16 | context_full | 1 | 0.0020773605 | 0 | 0 | 0.001778653 |
| analytic/fragmented_coherence_edge16 | context_no_width | 1 | 0.0020773605 | 0 | 0 | 0.001778653 |
| analytic/fragmented_coherence_edge16 | ratio_relaxed | 1 | 0.0020773605 | 0 | 0 | 0.001778653 |
| analytic/global_periodic_clean | baseline | 1 | — | 0 | — | 0 |
| analytic/global_periodic_clean | context_full | 1 | — | 0 | — | 0 |
| analytic/global_periodic_clean | context_no_width | 1 | — | 0 | — | 0 |
| analytic/global_periodic_clean | ratio_relaxed | 1 | — | 0 | — | 0 |
| analytic/sharp_edge_clutter_clean | baseline | 1 | — | 0 | — | 0 |
| analytic/sharp_edge_clutter_clean | context_full | 1 | — | 0 | — | 0 |
| analytic/sharp_edge_clutter_clean | context_no_width | 1 | — | 0 | — | 0 |
| analytic/sharp_edge_clutter_clean | ratio_relaxed | 1 | — | 0 | — | 0 |
| analytic/smooth_background_edge16 | baseline | 1 | 1.4430577e-05 | 0 | 0 | 9.149579e-05 |
| analytic/smooth_background_edge16 | context_full | 1 | 1.4430577e-05 | 0 | 0 | 9.149579e-05 |
| analytic/smooth_background_edge16 | context_no_width | 1 | 1.4430577e-05 | 0 | 0 | 9.149579e-05 |
| analytic/smooth_background_edge16 | ratio_relaxed | 1 | 1.4430577e-05 | 0 | 0 | 9.149579e-05 |
| analytic/weak_edge16 | baseline | 1 | 3.1199557e-05 | 0 | 0 | 0.0002540216 |
| analytic/weak_edge16 | context_full | 1 | 3.1199557e-05 | 0 | 0 | 0.0002540216 |
| analytic/weak_edge16 | context_no_width | 1 | 3.1199557e-05 | 0 | 0 | 0.0002540216 |
| analytic/weak_edge16 | ratio_relaxed | 1 | 3.1199557e-05 | 0 | 0 | 0.0002540216 |

| 方法 | 局部输入 | 严格净改善 | 新恢复旧拒绝 | 旧成功退步/丢失 | 新增干净/相邻损伤 |
|---|---:|---:|---:|---:|---:|
| ratio_relaxed | 5 | 1 | 1 | 0/0 | 0/0 |
| context_no_width | 5 | 1 | 1 | 0/0 | 0/0 |
| context_full | 5 | 1 | 1 | 0/0 | 0/0 |

## regression

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS_regression/clean | baseline | 4 | — | 0 | — | 0 |
| BSDS_regression/clean | context_full | 4 | — | 0 | — | 0 |
| BSDS_regression/clean | context_no_width | 4 | — | 0 | — | 0 |
| BSDS_regression/clean | ratio_relaxed | 4 | — | 2.8534598e-05 | — | 0.001198101 |
| BSDS_regression/edge16 | baseline | 4 | 0.0011002284 | 0 | 0 | 0.002444041 |
| BSDS_regression/edge16 | context_full | 4 | 0.00065930648 | 0 | 0 | 0.001404029 |
| BSDS_regression/edge16 | context_no_width | 4 | 0.00065930648 | 0 | 0 | 0.001404029 |
| BSDS_regression/edge16 | ratio_relaxed | 4 | 0.0011002284 | 3.0436905e-05 | 0 | 0.002444041 |
| BSDS_regression/edge32 | baseline | 4 | 0.0021471333 | 0 | 0 | 0.004923769 |
| BSDS_regression/edge32 | context_full | 4 | 0.00084646567 | 0 | 0 | 0.003562588 |
| BSDS_regression/edge32 | context_no_width | 4 | 0.00084646567 | 0 | 0 | 0.003562588 |
| BSDS_regression/edge32 | ratio_relaxed | 4 | 0.0013028721 | 3.2610969e-05 | 0 | 0.004768261 |
| BSDS_regression/edge8 | baseline | 4 | 0.00099848974 | 0 | 0 | 0.001294563 |
| BSDS_regression/edge8 | context_full | 4 | 0.00051621748 | 0 | 0 | 0.0005627215 |
| BSDS_regression/edge8 | context_no_width | 4 | 0.00051621748 | 0 | 0 | 0.0005627215 |
| BSDS_regression/edge8 | ratio_relaxed | 4 | 0.00099848974 | 2.9455069e-05 | 0 | 0.001443984 |
| DTD_regression/clean | baseline | 12 | — | 0 | — | 0 |
| DTD_regression/clean | context_full | 12 | — | 0 | — | 0 |
| DTD_regression/clean | context_no_width | 12 | — | 0.00053155367 | — | 0.009798455 |
| DTD_regression/clean | ratio_relaxed | 12 | — | 0.0014462022 | — | 0.01824691 |

| 方法 | 局部输入 | 严格净改善 | 新恢复旧拒绝 | 旧成功退步/丢失 | 新增干净/相邻损伤 |
|---|---:|---:|---:|---:|---:|
| ratio_relaxed | 12 | 1 | 1 | 2/2 | 3/0 |
| context_no_width | 12 | 5 | 5 | 0/0 | 0/0 |
| context_full | 12 | 5 | 5 | 0/0 | 0/0 |

## check

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS/clean | baseline | 4 | — | 0 | — | 0 |
| BSDS/clean | context_full | 4 | — | 0 | — | 0 |
| BSDS/clean | context_no_width | 4 | — | 2.2953363e-05 | — | 0.001198787 |
| BSDS/clean | ratio_relaxed | 4 | — | 0 | — | 0 |
| BSDS/edge16 | baseline | 4 | 0.0019495028 | 0 | 0 | 0.002536833 |
| BSDS/edge16 | context_full | 4 | 0.0019495028 | 0 | 0 | 0.002536833 |
| BSDS/edge16 | context_no_width | 4 | 0.001966129 | 2.3359741e-05 | 2.348676e-05 | 0.003664184 |
| BSDS/edge16 | ratio_relaxed | 4 | 0.0019495028 | 0 | 0 | 0.002536833 |
| BSDS/edge32 | baseline | 4 | 0.0029629187 | 0 | 0 | 0.004921954 |
| BSDS/edge32 | context_full | 4 | 0.0020691494 | 0 | 0 | 0.004921093 |
| BSDS/edge32 | context_no_width | 4 | 0.0020912874 | 2.3350931e-05 | 3.0256211e-05 | 0.005954911 |
| BSDS/edge32 | ratio_relaxed | 4 | 0.0021288991 | 0 | 0 | 0.004921093 |
| BSDS/edge8 | baseline | 4 | 0.0018212733 | 0 | 0 | 0.001343238 |
| BSDS/edge8 | context_full | 4 | 0.0018212733 | 0 | 0 | 0.001343238 |
| BSDS/edge8 | context_no_width | 4 | 0.0018290637 | 2.3592e-05 | 3.4994862e-05 | 0.002511678 |
| BSDS/edge8 | ratio_relaxed | 4 | 0.0018212733 | 0 | 0 | 0.001343238 |
| BSDS/global | baseline | 4 | 0.0070595581 | — | — | 0.07607666 |
| BSDS/global | context_full | 4 | 0.0070595581 | — | — | 0.07607666 |
| BSDS/global | context_no_width | 4 | 0.0070595581 | — | — | 0.07607666 |
| BSDS/global | ratio_relaxed | 4 | 0.0063227267 | — | — | 0.07105903 |
| BSDS/interior16 | baseline | 4 | 0.0018233201 | 0 | 0 | 0.002393065 |
| BSDS/interior16 | context_full | 4 | 0.0018233201 | 0 | 0 | 0.002393065 |
| BSDS/interior16 | context_no_width | 4 | 0.0018450236 | 2.2784379e-05 | 2.6457688e-05 | 0.00347954 |
| BSDS/interior16 | ratio_relaxed | 4 | 0.0018233201 | 0 | 0 | 0.002393065 |
| BSDS/patch32 | baseline | 4 | 0.0012196711 | 0 | 0 | 0.0002987789 |
| BSDS/patch32 | context_full | 4 | 0.0012196711 | 0 | 0 | 0.0002987789 |
| BSDS/patch32 | context_no_width | 4 | 0.0012196711 | 2.331734e-05 | 0 | 0.001485726 |
| BSDS/patch32 | ratio_relaxed | 4 | 0.0012196711 | 0 | 0 | 0.0002987789 |

| 方法 | 局部输入 | 严格净改善 | 新恢复旧拒绝 | 旧成功退步/丢失 | 新增干净/相邻损伤 |
|---|---:|---:|---:|---:|---:|
| ratio_relaxed | 20 | 2 | 2 | 0/0 | 0/0 |
| context_no_width | 20 | 2 | 2 | 0/0 | 5/4 |
| context_full | 20 | 2 | 2 | 0/0 | 0/0 |

## 边界

69022及上一轮所有图均为已用诊断/回归材料，不作为未参与设计的确认集。新4图在冻结后检查一次；控制例包括合法紧条纹的不可辨识反例，不能从零自然图误改推广为普遍纹理安全。

上下文只用于邻带对比，严格资格失败行始终不处理。因此可能仍残留条纹，证据不足时继续拒绝。宽背景或频率偏差造成的半区不一致未通过降低门槛掩盖。
