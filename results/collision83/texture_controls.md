# 局部无噪区与真实同频纹理的附加检验

沿用开发照片及旧横向渐变幅值 0.080 的相位。分别将第 26–122 或 188–255 列定义为测试条带，边界外用 8 列余弦过渡。测试包括四种情形：仅在条带外加干扰；条带内有额外定频织纹且干扰仅在外部；全图加干扰并在条带内叠加真实定频织纹；真实织纹与干扰在局部完全一致的不可辨对照。定频织纹振幅 0.035，以最近推算节点为频率，相位采用旧干扰起始相位加 π/2。表内误差只统计条带内、不含边界过渡。

| 条带与条件 | 带内输入 MSE | 带内 MSE 上版 → 本版 | 带内变化像素 上版 → 本版 | 整图去除比例 上版 → 本版 | 整图输出逐值相同 | 本版决定 |
|---|---:|---:|---:|---:|---|---|
| left / clean_gap | 0 | 0 → 0 | 0 → 0 | 0.0000 → 0.0000 | 是 | recurring_narrow_texture |
| left / stationary_weave_gap | 0 | 0 → 0 | 0 → 0 | 0.0000 → 0.0000 | 是 | recurring_narrow_texture |
| left / stationary_weave_plus_stripe | 0.00317628 | 0.0022799 → 4.05662e-05 | 11798 → 24832 | 0.4348 → 0.9558 | 否 | all_inferred_bands_verified |
| left / identical_observation | 0 | 0.000515836 → 0.00298464 | 11503 → 24832 | 0.6045 → 0.9811 | 否 | all_inferred_bands_verified |
| right / clean_gap | 0 | 0 → 0 | 0 → 0 | 0.0000 → 0.0000 | 是 | recurring_narrow_texture |
| right / stationary_weave_gap | 0 | 0 → 0 | 0 → 0 | 0.0000 → 0.0000 | 是 | recurring_narrow_texture |
| right / stationary_weave_plus_stripe | 0.00319075 | 0.0025567 → 1.19697e-05 | 6006 → 17408 | 0.4265 → 0.9749 | 否 | all_inferred_bands_verified |
| right / identical_observation | 0 | 0.000338068 → 0.00298672 | 5558 → 17408 | 0.5757 → 0.9722 | 否 | all_inferred_bands_verified |

再用带合法静止织纹的恢复图减去同一干扰条件下未加入该织纹时的恢复图，把差值投影到已知合法织纹新增量上。比例 1 表示保留该方向的全部变化。下表另列差值与真实织纹新增量的带内 MSE，以计入非平行误差。这可以区分总残留变少和真实织纹的保留。

| 条带与条件 | 合法织纹增量投影比例 上版 → 本版 | 增量误差 MSE 上版 → 本版 |
|---|---:|---:|
| left / stationary_weave_gap | 1.0000 → 1.0000 | 0 → 0 |
| left / stationary_weave_plus_stripe | 0.9309 → 0.9590 | 2.83606e-05 → 6.61117e-06 |
| right / stationary_weave_gap | 1.0000 → 1.0000 | 0 → 0 |
| right / stationary_weave_plus_stripe | 0.9870 → 0.9891 | 1.68723e-05 → 8.53031e-06 |

最后一种情形逐像素沿用原高幅观测，只更换清晰参考，将条带内同一相位、幅度、空间变化的波形视作本来就有的内容。任何单图确定性算法都会给同一输入输出相同恢复图。它测量单图无法区分两种成因时可能发生的局部损伤。前三种的观测确实重新生成，可分别检验拒绝路径和有真实同频织纹时的结果。

仅表示这张照片及本次局部构造。干净条带输出有变化为误改；两版输出相同只表明没有新增误改。[完整像素误差、外侧条纹减除比例及决策](texture_controls.json)

复现命令为 `python check_collision_textures.py --out results/collision83_reproduction`。
