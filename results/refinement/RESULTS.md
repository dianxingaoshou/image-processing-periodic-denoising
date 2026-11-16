# 风险评分聚焦验证

本轮先修离散像素频率范围、默认配置和数据测试边界，以风险评分为唯一主要算法改动。未增加高阶相位、前2候选重排或空间包络自由度。

三种决策复用完全相同的候选残差：P0修复后耦合基线、解耦机制对照、输入周期歧义与移除结构风险。所有表为强度2；曲线另存每图原始记录。

局部参考增益只在离线评价计算，忽略裁剪与邻块融合，不能当作全图最优解。得分分箱仅诊断排序，非概率校准。

## development

| 条件 | 方法 | 图数 | 平均PSNR dB | MSE | 比输入更差比例 | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| chirp | coupled_baseline | 8 | 30.358 | 0.0012739 | 0.0% | 0.041125 |
| chirp | decoupled_control | 8 | 28.039 | 0.0017403 | 0.0% | 0.043342 |
| chirp | input | 8 | 23.163 | 0.0048299 | 0.0% | 0.062989 |
| chirp | structure_risk | 8 | 31.049 | 0.0011771 | 0.0% | 0.041352 |
| clean | coupled_baseline | 8 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | decoupled_control | 8 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | input | 8 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | structure_risk | 8 | inf | 0.0000000 | 0.0% | 0.000000 |
| edge_strip | coupled_baseline | 8 | 35.804 | 0.0003257 | 0.0% | 0.006831 |
| edge_strip | decoupled_control | 8 | 35.728 | 0.0003270 | 0.0% | 0.006831 |
| edge_strip | input | 8 | 33.868 | 0.0004121 | 0.0% | 0.006831 |
| edge_strip | structure_risk | 8 | 35.743 | 0.0003309 | 0.0% | 0.006831 |
| global | coupled_baseline | 8 | 36.575 | 0.0002639 | 0.0% | 0.014738 |
| global | decoupled_control | 8 | 28.999 | 0.0012838 | 0.0% | 0.033949 |
| global | input | 8 | 21.611 | 0.0069041 | 0.0% | 0.076141 |
| global | structure_risk | 8 | 36.789 | 0.0002290 | 0.0% | 0.012033 |
| irregular | coupled_baseline | 8 | 35.906 | 0.0002803 | 0.0% | 0.006800 |
| irregular | decoupled_control | 8 | 35.679 | 0.0002879 | 0.0% | 0.006800 |
| irregular | input | 8 | 34.690 | 0.0003417 | 0.0% | 0.006800 |
| irregular | structure_risk | 8 | 36.192 | 0.0002707 | 0.0% | 0.006800 |
| local | coupled_baseline | 8 | 34.410 | 0.0003821 | 0.0% | 0.010039 |
| local | decoupled_control | 8 | 34.164 | 0.0003973 | 0.0% | 0.010039 |
| local | input | 8 | 33.543 | 0.0004472 | 0.0% | 0.010039 |
| local | structure_risk | 8 | 34.710 | 0.0003668 | 0.0% | 0.010039 |
| orientation | coupled_baseline | 8 | 25.261 | 0.0031237 | 0.0% | 0.053840 |
| orientation | decoupled_control | 8 | 24.998 | 0.0032794 | 0.0% | 0.055155 |
| orientation | input | 8 | 22.703 | 0.0053681 | 0.0% | 0.066716 |
| orientation | structure_risk | 8 | 25.553 | 0.0029687 | 0.0% | 0.052760 |

局部条件相近恢复质量：11/24 对具有非退化共同区间；输入结构风险相对基线干净区MSE差均值 -5.8177449e-09。

## regression

| 条件 | 方法 | 图数 | 平均PSNR dB | MSE | 比输入更差比例 | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| chirp | coupled_baseline | 12 | 24.707 | 0.0118731 | 41.7% | 0.160800 |
| chirp | decoupled_control | 12 | 23.995 | 0.0084040 | 33.3% | 0.124030 |
| chirp | input | 12 | 23.307 | 0.0047023 | 0.0% | 0.063054 |
| chirp | structure_risk | 12 | 25.321 | 0.0123610 | 41.7% | 0.166816 |
| clean | coupled_baseline | 12 | inf | 0.0097210 | 66.7% | 0.154785 |
| clean | decoupled_control | 12 | inf | 0.0059834 | 66.7% | 0.116107 |
| clean | input | 12 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | structure_risk | 12 | inf | 0.0103801 | 66.7% | 0.161340 |
| edge_strip | coupled_baseline | 4 | 34.546 | 0.0003548 | 0.0% | 0.006482 |
| edge_strip | decoupled_control | 4 | 34.528 | 0.0003561 | 0.0% | 0.006482 |
| edge_strip | input | 4 | 33.936 | 0.0004067 | 0.0% | 0.006931 |
| edge_strip | structure_risk | 4 | 34.423 | 0.0003659 | 25.0% | 0.006527 |
| global | coupled_baseline | 13 | 28.573 | 0.0106591 | 30.8% | 0.164153 |
| global | decoupled_control | 13 | 24.761 | 0.0079128 | 30.8% | 0.128440 |
| global | input | 13 | 21.758 | 0.0067175 | 0.0% | 0.076347 |
| global | structure_risk | 13 | 28.650 | 0.0111254 | 30.8% | 0.170838 |

局部条件相近恢复质量：1/4 对具有非退化共同区间；输入结构风险相对基线干净区MSE差均值 6.74157e-13。

## check

| 条件 | 方法 | 图数 | 平均PSNR dB | MSE | 比输入更差比例 | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| chirp | coupled_baseline | 4 | 26.834 | 0.0022817 | 0.0% | 0.048943 |
| chirp | decoupled_control | 4 | 26.143 | 0.0025474 | 0.0% | 0.051628 |
| chirp | input | 4 | 23.102 | 0.0048968 | 0.0% | 0.063052 |
| chirp | structure_risk | 4 | 27.200 | 0.0021623 | 0.0% | 0.048604 |
| clean | coupled_baseline | 4 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | decoupled_control | 4 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | input | 4 | inf | 0.0000000 | 0.0% | 0.000000 |
| clean | structure_risk | 4 | inf | 0.0000000 | 0.0% | 0.000000 |
| edge_strip | coupled_baseline | 4 | 35.430 | 0.0002968 | 0.0% | 0.005473 |
| edge_strip | decoupled_control | 4 | 35.402 | 0.0002981 | 0.0% | 0.005473 |
| edge_strip | input | 4 | 34.322 | 0.0003697 | 0.0% | 0.005489 |
| edge_strip | structure_risk | 4 | 35.353 | 0.0003031 | 0.0% | 0.005473 |
| global | coupled_baseline | 4 | 36.075 | 0.0015088 | 0.0% | 0.062943 |
| global | decoupled_control | 4 | 27.550 | 0.0023191 | 0.0% | 0.064516 |
| global | input | 4 | 21.554 | 0.0069928 | 0.0% | 0.076344 |
| global | structure_risk | 4 | 35.107 | 0.0015356 | 0.0% | 0.062930 |
| irregular | coupled_baseline | 4 | 36.313 | 0.0002612 | 0.0% | 0.006960 |
| irregular | decoupled_control | 4 | 36.013 | 0.0002693 | 0.0% | 0.006960 |
| irregular | input | 4 | 34.906 | 0.0003287 | 0.0% | 0.006960 |
| irregular | structure_risk | 4 | 36.596 | 0.0002554 | 0.0% | 0.006960 |
| local | coupled_baseline | 4 | 36.807 | 0.0002457 | 0.0% | 0.007673 |
| local | decoupled_control | 4 | 36.003 | 0.0002732 | 0.0% | 0.007673 |
| local | input | 4 | 33.726 | 0.0004267 | 0.0% | 0.009605 |
| local | structure_risk | 4 | 37.711 | 0.0002222 | 0.0% | 0.007673 |
| orientation | coupled_baseline | 4 | 24.591 | 0.0035820 | 0.0% | 0.062763 |
| orientation | decoupled_control | 4 | 24.477 | 0.0036616 | 0.0% | 0.062763 |
| orientation | input | 4 | 22.665 | 0.0054140 | 0.0% | 0.066837 |
| orientation | structure_risk | 4 | 24.731 | 0.0034935 | 0.0% | 0.063347 |

局部条件相近恢复质量：6/12 对具有非退化共同区间；输入结构风险相对基线干净区MSE差均值 -1.0159419e-08。

## 取舍与边界

保留P0耦合版为CLI默认，结构风险通过显式配置启用。算法拟合残差不变，因此候选召回及窄条支撑不足不会由风险评分修复。

开发8图用于诊断，旧审计/DTD用于回归；独立检查只有4张未使用BSDS图，不能推广为全面性能结论。未根据回归或独立检查再调参数。

全量旧v3实验未重跑。旧3900条主审计保持原样，精确源码位于v3-validated-baseline标签；当前修订不得沿用旧源码快照冒充同一实验。
