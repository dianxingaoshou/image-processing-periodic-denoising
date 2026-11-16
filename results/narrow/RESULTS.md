# 8/16像素窄条：诊断后定向修改

新通道沿条带方向提出1D周期候选，按法向逐行拟合系数，以分半一致性和紧支撑约束决定减除。它不在其他区域运行旧频率残差，是显式专用分支，不能冒称通用方法的单参数升级。

旧候选＋新支撑控制使用相同的新拟合／门控／支撑，仅将频率来源换回旧2D池前5个不同切向频率；其他新方案复用同一组1D候选。真实参考与掩膜只进入评价。

## development

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS/clean | compact_only | 8 | — | 0.00014976153 | — | 0.006586596 |
| BSDS/clean | full | 8 | — | 0 | — | 0 |
| BSDS/clean | legacy32_grid | 8 | — | 3.8145302e-05 | — | 0.002076715 |
| BSDS/clean | legacy64 | 8 | — | 0 | — | 0 |
| BSDS/clean | old_peaks_new_support | 8 | — | 0 | — | 0 |
| BSDS/clean | profile_gate | 8 | — | 9.6910333e-05 | — | 0.001964394 |
| BSDS/clean | proposal_only | 8 | — | 0.0060290089 | — | 0.08742182 |
| BSDS/edge16 | compact_only | 8 | 0.00054016128 | 0.00012510859 | 0.00018211271 | 0.006916283 |
| BSDS/edge16 | full | 8 | 0.0017441889 | 0 | 0 | 0.00253656 |
| BSDS/edge16 | legacy32_grid | 8 | 0.0028588406 | 4.0707912e-05 | 2.9384585e-07 | 0.004428241 |
| BSDS/edge16 | legacy64 | 8 | 0.0029226589 | 0 | 0 | 0.00253656 |
| BSDS/edge16 | old_peaks_new_support | 8 | 0.002624736 | 0 | 0 | 0.00253656 |
| BSDS/edge16 | profile_gate | 8 | 0.0012091419 | 0.00010815382 | 5.9361568e-05 | 0.003438176 |
| BSDS/edge16 | proposal_only | 8 | 0.0052315462 | 0.0060496149 | 0.0048702021 | 0.08731107 |
| BSDS/edge32 | compact_only | 8 | 0.00067564227 | 0.0001211242 | 9.1700679e-05 | 0.007169646 |
| BSDS/edge32 | full | 8 | 0.0018967434 | 0 | 0 | 0.004922092 |
| BSDS/edge32 | legacy32_grid | 8 | 0.0022115201 | 4.3595201e-05 | 7.9800016e-09 | 0.006812512 |
| BSDS/edge32 | legacy64 | 8 | 0.0025990336 | 1.6669388e-07 | 2.3337143e-06 | 0.004922092 |
| BSDS/edge32 | old_peaks_new_support | 8 | 0.0018967434 | 0 | 0 | 0.004922092 |
| BSDS/edge32 | profile_gate | 8 | 0.0012236059 | 0.00010960521 | 0.00022111583 | 0.005367655 |
| BSDS/edge32 | proposal_only | 8 | 0.005219892 | 0.0061580982 | 0.0042432513 | 0.08697179 |
| BSDS/edge8 | compact_only | 8 | 0.00070848097 | 0.00014778014 | 0.00034657435 | 0.008261655 |
| BSDS/edge8 | full | 8 | 0.0017190189 | 0 | 0 | 0.001343205 |
| BSDS/edge8 | legacy32_grid | 8 | 0.0029443381 | 3.9375796e-05 | 0 | 0.003356393 |
| BSDS/edge8 | legacy64 | 8 | 0.0029443381 | 0 | 0 | 0.001343205 |
| BSDS/edge8 | old_peaks_new_support | 8 | 0.0026472659 | 0 | 0 | 0.001343205 |
| BSDS/edge8 | profile_gate | 8 | 0.001366302 | 9.8887751e-05 | 5.8143862e-05 | 0.002617162 |
| BSDS/edge8 | proposal_only | 8 | 0.0052772241 | 0.0060197398 | 0.0050026815 | 0.08731829 |
| BSDS/global | compact_only | 8 | 0.0070125954 | — | — | 0.07676947 |
| BSDS/global | full | 8 | 0.006904087 | — | — | 0.07614058 |
| BSDS/global | legacy32_grid | 8 | 0.00084079826 | — | — | 0.02628986 |
| BSDS/global | legacy64 | 8 | 0.00026385911 | — | — | 0.01473794 |
| BSDS/global | old_peaks_new_support | 8 | 0.006904087 | — | — | 0.07614058 |
| BSDS/global | profile_gate | 8 | 0.0020831553 | — | — | 0.04387226 |
| BSDS/global | proposal_only | 8 | 0.0049999078 | — | — | 0.07539136 |
| BSDS/interior16 | compact_only | 8 | 0.0020553043 | 0.00015482982 | 0.00016930842 | 0.007310479 |
| BSDS/interior16 | full | 8 | 0.0020729268 | 0 | 0 | 0.00238609 |
| BSDS/interior16 | legacy32_grid | 8 | 0.002787257 | 3.8887731e-05 | 1.3225561e-06 | 0.004208031 |
| BSDS/interior16 | legacy64 | 8 | 0.0027946774 | 0 | 0 | 0.00238609 |
| BSDS/interior16 | old_peaks_new_support | 8 | 0.002564867 | 0 | 0 | 0.00238609 |
| BSDS/interior16 | profile_gate | 8 | 0.0017344786 | 0.00010017063 | 0.00014109325 | 0.003474813 |
| BSDS/interior16 | proposal_only | 8 | 0.0067433039 | 0.0059983188 | 0.0066453592 | 0.08756081 |
| BSDS/patch32 | compact_only | 8 | 0.0013461952 | 0.00014997683 | 3.7342237e-05 | 0.006827069 |
| BSDS/patch32 | full | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |
| BSDS/patch32 | legacy32_grid | 8 | 0.0012180134 | 3.8707026e-05 | 0 | 0.002344442 |
| BSDS/patch32 | legacy64 | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |
| BSDS/patch32 | old_peaks_new_support | 8 | 0.0012180134 | 0 | 0 | 0.0002987766 |
| BSDS/patch32 | profile_gate | 8 | 0.0012562786 | 0.00010108117 | 0.00023219043 | 0.002263168 |
| BSDS/patch32 | proposal_only | 8 | 0.0067768916 | 0.0060267846 | 0.0072746806 | 0.08744494 |

| 方法 | 局部输入 | 相对旧32基线联合改善 | 比输入与基线净改善且两类干净区零新增损伤 | 新干净区损伤 | 新相邻区损伤 |
|---|---:|---:|---:|---:|---:|
| old_peaks_new_support | 40 | 7 | 6 | 0 | 0 |
| proposal_only | 40 | 0 | 0 | 40 | 40 |
| profile_gate | 40 | 4 | 0 | 40 | 25 |
| compact_only | 40 | 9 | 9 | 25 | 20 |
| full | 40 | 15 | 15 | 0 | 0 |

## regression

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS_regression/clean | compact_only | 4 | — | 0.00015632212 | — | 0.005349559 |
| BSDS_regression/clean | full | 4 | — | 0 | — | 0 |
| BSDS_regression/clean | legacy32_grid | 4 | — | 0.00073475637 | — | 0.015972 |
| BSDS_regression/clean | legacy64 | 4 | — | 0.0026138986 | — | 0.04561437 |
| BSDS_regression/clean | old_peaks_new_support | 4 | — | 0 | — | 0 |
| BSDS_regression/clean | profile_gate | 4 | — | 0.00033075704 | — | 0.003226 |
| BSDS_regression/clean | proposal_only | 4 | — | 0.010663931 | — | 0.1149436 |
| BSDS_regression/edge16 | compact_only | 4 | 0.00094128279 | 0.00016932249 | 0.00010220214 | 0.005808476 |
| BSDS_regression/edge16 | full | 4 | 0.0018889643 | 0 | 0 | 0.002519278 |
| BSDS_regression/edge16 | legacy32_grid | 4 | 0.0030749308 | 0.00078374013 | 8.8769398e-05 | 0.01849128 |
| BSDS_regression/edge16 | legacy64 | 4 | 0.0030749308 | 0.0027881586 | 0 | 0.04813368 |
| BSDS_regression/edge16 | old_peaks_new_support | 4 | 0.0030749308 | 0 | 0 | 0.002533135 |
| BSDS_regression/edge16 | profile_gate | 4 | 0.001490502 | 0.00036190503 | 0.00050297097 | 0.005668736 |
| BSDS_regression/edge16 | proposal_only | 4 | 0.0087926982 | 0.010819566 | 0.0079029647 | 0.1145821 |
| BSDS_regression/edge32 | compact_only | 4 | 0.00089304258 | 0.00017378873 | 9.1549352e-05 | 0.006282342 |
| BSDS_regression/edge32 | full | 4 | 0.0022490108 | 0 | 0 | 0.004914516 |
| BSDS_regression/edge32 | legacy32_grid | 4 | 0.0029218911 | 0.00081369964 | 0.00027826577 | 0.02025999 |
| BSDS_regression/edge32 | legacy64 | 4 | 0.002955479 | 0.0029873128 | 9.9447713e-08 | 0.05051333 |
| BSDS_regression/edge32 | old_peaks_new_support | 4 | 0.0022490108 | 0 | 0 | 0.004914516 |
| BSDS_regression/edge32 | profile_gate | 4 | 0.0014903711 | 0.00035922607 | 0.00069454744 | 0.007954847 |
| BSDS_regression/edge32 | proposal_only | 4 | 0.0087289145 | 0.011321521 | 0.013707649 | 0.1144537 |
| BSDS_regression/edge8 | compact_only | 4 | 0.00090683304 | 0.00015625171 | 0.00010703007 | 0.005608163 |
| BSDS_regression/edge8 | full | 4 | 0.0016707223 | 0 | 0 | 0.00132889 |
| BSDS_regression/edge8 | legacy32_grid | 4 | 0.0031125059 | 0.00075845819 | 1.8509516e-06 | 0.01730089 |
| BSDS_regression/edge8 | legacy64 | 4 | 0.0031125059 | 0.0026982179 | 0 | 0.04694326 |
| BSDS_regression/edge8 | old_peaks_new_support | 4 | 0.0031125059 | 0 | 0 | 0.001341265 |
| BSDS_regression/edge8 | profile_gate | 4 | 0.001291423 | 0.00035066874 | 0.0001892339 | 0.004532908 |
| BSDS_regression/edge8 | proposal_only | 4 | 0.009120465 | 0.010721917 | 0.007273791 | 0.1147009 |
| DTD_regression/clean | compact_only | 12 | — | 0.0001289691 | — | 0.006916554 |
| DTD_regression/clean | full | 12 | — | 0 | — | 0 |
| DTD_regression/clean | legacy32_grid | 12 | — | 0.010234683 | — | 0.1598224 |
| DTD_regression/clean | legacy64 | 12 | — | 0.0097209739 | — | 0.1547851 |
| DTD_regression/clean | old_peaks_new_support | 12 | — | 0 | — | 0 |
| DTD_regression/clean | profile_gate | 12 | — | 0.0072796259 | — | 0.1039089 |
| DTD_regression/clean | proposal_only | 12 | — | 0.014061817 | — | 0.1685163 |

| 方法 | 局部输入 | 相对旧32基线联合改善 | 比输入与基线净改善且两类干净区零新增损伤 | 新干净区损伤 | 新相邻区损伤 |
|---|---:|---:|---:|---:|---:|
| old_peaks_new_support | 12 | 1 | 1 | 0 | 0 |
| proposal_only | 12 | 0 | 0 | 12 | 12 |
| profile_gate | 12 | 2 | 1 | 11 | 10 |
| compact_only | 12 | 1 | 0 | 12 | 6 |
| full | 12 | 5 | 5 | 0 | 0 |

## check

| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |
|---|---|---:|---:|---:|---:|---:|
| BSDS/clean | compact_only | 4 | — | 0.0001327226 | — | 0.008065075 |
| BSDS/clean | full | 4 | — | 0 | — | 0 |
| BSDS/clean | legacy32_grid | 4 | — | 0.00020908828 | — | 0.01065201 |
| BSDS/clean | legacy64 | 4 | — | 0.00037675644 | — | 0.01793548 |
| BSDS/clean | old_peaks_new_support | 4 | — | 0 | — | 0 |
| BSDS/clean | profile_gate | 4 | — | 7.9037209e-05 | — | 0.002102491 |
| BSDS/clean | proposal_only | 4 | — | 0.0024816908 | — | 0.04674672 |
| BSDS/edge16 | compact_only | 4 | 0.00016354343 | 0.00013212422 | 5.3586405e-05 | 0.008116325 |
| BSDS/edge16 | full | 4 | 0.0011002284 | 0 | 0 | 0.002444041 |
| BSDS/edge16 | legacy32_grid | 4 | 0.003510402 | 0.00018868384 | 0.00064465632 | 0.01236862 |
| BSDS/edge16 | legacy64 | 4 | 0.0041560233 | 0.0003258425 | 0.000864312 | 0.01927787 |
| BSDS/edge16 | old_peaks_new_support | 4 | 0.0030118587 | 0 | 0 | 0.002536475 |
| BSDS/edge16 | profile_gate | 4 | 0.00078598138 | 6.330787e-05 | 1.3784484e-05 | 0.002383852 |
| BSDS/edge16 | proposal_only | 4 | 0.0021933001 | 0.0024686499 | 0.0029093508 | 0.0454309 |
| BSDS/edge32 | compact_only | 4 | 0.00019341458 | 0.00013773636 | 5.2280467e-05 | 0.008586079 |
| BSDS/edge32 | full | 4 | 0.0021471333 | 0 | 0 | 0.004923769 |
| BSDS/edge32 | legacy32_grid | 4 | 0.0033036737 | 0.00015612433 | 0.0002256359 | 0.01391567 |
| BSDS/edge32 | legacy64 | 4 | 0.0039104669 | 0.00028738021 | 0.00046141668 | 0.02055074 |
| BSDS/edge32 | old_peaks_new_support | 4 | 0.0028940542 | 0 | 0 | 0.004923769 |
| BSDS/edge32 | profile_gate | 4 | 0.000757559 | 6.7755097e-05 | 1.4015707e-05 | 0.003560609 |
| BSDS/edge32 | proposal_only | 4 | 0.0025118633 | 0.0019922148 | 0.0017236175 | 0.04483301 |
| BSDS/edge8 | compact_only | 4 | 0.00014532042 | 0.00012898248 | 5.7422868e-05 | 0.007859117 |
| BSDS/edge8 | full | 4 | 0.00099848974 | 0 | 0 | 0.001294563 |
| BSDS/edge8 | legacy32_grid | 4 | 0.0034895581 | 0.0002014494 | 0.00062612005 | 0.01159058 |
| BSDS/edge8 | legacy64 | 4 | 0.0042350672 | 0.00035008958 | 0.0010034267 | 0.01864432 |
| BSDS/edge8 | old_peaks_new_support | 4 | 0.0030441114 | 0 | 0 | 0.001342844 |
| BSDS/edge8 | profile_gate | 4 | 0.00051879945 | 7.1349219e-05 | 0.00017585078 | 0.002248816 |
| BSDS/edge8 | proposal_only | 4 | 0.0023167532 | 0.0024646065 | 0.0028265833 | 0.04604017 |
| BSDS/global | compact_only | 4 | 0.007147705 | — | — | 0.07640303 |
| BSDS/global | full | 4 | 0.0070890312 | — | — | 0.07602721 |
| BSDS/global | legacy32_grid | 4 | 0.00073581723 | — | — | 0.03323075 |
| BSDS/global | legacy64 | 4 | 0.00015740702 | — | — | 0.01368845 |
| BSDS/global | old_peaks_new_support | 4 | 0.0070890312 | — | — | 0.07602721 |
| BSDS/global | profile_gate | 4 | 0.0011683266 | — | — | 0.02863416 |
| BSDS/global | proposal_only | 4 | 0.00067674491 | — | — | 0.02206663 |
| BSDS/interior16 | compact_only | 4 | 0.00024079418 | 0.00018622334 | 9.1948732e-05 | 0.01024892 |
| BSDS/interior16 | full | 4 | 0.0010649229 | 0 | 0 | 0.002388557 |
| BSDS/interior16 | legacy32_grid | 4 | 0.0028774432 | 0.00022170457 | 0 | 0.01286976 |
| BSDS/interior16 | legacy64 | 4 | 0.0028774432 | 0.00040187353 | 0 | 0.02032403 |
| BSDS/interior16 | old_peaks_new_support | 4 | 0.0028774432 | 0 | 0 | 0.002393034 |
| BSDS/interior16 | profile_gate | 4 | 0.00070893375 | 7.7818403e-05 | 1.1913551e-05 | 0.002939354 |
| BSDS/interior16 | proposal_only | 4 | 0.0012824625 | 0.0025339255 | 0.0022799249 | 0.04606347 |
| BSDS/patch32 | compact_only | 4 | 0.0012500873 | 0.00013482722 | 1.6783405e-06 | 0.008363721 |
| BSDS/patch32 | full | 4 | 0.0012500873 | 0 | 0 | 0.0002987871 |
| BSDS/patch32 | legacy32_grid | 4 | 0.0012500873 | 0.00021048655 | 3.4717248e-07 | 0.0109508 |
| BSDS/patch32 | legacy64 | 4 | 0.0012500873 | 0.00038273075 | 7.10973e-10 | 0.01823426 |
| BSDS/patch32 | old_peaks_new_support | 4 | 0.0012500873 | 0 | 0 | 0.0002987871 |
| BSDS/patch32 | profile_gate | 4 | 0.0015301361 | 7.3520749e-05 | 0.00010732376 | 0.00234582 |
| BSDS/patch32 | proposal_only | 4 | 0.0037900307 | 0.002479935 | 0.0024443332 | 0.04679621 |

| 方法 | 局部输入 | 相对旧32基线联合改善 | 比输入与基线净改善且两类干净区零新增损伤 | 新干净区损伤 | 新相邻区损伤 |
|---|---:|---:|---:|---:|---:|
| old_peaks_new_support | 20 | 3 | 0 | 0 | 0 |
| proposal_only | 20 | 0 | 0 | 20 | 20 |
| profile_gate | 20 | 7 | 4 | 15 | 15 |
| compact_only | 20 | 7 | 4 | 15 | 6 |
| full | 20 | 11 | 10 | 0 | 0 |

## 解释边界

主表：旧基线强度2，新通道强度1；另存固定强度曲线，不视为独立样本。阈值为开发前固定粗参数，不根据DTD或新检查调参。

新通道假设沿水平或垂直条带方向近似定频、延伸足够长。分半检验只隔离周期系数；候选频率来自完整输入，不是完全独立交叉验证。局部真实规则纹理仍可能满足全部条件，硬支撑并不证明语义上一定为噪声。

诊断里的已知相位、真实频率和孤立块参考增益只用于定位，不参与新方法。旧实验保持原样，默认通用入口不变。
