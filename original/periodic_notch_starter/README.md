# 图像周期噪声抑制：可运行起步实验

## 定位
这是课程选题可行性验证和后续改进的起点，不是已经完成的结课报告，也不是任何论文方法的逐行复现。
已实现：盲频峰检测、固定带宽高斯陷波、自适应双轴带宽高斯陷波、空间滤波基线、PSNR/SSIM/MAE、无噪声对照、规则纹理失败案例。
未实现：经过独立数据验证的纹理保护、局部时变条纹处理、真实场景普适恢复。不能把本实验直接作为新算法优于已有方法的证明。

## 安装与运行（建议单独虚拟环境）
Python 3.10 及以上。当前已在 Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / scikit-image 0.26.0 环境运行；精确环境以 results/environment.json 为准。

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_demo.py
python demo.py --out results
```

快速只跑一张图：
```bash
python demo.py --names camera --out quick_results
```
处理自己的图（默认转换为8位灰度，不生成无依据的PSNR/SSIM）：
```bash
python demo.py --input "my_noisy_image.png" --out my_results
```
部分 scikit-image 样例可能首次使用时需要联网下载缓存；BSDS500 不会被本脚本自动下载。未捆绑完整 BSDS500。

## 数据与实验设置
样例为 scikit-image 的 camera、coins、page、brick、grass；这些是公开样例，而不是本项目采集的数据。
每张图测试四个条件：单频整数频点、单频非整数频点、双频整数频点、不加噪声。另加一个无噪声的纯周期纹理反例。
噪声是在 [0,1] 浮点参考图上叠加正弦，再 clip 至 [0,1]。每个正弦幅度0.12、相位0.7；具体频率见 noise_config.json。
注意频率单位为“整张图像上的周期数”，而非每像素频率。不同尺寸图像的条纹空间周期不同，正式实验应明确选用哪种定义。
所有方法在同一浮点输入上评测。PNG仅作8位可视化；指标来自 arrays.npz 中的浮点数据，不能用PNG重新计算并期待逐位一致。
输入裁剪比例已写入CSV；发生裁剪后严格线性加性噪声模型不再成立，可能产生谐波和残留。
本轮参数经小样例探索确定，所有输出属于探索性实验，不是独立测试集结果。所有图像使用同一套参数。
固定带宽基线只采用 sigma=2，尚未在独立验证集寻优。空间基线为 sigma=1 高斯滤波、3x3中值滤波，不代表各类方法最优性能。

## 算法要点
1. Hann窗只用于检测副本，降低边界对频峰定位的影响；复原始终使用未加窗的复数频谱。
2. 对数幅度谱减去15x15局部中位背景；通过中位数/MAD估计全局稳健阈值，筛选局部极大值。
3. 排除中心8个频率格点的低频区域，最多保留8对峰，避免重复选择共轭峰。
4. 自适应带宽来自原始频谱局部超额功率的二阶矩，分别估计两个方向的宽度；这是待验证的启发式规则。
5. 固定/自适应方案共用同一检测器。对每对峰构造高斯衰减并保持共轭对称，逆变换取实部。
6. 保留DC；不做输出min-max拉伸；没有检测到峰时原样返回。

## 参数调整
- threshold 越低，越容易检出弱峰，也更容易误删真实纹理。CLI支持 --threshold 6.5。
- strength 控制抑制强度，默认1；可测试 --strength 0.8，但单纯减弱滤波不能解决纹理与噪声不可辨识。
- Config中的 sigma_min/sigma_max、bandwidth_scale 控制自适应带宽；应在验证集调参，不可逐张看测试真值调参。
- 保护低频区域意味着该范围内的噪声可能漏检，是明确限制。
- 标准化后的频谱异常分数不是统计校准后的“噪声概率”。

## 输出
每个案例有 comparison.png、reference.png（有真值时）、noisy.png、restored.png、notch_mask.png、spectrum_before.png、spectrum_after.png、peaks.json、arrays.npz。
总表 metrics.csv 包括图像名、条件、裁剪比例、方法、单次运行耗时、峰对数、PSNR、SSIM、MAE。
时间不包含文件读写；频域方法计入共享检测阶段实测耗时，不是严格性能基准。
本下载包的 snapshot 仅保留部分预览和完整指标表，不包含约130MB的全部浮点中间结果。运行脚本即可重建。

## 关键限制与下一步
- 对整数频点的理想正弦可能非常有效，不能据此推广到任意真实条纹或摩尔纹。
- 单图像中，真实周期纹理与附加周期干扰可能具有相同频谱；本原型会误删纯周期纹理，snapshot含失败图。
- 下一步优先：BSDS500独立划分，多个随机相位/非整数频率/强度，调优固定基线，自适应带宽消融，无噪声纹理保护测试。
- 真实图像没有严格配对的干净参考时，不报告PSNR/SSIM；展示局部细节和残差，说明失真与适用范围。
- 保留现有方法引用，明确自己的新增代码与实验；不要宣称“首次提出自适应陷波”。

## 参考来源
- OpenCV官方示例（基础频域陷波，非本原型代码来源）：https://docs.opencv.org/4.x/d2/d0b/tutorial_periodic_noise_removing_filter.html
- scikit-image数据：https://scikit-image.org/docs/stable/api/skimage.data.html
- scikit-image评价指标：https://scikit-image.org/docs/stable/api/skimage.metrics.html
- SciPy频谱与窗函数：https://docs.scipy.org/doc/scipy/tutorial/signal.html
- BSDS500官方页：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
- BSDS500官方压缩包：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
- 相关工作：Varghese et al., Adaptive Gaussian notch filter for removing periodic noise from digital images, IET Image Processing 14(8), 1529–1538, 2020. https://doi.org/10.1049/iet-ipr.2018.5707
