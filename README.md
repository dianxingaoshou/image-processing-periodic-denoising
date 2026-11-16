# 基于多尺度局部相位建模与保守决策的非平稳周期干扰抑制

本项目实现算法与可复现实验，不包含训练模型、图形界面或论文。算法只接收含噪灰度图；干净原图、合成频率和区域掩膜只由评价代码使用。

GitHub维护入口：[克隆、数据重建、测试与提交说明](MAINTENANCE.md)。仓库跟踪源码、配置、固定清单和数值证据；大数据与逐图中间产物在本地重建。代表性图表见 [docs/figures](docs/figures)，第三方条件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## v3 数值修复与安全权重解耦

v3保留全部v1/v2冻结代码和结果，新增 `denoiser_v3.py` 作为入口。数值修复包括全像素拟合、按实际分析窗口去重和支撑计数；共享浮点灰度转换消除了单图入口与实验的转换差异。缺少配置会报错，只有显式 `--use-default-config` 才运行未校准默认参数。

保守决策分为拟合权重q和安全权重s。旧组合为 `min(alpha*q*s,1)`，新组合为 `min(alpha*q,1)*s`；候选不合格时直接为0。因此增大强度不会取消安全上限。评分公式保持原样，耦合P0对照和解耦版复用完全相同的候选残差，便于判断组合顺序的作用。每次运行记录有风险但达到满额减除的块数。

```powershell
# 数值修复，保留已测量的耦合行为
.\.venv\Scripts\python.exe filter_image_v3.py "你的图片.png" --config configs/v3_p0_coupled.json --out results/own_image_v3
# 单独复现解耦权重的研究配置
.\.venv\Scripts\python.exe filter_image_v3.py "你的图片.png" --config configs/v3_frozen.json --out results/own_image_v3_decoupled
# 明确选择未经校准的默认参数
.\.venv\Scripts\python.exe filter_image_v3.py "你的图片.png" --use-default-config
# 重建固定输入并重新执行校准、主实验、模型外验证与核验
.\run_v3.ps1
```

开发集校准同时约束平均改动、最差20%图像的平均改动和局部干净区误差。新留出的60张官方test图用于本轮审计，之前两批测试图保留为回归记录。另使用三次相位、平滑随机相位、四角、窄边带、不规则区域，固定12张DTD自然纹理的未另加干扰及叠加干扰配对，以及2个无配对参考实例。

所有频率误差均在合成块中心评价；候选缺失和拒绝也保留在覆盖率分母中，不把重叠平均频率显示图当作逐像素真值预测。DTD子集是定向压力检查，不代表全DTD；自然照片也不保证物理上完全无噪声。真实图没有配对参考，不计算PSNR或SSIM。

完整结果见 `results/v3/RESULTS.md`。`configs/v3_frozen.json` 记录数值库分发文件与二进制哈希、生成器、指标、基线和依赖源码、协议及准确审计ID顺序。精确复算需要匹配的Python与依赖环境；改变环境会明确拒绝旧快照，应在项目副本中重新校准并保留新的记录。

本轮取舍见 `results/v3/DECISION.md`。解耦降低了部分尾部改动，但明显牺牲去噪效果，当前作为实验分支保留，不自动替换v2入口。`v3_p0_coupled.json` 是与主实验P0耦合对照一致的使用配置，不是另一次校准得到的新方法。

自然纹理的额外配对诊断见 `results/v3/texture_tradeoff/EXPLORATORY.md`。这部分是明确标记的后验探索，没有用于选择参数：在共同恢复质量区间内，耦合与解耦曲线大多重合，未支持普遍更优的选择性保护；冻结强度下，部分DTD配对输入仍出现净损伤。安全上限的数学修复不等于纹理识别能力已经成立。

前2候选精拟合与重叠残差一致性作为P2实验开关存在，默认关闭。其开发探索收益不稳定，记录在 `results/v3/optional_p2_development.csv`；留出只隔离给定候选后的精细拟合，不能称完全独立交叉验证。当前未具备MATLAB/Octave运行环境，未声称完成ARPENOS原版与移植版的逐输入数值核验。

## v2 优化版入口 冻结对照

优化版保留旧版代码和全部旧结果，新增 `optimized_denoiser.py` 统一入口。`local_estimation_v2.py` 负责多候选尺度验证、连续频率优化、二次相位与变幅残差；`texture_guard_v2.py` 负责允许减除的证据门槛、邻域与跨尺度评分、局部结构风险和全图低维周期歧义拒绝。

高能量频峰只获得候选资格。低谱异常分候选还需要独立尺度及邻域支撑才能减除，避免把灵活拟合本身误当成“识别到噪声”。评分仍是启发式可靠性分数。

```powershell
.\.venv\Scripts\python.exe filter_image_v2.py "你的图片.png" --out results/own_image_v2
# 完整重建数据、开发集校准、独立审计、模块消融及核验
.\run_optimization.ps1
```

优化参数只在原30张开发图确定。旧60张测试图保留为回归记录；另从未使用的官方test图像中按ID随机冻结60张审计图。完整优化结果见 `results/optimization_v2/RESULTS.md`，划分和实验规模见 `optimization_protocol.json`。模块消融固定使用审计清单前20图，匹配去噪曲线固定使用前10图，均不用于选择参数。

变幅与二次相位模型能增加表达能力，也会增加误拟合风险。全图单频、多频和渐变纹理的保守拒绝，不能解释为能识别自然图中的所有局部真实纹理；补充验证保留了同像素、不同真实解释的反例。

## 已实现的内容

- `local_spectral.py`：64、128、256 像素多尺度局部频谱，连续频率估计、尺度选择、局部正弦与背景平面联合拟合。
- 以周期/像素统一频率；邻域与跨尺度比较考虑正负共轭等价。根据峰证据、邻域一致性、跨尺度一致性和近纯周期风险控制减除强度，评分不解释为概率。
- 正端点 Hann 权重与归一化重叠融合，覆盖图像边缘。加窗只用于分析与拟合加权，最终从原始像素减去残差。
- 原型全局固定/自适应陷波、固定128局部基线、完整方法、关闭保守策略、保留跨尺度证据但强制选128的尺度消融。
- `published_baseline.py`：ARPENOS 作者源码指导的 Python 移植，默认 L=128、reg=0。MATLAB robustfit 替换为显式 IRLS/MAD，未核对 MATLAB 逐位一致性，也不声称复现论文数值。
- BSDS500 原图构成的30张开发、60张测试划分；四类退化、三档强度与无噪声对照。所有版本按原图划分，强度变化保留相同频率、相位和空间分布。

## 运行环境与快速使用

Windows PowerShell，建议 Python 3.11 或以上；当前验证环境由 `requirements-lock.txt` 记录。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests original/periodic_notch_starter/test_demo.py
.\.venv\Scripts\python.exe filter_image.py "你的图片.png" --out results/own_image
```

自己的图片会转换为灰度，至少32×32像素。输出恢复 PNG、浮点 NPY、诊断 NPZ 与逐块决策 JSON。没有参考图时不计算 PSNR 或 SSIM。诊断中的频率图单位为周期/像素，像素图是重叠加权显示；实际离散尺度选择见逐块 JSON。

## 完整复现

```powershell
# 首次下载官方原始图像归档，只提取JPEG；已有数据时可加 --offline
.\.venv\Scripts\python.exe data_pipeline.py --root data

# 只在30张开发图的中档退化及无噪声对照上校准少量运行参数
.\.venv\Scripts\python.exe run_experiments.py calibrate

# 冻结配置后，60张测试图 × (4类 × 3强度 + 无噪声) × 8方法
.\.venv\Scripts\python.exe run_experiments.py test
.\.venv\Scripts\python.exe analyze_results.py
.\.venv\Scripts\python.exe validate_tradeoff.py
.\.venv\Scripts\python.exe audit_failures.py
.\.venv\Scripts\python.exe verify_artifacts.py
```

`--limit N` 可用于小规模冒烟检查；使用该选项时请指定另一个 `--out`，避免覆盖正式结果。不要使用测试结果选择参数。重新校准会覆盖 `configs/frozen.json`，需同步重新生成全部结果。

## 实验文件

- `data/manifest.json`：原图ID、官方划分、处理方式及文件哈希；`data/source/provenance.json` 记录数据来源。
- `configs/frozen.json`：开发集校准记录与冻结参数。固定全局带宽在1、2、3格点中选择；局部强度在0.5、0.75、1中选择。目标是各图、各条件等权平均 MSE。
- `results/main/metrics.csv`：每图片、每条件、每强度、每方法的原始指标。`synthesis.jsonl` 保存完整合成参数和种子。
- `results/main/summary.csv`：按条件汇总；`paired_comparisons.csv`：按原图成组自举的配对 PSNR 差异，三档强度先在同图内平均。
- `results/main/validation.json`：行数、唯一性、异常计数与完整性核验；`source_hashes.json`：本次使用的算法源码哈希。
- `results/main/RESULTS.md`：本次实际结果与适用边界；`results/verification.json`：配置、数据、指标、强度扫描及失败重跑的独立核验。
- `results/main/figures/`：平均结果、预先固定前三个测试ID的对比图、频率/尺度/强度/残差诊断。示例未按效果好坏筛选。
- `results/supplement/`：强度曲线、在去噪量重叠范围内的损伤比较、规则纹理与不可辨识反例。

## 评价口径

像素范围为[0,1]；PSNR、SSIM明确 `data_range=1`，不对结果单独拉伸对比度。各方法统一投影到[0,1]，`output_projection_fraction` 记录外层投影像素比例；原型与局部方法已在内部裁剪，因此该字段不含其内部裁剪。指标基于浮点数组，PNG仅用于显示。

含噪区域定义为幅度包络大于0.01；干净区域定义为包络小于1e-6，合成前将这部分包络明确置零。仅 `local` 条件和无噪声对照存在干净区域；其他条件的干净区误差留空。不同条件总噪声能量不同，不能用它们的原始 PSNR 排序来判断算法难度。

`degraded` 表示输出 MSE 比输入高出1e-8，是质量退化统计，不等同于程序异常。无噪声输入还需直接查看 MAE，极小改动不会被阈值统计全部计入。时间是本机单次墙钟时间，重复使用的分析开销计入每个方法；不作为严格速度排名。

## 适用范围与已知边界

每个局部块目前拟合一个主要正弦成分。它适合平滑变化的单主频干扰；多频、突变、强遮挡、真实纹理与噪声同频时仍可能残留或误滤。近纯周期输入的保守拒绝有意保留歧义，因此同样像素被解释为“平滑背景加条纹”时也可能拒绝去噪。这是单图辨识边界，不能据此宣称能识别所有真实纹理。

完整方法与关闭保守策略的结果必须结合相近去噪量下的损伤曲线解释。只有减少改动量，不能证明纹理保护更好。当前实验是合成干扰验证，未覆盖有配对真值的真实传感器噪声。

## 来源和使用条件

原型保存在 `original/periodic_notch_starter/`，旧快照不作为本次实验结果。BSDS500 来源为 [Berkeley 官方数据页面](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)。ARPENOS 来源为 [作者代码页面](https://members.loria.fr/FSur/software/ARPENOS/)；作者代码及本项目相应移植仅供学术研究，见 `third_party/arpenos/` 中的原始条款和移植说明。文献与源码来源在 `literature_sources.json`。
