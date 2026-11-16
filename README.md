# 基于多尺度局部相位建模与保守决策的非平稳周期干扰抑制

[![Unit tests](https://github.com/dianxingaoshou/image-processing-periodic-denoising/actions/workflows/tests.yml/badge.svg)](https://github.com/dianxingaoshou/image-processing-periodic-denoising/actions/workflows/tests.yml)

一个图像处理课程研究项目，研究如何去除**只出现在局部、频率或方向随位置变化的周期条纹**，同时减少对干净区域和真实规则纹理的误改。

项目包含可运行的灰度图去噪算法、可控退化生成器、对照方法、模块消融及可复现实验。算法以局部周期残差拟合和加权减除为主，无需训练神经网络；少量运行参数在开发集校准。干净参考图、真实频率与干扰掩膜只用于校准和评价，不传给去噪器。

**当前结论：**新窄条专用通道在4张新图的8／16像素边缘条件中，平均含噪区MSE相对输入降低约67%／63%；本批局部输入及12张DTD对照未出现新增干净区改动。仍有拒绝处理和不适用条件，不能推广为任意纹理安全保证。该通道使用独立入口，原通用方法保持不变。

快速入口：[运行自己的图片](#快速运行自己的图片) · [复现实验](#测试与实验复现) · [实验结果](#实验结果) · [目前的不足](#目前的不足) · [维护说明](MAINTENANCE.md)

## 项目做了什么

```mermaid
flowchart LR
    A[含噪灰度图] --> B[重叠分块与多尺度频谱]
    B --> C[候选频率与分析尺度选择]
    C --> D[局部相位和幅度拟合]
    D --> E[可靠性与纹理风险决策]
    E --> F[加权减除与重叠融合]
    F --> G[恢复图与诊断记录]
```

- **多尺度分析：**默认使用64、128、256像素分析窗口，可增加32窗口。32候选须满足至少3个径向周期与0.8分层拟合稳定性，否则继续使用大窗口候选。频率统一为周期／像素，考虑正负共轭等价；默认每尺度最多3个初筛候选，精拟合一个。
- **非平稳残差建模：**实现连续频率细化、二次相位和非负变幅模型，表示条纹频率、方向及幅度的局部变化。拟合改善与复杂度约束共同控制模型选择。
- **保守决策：**结合频峰证据、邻域与不同分析窗口的一致性、残差外推及局部结构风险；对近纯周期歧义输入允许不处理。分数是启发式可靠性，不是噪声概率。
- **空间包络与完整复原：**可在固定相位下，用3×3双线性非负控制网格替代仿射包络，并与原模型比较惩罚后拟合收益。复原块可选择64／步长32或32／步长16，保持50%重叠；正端点窗口及归一化融合覆盖边缘。
- **系统实验：**实现全局陷波、固定尺度、历史版本及ARPENOS Python移植对照，记录质量指标、区域损伤、运行时间、失败案例、频率误差和处理覆盖率。
- **窄条专用通道：**沿长方向提出1D频率，按法向逐行拟合正弦／余弦系数；联合检查两半系数一致性、交叉拟合收益、连续支撑宽度和相邻带对比。估计支撑外修正严格为零，不在其他区域附加旧方法的频率残差。

### 前一轮风险评分修订

| 要求 | 当前实现与验收 |
|---|---|
| 四角合格不能保证内部频率合格 | 检查每个离散像素中心的瞬时频率；内部过零、接近下限、合法低频／变频及上界均有回归用例。拒绝时残差归零，预测与 `final_sse` 同步回退背景模型。未声称约束整个连续矩形域。 |
| 文档推荐与CLI默认一致 | 默认加载P0耦合配置；解耦和新风险均显式选择。缺配置直接报错。 |
| 区分离线测试与数据验收 | `requires_bsds` 标记区分集成测试；缺参考数组时明确跳过，`--require-bsds` 则失败。CI按标记选择，不依赖测试名称。 |
| 检查安全评分是否对应实际收益 | 新增逐候选参考增益、误减除／过度保护、得分分箱及尾部损伤诊断；真实参考仅进入评价模块。 |
| 风险依据由剩余复杂度转向被移除结构 | 新增输入局部周期歧义、谐波支撑和减除前后结构损失；保持候选残差、资格及拟合权重 `q` 不变，便于归因。 |

该轮只修改风险评分。本轮继续实现**局部支撑增强**，分别控制32分析窗口、32复原块和非负包络，保护模块固定为原风险耦合版，不把两个研究方向混在同一对照中。

### 上一轮空间支撑修订

| 开关 | 独立改变什么 | 约束 |
|---|---|---|
| `a32` | 分析尺度加入32，其余几何不变 | 至少3周期；固定分层、奇偶平衡的互补拟合系数稳定性≥0.8；不满足则保留大窗口 |
| `s32` | 复原块64→32，步长32→16 | 保持50%重叠，边缘仍完整覆盖；分析窗口不随之自动改变 |
| `grid` | 固定相位后，增加3×3非负包络备选 | 平滑与幅度正则，控制点上限为常幅估计2.5倍；惩罚后收益不足时保留原幅度模型 |

包络与背景平面通过有界线性最小二乘联合求解；没有向算法提供真实干扰掩膜。实现依据见 [SciPy `lsq_linear` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)。小窗口稳定性和网格拟合改善都不能证明候选一定是噪声。

### 本轮先定位，再修改

诊断只使用已用开发图和旧失败案例，新增检查图未参与定位。统计所有与含噪区域相交的块，避免块中心漏掉8像素条纹。

| 环节 | 确认的问题 | 针对性修改 |
|---|---|---|
| 候选检测 | 原基线8像素正确候选0/56；32＋网格组合在门控前仅4/120、门控后3/120。窄条被边缘窗衰减，短法向支撑也难形成紧凑2D峰。 | 沿长切向聚合最强少量剖面，提出至多5个1D候选，再用相同流程细化频率；不对整图法向平均而稀释窄条。 |
| 门控 | 16像素正确候选存在被拒绝；按含噪重叠Hann质量加权，22.5%落在资格拒绝阶段。直接放开会使相邻MSE由2.94e−7升至4.19e−5。 | 新通道按局部幅度、两半系数一致性和交叉拟合收益判断；不全局放宽旧阈值。 |
| 包络与外溢 | 已知相位对照显示常幅稀释和3×3间距限制窄条；抬高幅度上限也增加外泄。 | 按输入估计逐行系数与连续支撑，不沿网格外推；支撑外不减除。 |
| 错误归属 | 旧`48017/edge32`相邻损伤来自错误频率，不能全归因于正确条纹包络外溢。 | 新通道只减去自身通过检查的分量，其他区域保持输入；这是专用替代方案，不是给旧残差整体加大权重。 |

完整诊断、同残差反事实和源码一致性记录见 [检测定位](results/narrow_diagnosis/detection_FINDINGS.md)、[门控定位](results/narrow_diagnosis/gate_findings.md)、[包络定位](results/narrow_diagnosis/envelope_FINDINGS.md)。真实频率、已知相位与参考最优增益均仅用于这些离线定位，不进入推理。

频率细化使用 [SciPy有界标量局部优化](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)，不保证找到全局最佳频率；旧候选对照与新候选使用相同细化过程。

## 版本与配置怎么选

| 版本／配置 | 内容 | 当前定位 |
|---|---|---|
| v1：`local_spectral.py` | 多尺度局部正弦估计与保守减除 | 历史基线 |
| v2：`optimized_denoiser.py` | 多候选初筛、连续频率、二次相位、变幅模型与风险评分 | 冻结研究基线，原入口保留 |
| 当前默认：`configs/v3_p0_coupled.json` | 原P0耦合参数，含离散频率范围修复 | 自有图片默认入口；旧表不能直接充当当前修订的全量结果 |
| v3：`configs/v3_frozen.json` | 在同一P0残差上将拟合强度与安全上限解耦 | 实验配置，不作为已证实更优的v2替代 |
| 风险实验：`configs/refined_risk.json` | 相同估计器与耦合增益，仅替换纹理风险评分 | 固定粗参数、未校准；显式启用，未升级为默认 |
| 支撑实验：`configs/support_*.json` | 32分析、32复原及非负包络的8种因子组合 | 同一保护配置与强度，独立比较各开关及组合 |
| 窄条专用：`configs/narrow_full.json` | 长方向剖面拟合＋局部门控＋紧支撑 | 独立入口 `filter_narrow.py`；不作为任意周期干扰的默认替代 |

v3将拟合权重记为 `q`，安全权重记为 `s`，二者在 `[0,1]` 内；不具减除资格的候选直接置零。两种组合为：

```text
耦合：g = min(alpha * q * s, 1)
解耦：g = min(alpha * q, 1) * s
```

解耦保证最终增益不超过安全权重，强度不能取消安全上限；它不保证安全评分正确识别了真实纹理。耦合与解耦对照使用相同残差和相同强度，便于判断组合顺序的作用。

P0还修复了固定隔点拟合在0.25周期／像素附近的退化，以及小图把相同有效窗口当作多个尺度支撑的问题。不同分析窗口仍有重叠，不视为统计独立样本。完整取舍见 [v3结论](results/v3/DECISION.md)。

新风险分支使用 `r = texture_penalty × L × max(A, H)`：`L` 为平滑梯度能量的正向损失比例，`A` 为输入局部少频带集中度映射，`H` 为候选基频二／三次谐波的输入能量支撑。它不再通过拟合解释率降低纹理风险；这些特征仍可能同时响应合法纹理和真实干扰，属于启发式歧义证据。

## 快速运行自己的图片

已验证环境为 **Windows PowerShell、Python 3.14**。以下命令中，`python` 应指向Python3.14；自有图像处理不需要先下载实验数据。

```powershell
git clone https://github.com/dianxingaoshou/image-processing-periodic-denoising.git
cd image-processing-periodic-denoising
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

# 把input.png替换成实际图片路径
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_p0_coupled.json --out results/own_image_v3
```

输入至少32×32像素。RGB／RGBA输入会按共享规则转换为浮点灰度；RGBA先与明确背景合成，当前输出是灰度恢复图。

| 输出文件 | 内容 |
|---|---|
| `restored.png` | 用于查看的8位恢复图 |
| `restored.npy` | 浮点恢复数组，避免PNG量化影响后续评价 |
| `diagnostics.npz` | 频率、尺度、强度及其他诊断数组 |
| `decisions.json` | 逐块决策、输入与配置来源、SHA256及预处理记录 |

没有参考图时，入口不会生成PSNR或SSIM。诊断频率图是块中心频率的重叠加权显示，不是完整逐像素频率场。

要检查解耦研究配置，请显式指定：

```powershell
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_frozen.json --out results/own_image_v3_decoupled
```

检查新风险评分：

```powershell
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/refined_risk.json --out results/own_image_structure_risk
```

分别检查空间支撑增强：

```powershell
# 只增加32分析窗口
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_a32.json --out results/own_image_a32
# 只改为32复原块／步长16
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_s32.json --out results/own_image_s32
# 只增加3×3非负包络
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_grid.json --out results/own_image_grid
# 三项同时启用
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_a32_s32_grid.json --out results/own_image_support_all
```

处理充分延展的水平／垂直窄条：

```powershell
.\.venv\Scripts\python.exe filter_narrow.py "input.png" --config configs/narrow_full.json --out results/own_image_narrow
```

`narrow_*.json`配置用于这个专用入口。输出同样包含恢复PNG、浮点NPY、诊断NPZ和`decisions.json`，额外记录逐剖面幅度、稳定性、支撑区间及拒绝原因。至少一个切向需达到64像素且观察到3周期；方向变化明显、条纹很短或输入证据不足时允许不处理。切向频率图不能当作完整二维频率场。

**注意配置选择：**省略 `--config` 时默认加载 `configs/v3_p0_coupled.json`。配置文件缺失会报错；只有显式 `--use-default-config` 才允许使用未经校准的程序默认参数，这不等于推荐配置。

## 测试与实验复现

### 1. 不下载数据的数值测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests original/periodic_notch_starter/test_demo.py -m "not requires_bsds"
```

本轮为 **289项通过、2项数据集成测试按标记排除**。GitHub Actions使用这条命令，仅检验数值与接口行为，不代表重新完成全部图像实验。也可直接运行 `pytest -q -rs`，缺少数据时会明确显示跳过原因。

### 2. 重建固定数据并运行完整测试

```powershell
.\.venv\Scripts\python.exe prepare_v3_data.py
.\.venv\Scripts\python.exe -m pytest -q tests original/periodic_notch_starter/test_demo.py --require-bsds
```

首次需要下载官方BSDS500归档及固定图片。脚本按已有清单和哈希重建210张BSDS参考、12张DTD参考及2个无配对实例，不重新选择审计图。本轮完整测试为 **291项通过**，记录在 [测试日志](results/narrow/tests.xml)；支撑轮220项、风险轮154项与历史v3的104项记录保留在对应旧结果目录。

原图和NPY未纳入Git。无数据时跳过表示“未验证”，不是测试通过；完整验收必须使用 `--require-bsds`。

### 3. 复现本轮窄条对照

```powershell
.\run_narrow.ps1
```

复用 [窄条协议](narrow_protocol.json)，依次执行测试、开发、冻结、旧数据回归、新4图检查、报告、绘图和输入／产物核验。默认写入新的 `results/narrow_reproduction/`，拒绝覆盖已有阶段；用 `-Output` 指定新目录。单阶段入口为 `python narrow_experiments.py development --out results/my_narrow_run`，后续依次执行 `freeze`、`regression`、`check`、`report`。

新4图从旧210图、风险轮4图和支撑轮4图之外按文件名及种子预选；输入JPEG、像素哈希和退化参数均保存。不要重新选择检查图或将本轮检查再次当作未参与设计的数据。

### 4. 复现上一轮支撑对照

```powershell
.\run_support.ps1
```

复用 [固定协议](support_protocol.json)，执行完整测试、开发8图对照、源码与参数冻结、旧数据回归、另4张未使用BSDS图片的一次独立检查及报告生成。默认写入新的 `results/support_reproduction/`，已有阶段拒绝覆盖；可用 `-Output results/my_support_run` 指定新目录。

单阶段入口为 `python support_experiments.py development --out results/my_support_run`，后续依次为 `freeze`、`regression`、`check`、`report`，使用相同 `--out`。不要重新运行 `plan` 选择图片；独立图片的文件名和SHA已提交，需保留完整官方归档提取出的 `data/source/images/test/`。

历史完整提交为 `support-ablation-baseline` 标签。本轮未修改其算法源码、配置或原始结果。

### 5. 复现更早风险实验

精确源码对应 `risk-refinement-baseline` 标签，先建立独立目录并在其中安装环境：

```powershell
git worktree add ..\periodic-risk-baseline risk-refinement-baseline
```

在该目录中运行：

```powershell
.\run_refinement.ps1
```

先重建固定输入，再依次执行开发诊断、冻结源码／配置、旧数据回归和一次独立检查，最后生成报告。默认写入新的 `results/refinement_reproduction/`，已有阶段结果会拒绝覆盖。脚本复用仓库中的 [固定协议](refinement_protocol.json)，不重新选择图像。

每阶段也可单独运行：`python refinement_experiments.py development --out results/my_study`；后续 `freeze`、`regression`、`check`、`report` 使用同一 `--out`。独立检查使用官方归档中4张未纳入旧清单的图片，文件名及SHA已固定，需保留 `data/source/images/test/`。

### 6. 复现历史v3实验

历史3900条主审计对应标签 `v3-validated-baseline`。当前源码已修订，旧快照校验失败是预期行为；要复现旧表，先建立该标签的独立工作目录，再安装依赖和重建数据：

```powershell
git worktree add ..\periodic-v3-baseline v3-validated-baseline
```

在该目录中运行：

```powershell
.\run_v3.ps1
```

脚本依次执行数据重建、测试、开发集校准、主审计、模型外与纹理验证、真实案例、强度曲线、边界检查、失败复核及报告生成。**它会重新校准并覆盖所在目录的配置与结果**。本轮未重跑旧3900条主审计，也没有更新旧冻结SHA掩盖源码差异。

冻结配置校验源码、数据生成、指标、基线、协议、审计ID及顺序，以及Python和数值库分发文件／二进制哈希。换环境可能不匹配旧快照；不要改SHA绕过检查，也不要把新环境运行直接标成旧冻结结果的精确复现。详细流程见 [MAINTENANCE.md](MAINTENANCE.md)。

## 实验结果

历史参数仅在30张开发图上选择；风险与支撑扩展均使用固定粗参数，在其中8张图上诊断，未根据审计或DTD调参。同一原图的不同退化版本保持同一划分；历史三档幅度为0.06、0.12、0.18，同图同条件的频率、相位和包络保持配对。像素范围为 `[0,1]`，PSNR、SSIM明确使用 `data_range=1`，不对各结果分别拉伸对比度。

### 本轮：诊断驱动的8／16像素窄条优化

开发8图×7条件、回归12张DTD加4张旧图×4条件、新4图×7条件，共112个输入、2464行结果。所有原图的退化版本保持配对，新图仅在冻结后检查一次；三档减除强度不增加独立样本数。

在独立4图上，下面是含噪区MSE的逐图均值，越低越好。旧32＋网格基线使用强度2，新通道使用强度1；新通道另有强度曲线记录。完整方案在这些局部输入的全部干净区及相邻16像素区域MSE均为0。

| 条件 | 含噪输入 | 旧32＋网格 | 新完整方案 | 相对输入MSE降低 | 严格净改善 |
|---|---:|---:|---:|---:|---:|
| 边缘8像素 | 0.00304411 | 0.00348956 | 0.00099849 | 67.2% | 3/4 |
| 边缘16像素 | 0.00301186 | 0.00351040 | 0.00110023 | 63.5% | 3/4 |
| 边缘32像素 | 0.00289405 | 0.00330367 | 0.00214713 | 25.8% | 1/4 |
| 内部16像素 | 0.00287744 | 0.00287744 | 0.00106492 | 63.0% | 3/4 |
| 小区域32像素 | 0.00125009 | 0.00125009 | 0.00125009 | 0% | 0/4 |

严格净改善要求：含噪区MSE同时低于输入与旧32基线，且全部干净区、相邻干净区均无新增损伤，数值容差`1e-12`。20个局部输入中完整方案10个达标；另1个仅缓解旧基线失败，未算为净去噪成功。

环节消融使用相同的新逐行拟合；旧候选对照也使用相同频率细化，只更换频率来源。新通道的4种方案复用完全相同的1D候选，所有方案保留共同幅度下限0.02：

| 方案 | 严格净改善／20 | 新增干净区损伤／20 | 新增相邻损伤／20 |
|---|---:|---:|---:|
| 旧2D候选＋新门控＋新支撑 | 0 | 0 | 0 |
| 新1D候选，关闭一致性门控与紧支撑 | 0 | 20 | 20 |
| 新候选＋一致性门控 | 4 | 15 | 15 |
| 新候选＋紧支撑 | 4 | 15 | 6 |
| 新候选＋门控＋紧支撑 | **10** | **0** | **0** |

这支持本批数据中三个环节需要配合：提高候选召回不足以确保安全，门控不能替代支撑，支撑也不能替代可靠性判断。12张DTD无新增干扰照片中，完整方案平均及尾部修改MAE均为0；旧64和旧32＋网格平均MAE分别为0.054074、0.053021。这里只能报告这批照片未修改，不能宣称真实周期纹理已被完全识别。

**仍然失败或不适用：**`69022`的8／16像素候选频率已接近真值，但因邻带对比不足被拒绝；小斑块仍未恢复。全局干扰条件中新通道平均MSE为0.00708903，与输入相同，而旧32基线为0.00073582。因此它不是通用去噪器的替代。分半检验只验证周期系数，频率提出仍读取完整输入，不是完全独立交叉验证；局部真实条纹与干扰高度相似时仍可能误删。

见 [完整结果](results/narrow/RESULTS.md)、[取舍与失败分析](results/narrow/DECISION.md)、[逐图指标](results/narrow/check/metrics.csv)、[净改善与失败缓解配对](results/narrow/check/paired_deltas.csv)、[输入与产物核验](results/narrow/verification.json)。算法、配置和新检查ID已冻结，不根据这些结果再调阈值。

![候选、门控与紧支撑的独立消融及净改善验证](docs/figures/narrow-ablation.png)

### 上一轮：32窗口与空间包络

以原耦合方法为基线，固定风险配置与强度，分别打开三个开关及其组合，共8种方法。开发8图×7条件、回归12张DTD原照片加4个旧BSDS边缘案例、另4图×7条件独立检查，共100个输入、2500行质量记录。局部条件为8／16／32像素边缘条纹、16像素内部条纹和32像素小区域，另有全局与无新增干扰对照。

独立检查的20个局部输入，主判据为：**相对同输入基线，含噪区MSE下降，全部干净区和相邻16像素干净区MSE均不增加**。`1e-12`仅作数值并列容差。下表固定强度2，重复条件不是独立照片。

| 方案 | 相对基线联合改善 | 全部干净区损伤增加 | 相邻区损伤增加 | DTD无新增干扰平均MAE | DTD最差20%平均MAE |
|---|---:|---:|---:|---:|---:|
| 基线 | — | — | — | 0.054074 | 0.154785 |
| 只加32分析 `a32` | 0/20 | 5/20 | 2/20 | 0.053907 | 0.154785 |
| 只改32复原 `s32` | 1/20 | 0/20 | 1/20 | 0.052489 | 0.159514 |
| 只加网格包络 `grid` | 0/20 | 0/20 | 0/20 | 0.054083 | 0.154992 |
| 32分析＋32复原 | 2/20 | 5/20 | 3/20 | 0.052432 | 0.159514 |
| 三项同时启用 | 2/20 | 5/20 | 3/20 | 0.053021 | 0.159822 |

**结论：尚未实现稳定的“窄条去除改善且不增加干净区损伤”，默认配置保持不变。** 一个正例是`309040/edge32`：32分析＋32复原将含噪区MSE从0.00298812降至0.00285376，约降低4.5%，干净区未新增改动；加网格后该案例不再增加收益。8／16像素边缘条纹和小区域在本次检查中仍基本没有去除。

不能把“比基线少损伤”全部算成成功去噪：`s32`唯一相对基线联合改善的`48017/interior16`，含噪区MSE虽从0.013712降至0.008989，仍高于输入0.002819，是减轻原有失败。全部干净区平均误差也会掩盖边界问题：`s32`在32像素边缘条件的相邻区MSE从9.94e−8增至2.78e−4，即使远处干净区总体损伤下降，也不能说边缘安全。

单独网格包络在20个局部输入中没有带来额外含噪区收益；DTD平均及尾部误改也略升。32复原在12张DTD上平均MAE下降、尾部损伤上升，说明平均值不能代替失败分析。相近恢复质量比较仅1/20对可匹配，不能建立普遍纹理保护优势。运行时间是本机单次记录，32复原块的块数由49增至225，开销明显增加。

所有8方案、逐图区域差值、无法匹配原因及输入哈希见 [完整结果](results/support/RESULTS.md)、[取舍与失败分析](results/support/DECISION.md)、[独立检查原始记录](results/support/check/metrics.csv)、[配对差值](results/support/check/paired_deltas.csv) 与 [冻结及完整性核验](results/support/verification.json)。块中心覆盖率随复原几何改变，不能直接据此比较检出能力，本轮以真实区域像素误差为主。

![32分析、32复原及非负包络的质量与损伤对照](docs/figures/support-ablation.png)

### 更早：风险评分聚焦验证

开发8图×7条件、回归41个输入、独立4图×7条件，共125个输入。三种决策复用相同候选残差，比较4个固定强度，包含输入对照共1625行记录；强度采样与重叠块不是独立图像样本。

以下独立检查均为幅度0.12、强度2，4张图的PSNR均值。它是有限检查，不是大规模性能排名。

| 条件 | P0修复后耦合基线 | 新风险评分 |
|---|---:|---:|
| 全局定频 | 36.075 | 35.107 |
| 局部定频 | 36.807 | 37.711 |
| 频率渐变 | 26.834 | 27.200 |
| 方向变化 | 24.591 | 24.731 |
| 边缘窄条 | 35.430 | 35.353 |
| 不规则局部区域 | 36.313 | 36.596 |

**主要负面结果：**12张DTD未另加干扰对照的平均修改MAE由基线0.054074增至0.057386，最差20%均值由0.154785增至0.161340；两者均修改8/12张。新风险没有解决严重纹理误改，因此保留实验配置，CLI默认仍为耦合基线。

两个定位案例的风险排序有变化：合法 `banded` 纹理的安全权重中位数由0.930降至0.548，真实全局干扰案例由0.607升至0.792。但在强度2的耦合公式下，两者增益中位数仍为1，说明**评分变化不等于实际保护改善**。开发集与独立检查的整体得分／参考增益相关性也未改善，不能据此宣称评分已有效。

相近局部恢复质量比较中，独立检查只有6/12对具有非退化共同区间，其余明确保留为无法匹配；干净区误差差值很小，未据此声称实质性纹理保护。详细每图误差、退化比例、尾部损伤与安全分箱见 [风险轮报告](results/refinement/RESULTS.md)、[取舍与失败分析](results/refinement/DECISION.md)、[独立检查原始记录](results/refinement/check/metrics.csv) 和 [离线收益分箱](results/refinement/check/gain_calibration.json)。

![风险轮单项消融：独立检查质量与DTD纹理误改](docs/figures/refinement-risk.png)

### v2：局部相位建模的效果

下表为v2审计的同批60张图，数值是三档强度下逐样本PSNR的均值，单位dB，越高越好。

| 条件 | 含噪输入 | v1完整方法 | ARPENOS Python移植 | v2完整方法 |
|---|---:|---:|---:|---:|
| 全局定频 | 22.43 | 32.55 | 33.69 | **36.49** |
| 局部定频 | 34.35 | 34.99 | **39.58** | 36.69 |
| 频率渐变 | 23.97 | 26.02 | 29.66 | **30.28** |
| 方向变化 | 23.54 | 24.44 | 24.41 | **27.53** |

这批结果支持v2对变化条纹的改进，但局部定频仍低于ARPENOS移植版。完整MSE、MAE、SSIM、配对区间和消融见 [v2结果](results/optimization_v2/RESULTS.md) 与 [原始汇总](results/optimization_v2/audit/summary.csv)。

![v2审计的PSNR增益与无新增干扰对照改动](docs/figures/v2-overview.png)

### v3：数值修复与安全权重的代价

历史v3使用另一批60张图，并在同批输入上重跑冻结v2。两张表的测试图不同，不能跨表计算版本收益。它们在当轮作为审计数据，**在本轮复核后已成为回归与诊断材料**，不能再次宣称完全未参与设计。

| 条件 | 冻结v2 | v3 P0耦合对照 | v3解耦 |
|---|---:|---:|---:|
| 全局定频 PSNR dB | 36.70 | 36.50 | 29.89 |
| 局部定频 PSNR dB | 37.19 | 37.14 | 36.40 |
| 频率渐变 PSNR dB | 30.29 | 30.17 | 28.48 |
| 方向变化 PSNR dB | 27.51 | 27.24 | 26.61 |
| 无新增干扰，平均MAE ↓ | 0.00010773 | 0.00010528 | 0.00009009 |
| 无新增干扰，最差20%平均MAE ↓ | 0.00053863 | 0.00052642 | 0.00045046 |

尾部指标是60张图中MAE最高12张的均值。解耦相对P0耦合将尾部改动降低约 **14.4%**，同时全局定频PSNR下降 **6.61 dB**。因此，安全上限机制修复不能直接当作去噪性能升级。

比较相近去噪量时，仅在双方共同区间内对采样曲线插值，并保留无法匹配的案例。局部干扰曲线和DTD配对探索没有支持解耦相对耦合具有普遍更优的选择性保护；相对关闭保护则损伤较低。插值点不是独立样本。

![固定10图局部干扰的实际运行点及共同去噪区间比较](docs/figures/v3-matched-removal.png)

### 验证范围与结果位置

| 实验 | 输入规模 | 方法结果记录 | 回答的问题 |
|---|---|---:|---|
| v3主审计 | 60图×四类×三强度，加60个无新增干扰对照 | 3900 | 正确性修复、耦合／解耦与关闭保护的权衡 |
| 模型外与位置压力测试 | 固定20图，三次相位、随机相位、四角、边缘窄带、不规则区域 | 500 | 偏离二次相位模型、偏置局部干扰时的表现 |
| DTD纹理配对 | 固定12图，未另加干扰及全局／渐变干扰 | 180 | 是否既保留纹理又能去掉新增干扰 |
| 无参考实例 | OpenCV周期干扰图、作者canvas案例 | 仅固定配置定性展示 | 合成模型之外的可见行为 |

v3主数据清单共210张BSDS图：30开发、120过往验证回归、60当轮审计。频率误差在块中心评价；候选覆盖率、实际处理覆盖率与干净区误处理率的分母包含缺失和拒绝的块，这些是块中心比例，不是像素覆盖率。PSNR均值、MSE均值、区域损伤和程序异常分别记录。

- [v3完整结果](results/v3/RESULTS.md)、[原始汇总](results/v3/audit/summary.csv)、[本轮取舍](results/v3/DECISION.md)
- [模型外结果](results/v3/stress/summary.csv)、[DTD结果](results/v3/textures/summary.csv)
- [DTD配对后验探索](results/v3/texture_tradeoff/EXPLORATORY.md)、[完整性与复现核验](results/v3/verification.json)

## 目前的不足

1. **真实纹理仍会误滤。** 在12张定向DTD子集上，冻结解耦配置的全局加噪恢复MSE为0.008479，高于输入0.006679；渐变加噪为0.008404，高于输入0.004702，仍有净损伤。减少未加噪照片的改动量，不足以证明纹理保护成功。
2. **安全评分仍是启发式。** 它不能保证区分真实周期纹理与附加干扰；同样像素可以有不同真实解释。保守拒绝也可能留下平滑背景上的真实条纹。
   风险轮的新评分改善了两个定位案例的分数排序，却没有改善整体收益排序或DTD尾部损伤；耦合公式饱和还可能抵消风险权重的影响。
3. **模型表达与鲁棒性有限。** 每块主要估计一个周期分量，多候选不等于任意多频同时分离。非二次相位、弱局部条纹及突变边界仍可能残留或误估。
4. **数据证据有限。** DTD只有12张定向压力图，不能代表整个数据集；“无新增干扰”不保证照片物理上完全无噪声。真实实例只有2个，没有配对干净参考，不报告参考PSNR或SSIM。
5. **可选重排尚未证明稳定收益。** 前2候选精拟合和邻块重叠残差重排仅有小规模开发诊断，默认关闭。候选提出仍读完整窗口，留出只隔离精细参数拟合，不是完全独立交叉验证。
6. **已发表方法仅完成源码指导移植。** ARPENOS使用IRLS/MAD替代MATLAB `robustfit`，仅包含 `reg=0` 路径；未完成MATLAB／Octave原版逐输入数值核验，不冒称原论文指标复现。
7. **当前是灰度研究原型。** 不提供彩色联合复原；运行时间为本机单次测量，不能据此建立严格速度排名。
8. **局部空间支撑仍受限。** 上一轮通用网格受控制点间距和幅度稀释限制。新专用通道改善了部分8／16像素条纹，但要求长方向近似定频、足够延展；弯曲条纹、短斑块、邻带同样含强周期纹理及证据不足案例仍可能失败或拒绝处理。观测支撑不是语义真值。

## 代码与产物导航

| 路径 | 作用 |
|---|---|
| [denoiser_v3.py](denoiser_v3.py) | 组合估计器与保护模块，提供 `prepare / restore / denoise` |
| [local_estimation_v3.py](local_estimation_v3.py) | 候选、有效窗口、相位／幅度拟合及可选重排 |
| [texture_guard_v3.py](texture_guard_v3.py) | 拟合／安全权重、候选资格与纹理风险 |
| [image_io_v3.py](image_io_v3.py)、[filter_image_v3.py](filter_image_v3.py) | 共享预处理与单图入口 |
| [validation_data_v3.py](validation_data_v3.py)、[prepare_v3_data.py](prepare_v3_data.py) | 固定数据、退化生成与输入重建 |
| [v3_experiments.py](v3_experiments.py)、[v3_evidence.py](v3_evidence.py) | 校准、批量实验、覆盖评价与匹配去噪证据 |
| [reproducibility_v3.py](reproducibility_v3.py)、[v3_verify.py](v3_verify.py) | 源码／依赖冻结及产物核验 |
| [gain_diagnostics.py](gain_diagnostics.py)、[refinement_experiments.py](refinement_experiments.py) | 离线参考增益、风险分箱、聚焦配对实验及独立检查 |
| [small_window.py](small_window.py)、[spatial_envelope.py](spatial_envelope.py) | 32窗口资格与固定相位非负包络 |
| [support_data.py](support_data.py)、[support_experiments.py](support_experiments.py) | 窄条退化、8因子组合、区域损伤及配对验证 |
| [run_support.ps1](run_support.ps1)、[verify_support.py](verify_support.py) | 一键复现、输入重建及产物完整性核验 |
| [narrow_profiles.py](narrow_profiles.py)、[narrow_support.py](narrow_support.py)、[narrow_denoiser.py](narrow_denoiser.py) | 专用切向候选、局部门控、紧支撑减除 |
| [filter_narrow.py](filter_narrow.py)、[narrow_experiments.py](narrow_experiments.py)、[run_narrow.ps1](run_narrow.ps1) | 窄条入口、环节消融与批量复现 |
| [tests/](tests/)、[configs/](configs/) | 数值边界测试与已测配置 |
| [results/](results/)、[data/v3_manifest.json](data/v3_manifest.json) | 原始CSV、报告、核验记录及数据清单 |
| [original/](original/)、[third_party/](third_party/) | 起步原型、外部源码与来源说明 |

Git保存源码、测试、固定清单和数值证据；原图、运行环境、浮点数组及密集逐图诊断在本地重建。仓库中的代表图位于 [docs/figures/](docs/figures/)。修改、分支提交和完整复现注意事项见 [MAINTENANCE.md](MAINTENANCE.md)。

## 数据来源与使用条件

BSDS500、DTD、OpenCV和ARPENOS的来源、引用及哈希保存在数据清单与各自provenance文件中。ARPENOS作者代码及相应移植限学术研究使用；仓库公开不改变第三方材料的许可条件。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[文献来源](literature_sources.json) 和 [ARPENOS移植差异](third_party/arpenos/PORT_NOTES.txt)。
