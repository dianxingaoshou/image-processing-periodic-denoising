# 仓库维护与复现

仓库地址：<https://github.com/dianxingaoshou/image-processing-periodic-denoising>。

Git保存所有版本源码、测试、冻结配置、数据清单、生成参数、原始CSV、汇总与核验记录。原图、浮点数组、逐图诊断和压缩交付包留在本地，通过脚本重建。少量代表图放在 `docs/figures/`。现有本地文件没有删除。

## 克隆与安装

```powershell
git clone https://github.com/dianxingaoshou/image-processing-periodic-denoising.git
cd image-processing-periodic-denoising
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

原验证环境使用Python3.14，精确冻结检查还包含Python版本、数值库分发文件与编译扩展哈希。重新安装后即使版本相同，安装产物也可能不同；旧快照不匹配时不要修改哈希绕过校验。在项目副本或开发分支重新校准，保留旧提交与结果以便回溯。

## 先重建输入再跑完整测试

```powershell
.\.venv\Scripts\python.exe prepare_v3_data.py
.\.venv\Scripts\python.exe -m pytest -q tests original/periodic_notch_starter/test_demo.py --require-bsds
```

数据清单会入Git，原图和NPY不会。`requires_bsds`集成测试在缺数据时明确跳过；完整验收加 `--require-bsds`，缺数据则失败。`prepare_v3_data.py`按已冻结的文件名和哈希重建210张BSDS参考、12张DTD参考及2个无配对实例，不重新选择审计图。

GitHub Actions通过 `-m "not requires_bsds"` 执行不依赖下载数据的测试；这不等于重跑完整实验或核验旧环境快照。

## 处理自己的图片

`filter_project.py` 是项目面向单图的统一入口。两份默认配置由 `project_pipeline.load_config()` 从仓库内加载；判断规则及波形一致性门槛在 `project_pipeline.py`。修改这里以后，用两条独立路径和输入做同图回归，检查窄条资格孔、支撑外像素、宽干扰残留和真实纹理误改。

```powershell
.\.venv\Scripts\python.exe filter_project.py "input.png" --out results/own_image_project
.\.venv\Scripts\python.exe project_comparison.py --out results/project_reproduction
.\.venv\Scripts\python.exe plot_project.py --examples results/project_reproduction/examples --out results/project_reproduction/overview.png
```

单图推理不依赖数据集。第二条命令需要先按上节重建数据，并保留官方BSDS归档中 `test/69022.jpg` 的提取件。它将已用开发、回归及固定纹理输入的逐图指标、哈希、处理路径写入新的输出目录，示例数组供第三条命令直接绘图。首次集成对照见 [results/project](results/project/)。历史独立入口供重现分支和各项消融：

```powershell
# P0数值修复，保留已测量的耦合权重行为
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_p0_coupled.json --out results/own_image_v3

# 解耦权重仅作为当前实验分支使用
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_frozen.json --out results/own_image_v3_decoupled

# 新风险评分显式试用；不代表已证明更优
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/refined_risk.json --out results/own_image_structure_risk

# 32分析、32复原与3×3空间包络均可独立试用
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_a32.json --out results/own_image_a32
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_s32.json --out results/own_image_s32
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/support_grid.json --out results/own_image_grid

# 充分延展、近似定频的水平／垂直窄条，用独立入口
.\.venv\Scripts\python.exe filter_narrow.py "input.png" --config configs/narrow_full.json --out results/own_image_narrow

# 本轮上下文修复，保持原资格门槛和严格输出
.\.venv\Scripts\python.exe filter_narrow.py "input.png" --config configs/context_context_full.json --out results/own_image_context
```

当前证据未支持把解耦版视为普遍更优替代。先读 `results/v3/DECISION.md`、`results/v3/RESULTS.md` 和 `results/v3/texture_tradeoff/EXPLORATORY.md`，其中保留性能损失、纹理净损伤和负面结果。

单独运行 `filter_image_v3.py` 默认加载P0耦合参数。32窗口、32复原块及非负网格包络是独立开关，8种组合位于 `configs/support_*.json`；它们不改变风险模块设置。结果见 `results/support/RESULTS.md`。前一轮风险实验见 `results/refinement/RESULTS.md`，源码保存在 `risk-refinement-baseline` 标签；旧审计与DTD已作为回归材料。

## 修改与提交

```powershell
git switch -c improve/my-change
# 修改代码，运行有关测试；涉及冻结科学源码时重新校准并生成新记录
git status
git add 修改的文件
git commit -m "Describe the concrete change"
git push -u origin improve/my-change
```

初始导入提交和 `v3-validated-baseline` 标签用于回到历史v3完整代码。本轮修改了科学源码，旧快照在当前分支不再匹配；保留旧快照及原始表，不修改其SHA冒充精确复现。`.gitattributes`关闭换行归一化，避免Git自动转换换行使快照失效。

本轮重跑入口为 `run_context.ps1`，默认写入 `results/context_reproduction/`。先执行开发与10个固定解析纹理控制，再冻结、回归和一次新图检查；最后生成报告、图及输入完整性核验。拒绝覆盖已有阶段，新一轮使用新的 `-Output`。定位证据在 `results/rejection_diagnosis/`，结果在 `results/context/`。

支撑的 `contrast_context='qualified'` 保留原路径；`amplitude`只改变邻带取样范围，不扩大减除资格。348项完整测试及原默认逐像素一致性核对已记录。旧精确快照因源码新增上下文选项而在当前分支不再匹配；用 `git worktree add ..\periodic-narrow-baseline narrow-profile-baseline` 建立79da40c独立目录后，再运行该目录的 `run_narrow.ps1`，不修改旧SHA。

上一轮窄条对照使用 `run_narrow.ps1`，默认新目录 `results/narrow_reproduction/`。本轮69022等已成为设计回归材料，新4图从旧210图及前三轮各4图之外预选，保留完整官方JPEG提取目录以重建输入。

上一轮空间支撑完整提交为 `support-ablation-baseline` 标签，原 `run_support.ps1` 与结果保持不变；默认输出 `results/support_reproduction/`。本轮新增4张检查图排除了旧210图及风险／支撑轮各4图，保留完整官方JPEG提取目录以重建输入。

前一轮风险实验用 `git worktree add ..\periodic-risk-baseline risk-refinement-baseline` 建立独立目录，再在该目录安装环境、执行 `run_refinement.ps1`；当前支撑增强源码不应通过修改旧SHA冒充旧冻结实验。

历史全量实验用 `git worktree add ..\periodic-v3-baseline v3-validated-baseline` 建立独立目录，在其中安装环境并执行 `run_v3.ps1`。该脚本会重新校准并覆盖所在目录配置与结果；当前源码重跑得到的是新实验，不能直接沿用历史表的结论。

## 来源与使用条件

第三方代码、数据和图像各有使用条件，见 `THIRD_PARTY_NOTICES.md` 与对应来源记录。仓库现为公开，公开可见不改变第三方材料的使用条件，也未给全部材料统一授予通用开源许可。
