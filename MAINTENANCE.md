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
.\.venv\Scripts\python.exe -m pytest -q tests original/periodic_notch_starter/test_demo.py
```

数据清单会入Git，原图和NPY不会。部分数据完整性测试会读取NPY，因此刚克隆后直接运行全量测试会缺少数据。`prepare_v3_data.py`按已冻结的文件名和哈希重建210张BSDS参考、12张DTD参考及2个无配对实例，不重新选择审计图。

GitHub Actions只执行不依赖下载数据的数值测试，显式排除两个本地数据完整性用例；这不等于重跑完整实验或核验旧环境快照。

## 处理自己的图片

```powershell
# P0数值修复，保留已测量的耦合权重行为
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_p0_coupled.json --out results/own_image_v3

# 解耦权重仅作为当前实验分支使用
.\.venv\Scripts\python.exe filter_image_v3.py "input.png" --config configs/v3_frozen.json --out results/own_image_v3_decoupled
```

当前证据未支持把解耦版视为普遍更优替代。先读 `results/v3/DECISION.md`、`results/v3/RESULTS.md` 和 `results/v3/texture_tradeoff/EXPLORATORY.md`，其中保留性能损失、纹理净损伤和负面结果。

## 修改与提交

```powershell
git switch -c improve/my-change
# 修改代码，运行有关测试；涉及冻结科学源码时重新校准并生成新记录
git status
git add 修改的文件
git commit -m "Describe the concrete change"
git push -u origin improve/my-change
```

初始导入提交和 `v3-validated-baseline` 标签用于回到本次完整代码。保持原模块相对路径；移动或重新格式化被冻结文件会改变字节哈希。`.gitattributes`关闭换行归一化，避免Git自动转换换行使既有快照失效。

完整重跑入口为 `run_v3.ps1`。它会重建数据、测试、校准、执行主实验与模型外验证并重新生成结果，因此重跑前先提交现有记录；如需保留两轮完整产物，使用项目副本。

## 来源与使用条件

第三方代码、数据和图像各有使用条件，见 `THIRD_PARTY_NOTICES.md` 与对应来源记录。本仓库保持私有，未给第三方材料统一授予通用开源许可。
