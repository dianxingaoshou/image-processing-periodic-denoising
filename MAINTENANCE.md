# 维护与复现实验

## 运行路径

维护当前算法从 [project_phase.py](src/project_phase.py) 开始。该入口复用已经取得的频谱、扫频模型和窄条判断。对某张图实际采用哪条路径，可在单图 CLI 的 `decisions.json` 中查看 `route`、`reason` 和 `evidence`。两份正式配置在 [src/configs/](src/configs/)。

| 要修改的内容 | 对应源码 |
|---|---|
| 新相位的带外拟合和带内确认 | [phase_refit.py](src/phase_refit.py) |
| 实测织纹峰保护、前版扫频确认及主峰修正 | [collision_review.py](src/collision_review.py)、[guarded_carrier.py](src/guarded_carrier.py)、[peak_guard.py](src/peak_guard.py) |
| 8／16 像素窄条的候选、紧致支撑和参考邻区 | [narrow_profiles.py](src/narrow_profiles.py)、[narrow_support.py](src/narrow_support.py)、[narrow_context.py](src/narrow_context.py) |
| 一般图像的多尺度估计与叠加恢复 | [local_estimation_v3.py](src/local_estimation_v3.py)、[project_pipeline.py](src/project_pipeline.py) |

单图入口可从任意目录调用，图像尺寸保持不变。研究实验统一从仓库根目录运行。先按 [README](README.md) 安装依赖，并使用同一虚拟环境的 Python。

```powershell
python -m pytest -q
python experiments/run.py check
```

测试覆盖推理模块、织纹保护、图像输入输出与已发表基线。第二条命令核对当前源文件、历次比较所固定的输入协议、基线数据和本轮结果的哈希。它不需要照片。数值复现需相同的图像解码和数值库，精确版本见 [requirements-lock.txt](requirements-lock.txt)。

## 从公开来源重建对照

实验入口把运行所需源码和登记数据按冻结协议记录的相对路径复制到忽略版本控制的 `.study/`，所有命令在该工作目录运行。这样仓库中每份推理源码仅保留一个文件，历史评估仍可按最初记录的路径和文件字节核对。

```powershell
python experiments/run.py prepare
python experiments/run.py diagnose --out results/reproduction
python experiments/run.py evaluate freeze --out results/reproduction
python experiments/run.py evaluate run --out results/reproduction
python experiments/run.py benchmark --out results/reproduction --rounds 4
python experiments/run.py benchmark-controls --out results/reproduction
python experiments/run.py figures --results results/reproduction --out figures
```

第一条命令校验 7 张 BSDS、16 张 DTD 源图及用到的数组。从头安装时它下载官方 BSDS500 压缩包，DTD 只下载指定图像。已有研究数据时可使用 `prepare --cache "D:\data\prepared"`，传入含有 `source/` 和 `textures_v3/` 的数据目录。只需生成发布图时，完成 `prepare` 后可直接执行 `python experiments/run.py figures --results results/phase_refit --out figures`。

新实验结果位于 `.study/results/reproduction/`，灰度与差值图位于 `.study/figures/`。再次运行 `freeze` 请给 `--out` 指定另一新目录。基准计时交错运行两个版本的常规入口；请在质量评估结束后单独运行。所有生成的数值、时间和环境指纹都保留在相应结果目录中。

## 发布与来源

当前主分支按推理代码、可运行实验、必要基线、测试和结果分层。发布图展示恢复误差，完整灰度对照可以在本地重建。官方 DTD 提供研究用途的图像和数据说明。[DTD 来源](https://www.robots.ox.ac.uk/~vgg/data/dtd/) · [第三方说明](THIRD_PARTY_NOTICES.md)

此前各轮开发的诊断工具、消融、原版目录及旧冻结报告在 [研究快照 1740a94](https://github.com/dianxingaoshou/image-processing-periodic-denoising/tree/1740a94)。需要旧版工作目录时，可从已有克隆执行 `git worktree add --detach ../periodic-research-archive 1740a94`。
