# 第三方来源与使用条件

本仓库包含课程研究代码、外部方法移植和数据来源记录；没有把全部内容统一重新许可为MIT等通用开源许可。

- **ARPENOS**：作者代码及其Python移植用于学术研究。保留 `third_party/arpenos/author_page.html`、原始MATLAB代码、下载哈希、引用与 `PORT_NOTES.txt`。该移植有明确数值差异，未完成MATLAB/Octave原版逐输入核验。
- **AGNF资料**：`third_party/agnf/license.txt`保留作者提供的BSD三条款。项目未将起步原型冒称AGNF原版复现。
- **BSDS500**：只使用原始图像构建合成干扰任务。来源、引用与固定文件哈希在 `data/source/provenance.json`、各版manifest中；原图和归档不纳入Git。
- **DTD** 的 12 张纹理压力测试照片和后续选择的 4 张照片均来自官方 test1 划分。本轮发布的误差图展示两个恢复结果与清晰参考的亮度差。原照片、完整灰度对照和浮点参考由实验脚本在本地重建。[官方数据与研究用途说明](https://www.robots.ox.ac.uk/~vgg/data/dtd/) · [子集来源](data/textures_v3/provenance.json) · [4 张补充照片的来源与哈希](results/weave/photos_run.json)
- **无参考实例**：OpenCV周期干扰图和ARPENOS作者canvas案例的来源、解释边界与使用说明保存在 `third_party/real_examples/provenance.json`。原图不纳入Git，没有配对参考时不报告PSNR/SSIM。

上述记录应随代码保留。原作者及图片权利人的权利不因本次仓库整理而改变。
