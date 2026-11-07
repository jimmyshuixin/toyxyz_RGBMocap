# Fork 集成与验证

本 fork 以原版 `75cda26531b47f41c75d4aa4a82c4f135daa6eea` 为 PMX 重定向基准。

- [ExMR 长视频优化](LONG_VIDEO_EXMR.md)与独立人脸功能继续保留。源数据准备仍复用静态 NPZ 数组。
- PMX 映射、接触求解、骨架注册、Blender 烘焙及 FBX 辅助骨选择恢复原版；烘焙 NPZ 单次读取和接触几何分块优化随该路径撤回。
- [两段躯干 PMX 扩展](PMX_TWO_SPINE.md)已撤回。现在必须具有原版要求的真实「上半身」「上半身1」「上半身2」骨骼；不再自动补虚拟骨或替换编号映射。

功能已经集成到源码，无需再次运行旧补丁安装器。历史补丁包及其回滚备份仅保留在原安装目录，未纳入 Git。既有模型、视频、动作项目、角色和运行环境也不属于提交内容。

## 使用与更新

按根目录 README 安装或启动。长视频使用 ExMR；其他后端原有帧数限制不变。PMX 请使用符合原版骨架要求的模型。旧扩展映射缓存会被明确拒绝，需要重新导入支持的模型，再执行重定向和 VMD 导出。兼容的三段躯干缓存仍可用。既有动作文件不会自动重写，原 PMX 和贴图目录应保持完整。

本地 `origin` 指向 `jimmyshuixin/toyxyz_RGBMocap`，`upstream` 指向 `toyxyz/toyxyz_RGBMocap`，`main` 跟踪 `origin/main`。因此 `update.cmd` 获取的是本 fork 的更新，并执行运行环境验证/修复。

以后需要主动合并原作者的新版本时，先关闭应用，提交或备份本地源码改动，再执行：

```text
git fetch upstream
git merge upstream/main
```

若有冲突，应同时保留上游变化与本 fork 的功能，复跑下列测试后再运行 `install.cmd` 并推送到 `origin`。如果源码发生变化，也应更新 `SOURCE_MANIFEST.json` 中已跟踪文件的大小与 SHA-256；不能用整个安装目录生成公开文件清单。

## 集成时历史验证（2026-11-07，恢复原版之前）

以下数量和结果对应已集成的旧扩展状态，不代表恢复原版后的模型支持范围。

| 检查 | 环境与结果 |
| --- | --- |
| 长视频回归 | Windows；Python 3.12.10、Torch 2.8.0+cu128、NumPy 2.3.5、SciPy 1.15.2；14 项通过 |
| PMX 回归 | Windows；Python 3.10.22、NumPy 1.23.5、SciPy 1.14.1；17 项通过 |
| 长序列结构 | 默认 BodyNet、FootRefiner、HandRefiner 的 10,800 帧随机输入 forward 通过；单次查询/键数量上限为 256/375 |
| 注意力数值对照 | CPU FP32 最大绝对误差 6.56e-7；这是内部张量误差，不是动作捕捉精度 |
| 兼容性 | 缓存身份、NPZ 复用、异常关闭、两段/三段躯干及原有映射路径通过测试 |

现行回归可从源码根目录运行（测试内容已随恢复原版调整）：

```text
runtime_exmr\python.exe -B -X utf8 -m unittest discover -s tests -p "test_longvideo_*.py" -v
runtime\python.exe -B -X utf8 -m unittest discover -s tests -p "test_pmx_upstream*.py" -v
```

测试使用合成 CPU 输入、未训练网络和 Blender API 替身，不需要模型权重或角色素材。历史集成当时未执行真实视频推理、六分钟 CUDA 全流程、真实 Blender/PMX 导入导出、Windows GUI 交互或 MMD 播放。PMX 文档中的历史 Linux/Blender 数据是旧补丁的验证记录，不能代替本次验证。

保持 Resource Safety 及原有 OOM 恢复机制。长视频优化不保证任意显卡的显存占用、耗时或动作质量。原项目、ExMR 和素材各自的许可继续适用。
