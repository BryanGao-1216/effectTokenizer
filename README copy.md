# Direct Effect Tokenizer

这一版不训练 VAE，也不使用 MiniBatch K-means。每个 action chunk 直接压缩为一个
七维“终点效果”向量，再用普通 Lloyd K-means 得到一个 token。

## 数据与特征约定

处理顺序固定为：

1. 使用 OpenX/VQ-VLA 中每个数据集自己的 standardization transform，把动作统一为
   EEF delta XYZ、delta RPY 和绝对夹爪状态；
2. 在计算统计量和 action chunk 之前重采样到 10 Hz；
3. 对每个数据集、每个动作维度只按照自己的 q01/q99 裁剪，不在数据集内部缩放；
4. 形成固定长度 action chunk，尾部相对动作补零、绝对夹爪重复最后值；
5. 计算 `sum(XYZ), sum(RPY), gripper[-1]-gripper[0]` 七维 effect；
6. 将所有采样到的 effect 合并，用训练集的一套 pooled z-score 做统一标准化；
7. 在标准化后的 effect 上执行普通 full-data Lloyd K-means。

pooled z-score 会让平移、旋转、夹爪三个部分不会仅因单位和数值尺度不同而主导欧氏
距离。`--gripper-weight` 可在标准化之后额外调节夹爪变化的重要性，默认值为 1。

`--kmeans-assignment-batch-size` 只把距离矩阵分块以限制显存占用；每次中心更新仍遍历
全部拟合样本，因此不是 MiniBatch K-means。K-means++ 默认也使用全部拟合样本；只有
显式把 `--kmeans-init-candidate-samples` 设为正数时，初始化阶段才会使用候选子集，
后续 Lloyd 迭代仍然使用全部数据。

## 训练

在 `effectTokenizer` 项目根目录运行：

```bash
bash scripts/train.sh
```

默认使用 500000 个 chunk、256 个 token。训练产物包括：

- `effect_tokenizer.pt`：中心、全局 mean/std、数据约定及 K-means 配置；
- `effect_tokenizer.json`：便于直接查看的同内容元数据和原始单位下的中心；
- `training_metrics.json`：拟合集上的误差、margin 与 token 使用分布；
- `dataset_statistics.json`：本次 OpenX 数据集的逐数据集统计量。

## 评估和轨迹查看

```bash
bash scripts/eval.sh
```

评估会从 checkpoint 读取训练时的 10 Hz、horizon、stride、全局 mean/std 等配置，避免
训练和评估预处理不一致。默认输出：

- `summary.md` 和 `metrics.json`：held-out 聚类误差、R²、usage、perplexity、entropy、
  top probability、assignment margin；
- `per_token_metrics.csv`：每个 token 的频率、簇内误差和原始单位中心；
- `token_usage_polar.html`：可交互的极坐标 token 使用分布；
- `token_trajectories/index.html`：每个 token 的可交互轨迹浏览器；
- `trajectory_examples.npz`：绘图所用原始轨迹、均值轨迹、中心和计数。

每个 token 页面包含位置三维曲线、旋转三维曲线和夹爪二维曲线。蓝线是实际归入该
token 的 held-out 原始轨迹，红线是这些轨迹的逐时刻均值。黑色虚线只表示 K-means
中心对应的终点 effect；直接聚类没有 decoder，因此它不是模型生成的唯一轨迹。
