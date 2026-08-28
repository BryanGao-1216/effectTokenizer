# 使用 Open X-Embodiment 训练 Action VQ-VAE

`myStudy` 的 VQ-VAE 训练使用 VQ-VLA 同款的 OXE 数据语义，既支持准备好的 TFDS/RLDS builder，也支持直接读取 `jxu124/OpenX-Embodiment` 的 tar shards，不再接收或生成用于训练的 `.npy` action chunk。

## 数据流

RLDS trajectory 会先经过各数据集专用的 OXE standardization transform，统一为相对末端执行器动作；动作始终保留数据集原生帧率，再按每个数据集原生 action 的 q01/q99 将非夹爪维度归一化到 `[-1, 1]`，夹爪维度保持原值。`--window-duration-seconds` 指定统一的物理时间窗，程序根据独立频率表换算每个数据集的窗口帧数。例如 1 秒窗口对 20 Hz 数据取 20 帧，对 10 Hz 数据取 10 帧，不再做上采样或下采样。

轨迹尾部不足一个时间窗时，相对 XYZ/RPY 动作补零，绝对夹爪动作重复最后一个有效值。不同原生帧数的 chunk 仅在组成 batch 时中性补到当前 mixture 的最大帧数；effect 累计值不会因此改变，评估也会根据随样本保存的真实窗口帧数忽略这些 batch padding。频率表位于 `scripts/action_vqvae/rlds/oxe/control_frequencies.py`；遇到没有已核实频率的新数据集会直接报错，不会猜测。

频率表记录的是数据集的标称控制率，因为 OXE 没有跨数据集统一、可靠的 step timestamp。OpenVLA 的 LIBERO `*_no_noops` 生成脚本会在调用下一次 20 Hz 仿真 step 前跳过 no-op，因此这里仍按 20 Hz 的时间压缩重放序列处理。

窗口起点由 `--sampling-stride-seconds` 控制，省略时使用窗口时长的四分之一。它也会按各数据集原生 Hz 换算成帧数；默认 1 秒窗口对应 0.25 秒 stride。对于无法整数表示的组合使用最接近的正整数帧数，例如 12.5 Hz 的 1 秒窗口取 13 帧。

训练入口不会再做模型侧二次归一化。`DataLoader` 固定使用 `num_workers=0`，TFDS/DLimp 会自行管理并行读取。

## 环境和数据

建议使用 Python 3.10/3.11：

```bash
pip install -r requirements-oxe.txt
```

对于原 TFDS 后端，`--data-root-dir` 应指向 TFDS/RLDS 数据根目录，目录下放各 dataset builder，例如 `fractal20220817_data`、`kuka`、`bridge_orig`。

对于 OpenX tar 后端，不需要解压，保持以下结构：

```text
/data/OpenX/
├── asu_table_top_converted_externally_to_rlds/
│   └── asu_table_top_converted_externally_to_rlds_00000.tar
├── bc_z/
│   ├── bc_z_00000.tar
│   └── bc_z_00001.tar
└── droid/
    └── droid_00000.tar
```

启动时使用：

```bash
python scripts/action_vqvae/train_action_vqvae.py \
  --data-root-dir /data/OpenX \
  --train-dataset-name my_openx_mix \
  --rlds-storage-format webdataset \
  --window-duration-seconds 1.0 \
  --sampling-stride-seconds 0.25 \
  --action-dim 7 \
  --shuffle-buffer-size 50000 \
  --no-rlds-validation \
  ...
```

`--rlds-storage-format=auto` 是默认值。它会逐个检查 mixture 中的数据源：存在
`root/<dataset_name>/*.tar` 的数据源走 tar/pickle 流程，其余数据源走 TFDS/RLDS
流程；两种来源同时存在时自动启用 `hybrid`，在统一的 action chunk 层按 mixture
权重采样并使用同一个 shuffle buffer。也可以显式指定
`--rlds-storage-format=hybrid`。显式指定 `tfds` 会忽略已有 tar，显式指定
`webdataset` 则要求所有来源都存在 tar shard。tar 中保存的是 pickle，只应读取可信的数据文件。

同一个逻辑数据集如果同时具有 tar 和 TFDS 副本，`auto/hybrid` 默认只读取 tar，避免重复采样相同 episode；需要强制读取 TFDS 时使用 `--rlds-storage-format=tfds`。

tar 第一次使用时会扫描每个数据集，计算 OXE standardizer 之后、原生帧率 action 的
`mean/std/min/max/q01/q99`，并在 shard 旁缓存为 `dataset_statistics_<hash>.json`。之后会直接复用缓存。训练流仍然按照 mixture 权重抽样，并先填满 `--shuffle-buffer-size` 个 action chunk；快速测试时可将其设为 `1000`。

窗口时长不会改变逐帧 q01/q99，因此不同窗口时长可以复用同一个原生帧率统计缓存。

tar 没有官方 validation split，因此本地读取器按排序后的 episode 固定使用前 95% 训练、后 5% 验证。设置 `--no-rlds-validation` 可关闭验证。

两种后端都不是 LeRobot parquet 格式。

## 启动训练

```bash
DATA_ROOT=/path/to/rlds_root \
TRAIN_DATASET_NAME=oxe_magic_soup \
OUTPUT_DIR=outputs/my_oxe_run \
bash scripts/train.sh
```

也可以训练单个 builder：

```bash
DATA_ROOT=/path/to/rlds_root \
TRAIN_DATASET_NAME=bridge_orig \
OUTPUT_DIR=outputs/bridge_only \
bash scripts/train.sh
```

常用环境变量：

```bash
VQ_LOSS_WEIGHT=5.0 \
NUM_QUANTIZERS=4 \
CODEBOOK_SIZE=256 \
EMA_DECAY=0.8 \
VQ_EPS=1e-5 \
KMEANS_INIT_ITERS=10 \
DEAD_CODE_THRESHOLD=0.0 \
bash scripts/train.sh
```

`DEAD_CODE_THRESHOLD=0.0` 表示关闭 dead-code 自动替换；设置为正数后，EMA cluster size 低于该阈值的 code 会用当前 batch 的 latent 重新初始化。

当前目标函数为：

```text
L = L_full
  + alpha * L_Q0
  + beta * E[L_prefix]
  + gamma * L_usage
  + vq_loss_weight * L_commitment
```

其中 `L_Q0` 强制第一层 codebook 独立重建动作窗口，`L_prefix` 对每个 batch 中随机采样的 residual-code 前缀进行重建，`L_usage` 同时约束 Q0 的边缘分布均衡和单样本分配置信度。相应参数为 `--q0-loss-weight`（默认 1.0）、`--prefix-loss-weight`（默认 0.5）、`--usage-loss-weight`（默认 0.01）和 `--usage-temperature`（默认 1.0）。VQ 必需的 commitment loss 仍由 `--vq-loss-weight` 单独控制。

## TensorBoard

启动脚本默认启用 TensorBoard，event 文件写入 `${OUTPUT_DIR}/tensorboard`，每 10 个 step 写入一次并每 30 秒刷新。目录和频率可通过环境变量修改：

```bash
TENSORBOARD_LOG_DIR=outputs/my_oxe_run/tensorboard \
TENSORBOARD_LOG_EVERY_STEPS=10 \
TENSORBOARD_FLUSH_SECS=30 \
bash scripts/train.sh
```

另开终端启动面板：

```bash
tensorboard --logdir outputs/my_oxe_run/tensorboard --port 6006
```

其中会分别记录 full/Q0/prefix reconstruction、usage KL/entropy、VQ commitment 及其加权值、学习率、梯度范数，以及每一层 codebook 的 perplexity、已使用 code 数和最大 code 占比。控制台优先显示下游实际使用的 Q0 指标。

## 交互式查看聚类中心

聚类中心部分直接读取 checkpoint 中各级 Residual-VQ codebook；code 使用率部分读取 OXE/RLDS 测试数据，整个流程都不需要 `.npy` 文件：

```bash
bash scripts/visualize.sh
```

默认生成 `outputs/action_vqvae/codebook_viz/codebook_centers_3d.html` 并尝试在浏览器中打开。页面支持拖拽旋转、滚轮缩放、平移、悬停查看 code id/EMA cluster size/dead-code 状态，以及通过图例单独显示或隐藏某一级量化器。

codebook 中心原本位于 `latent_dim` 维空间，页面中的三个坐标轴是对所有 Residual-VQ 层共享拟合的 PCA 投影，因此不同量化器之间的相对位置仍可比较。脚本还会输出同名 CSV（每个中心的 3D 坐标和状态）与 JSON（投影解释方差及 codebook 摘要）。

脚本还会从 OXE/RLDS validation split 读取测试 action chunk，经过当前 checkpoint 编码后生成 `test_code_usage_polar.html`。圆周表示 code ID，半径表示该 code 在测试数据中的选择频率，每一级 Residual-VQ 对应一条折线；图例同时显示每层已使用 code 数和 perplexity。原始计数与频率保存在 `test_code_usage.csv`，汇总保存在 `test_code_usage.json`。

测试数据参数可通过环境变量调整：

```bash
DATA_ROOT=/path/to/rlds_root \
TEST_DATASET_NAME=bridge_orig \
TEST_SAMPLES=8192 \
TEST_BATCH_SIZE=512 \
bash scripts/visualize.sh
```

如果只需要查看 checkpoint 中的聚类中心，可以直接运行 Python 脚本并加上 `--no-code-usage`；此时完全不加载测试数据。

## 离线评估

当前 Residual-VQ 模型可使用完整离线评估流程：

```bash
bash scripts/eval.sh
```

该流程从 OXE validation stream 读取数据，评估 Q0 action abstraction、直接 K-means
基线、逐 residual-prefix 重建、每层 codebook health、Q0 cycle consistency 和 token
扰动稳定性。详细指标和输出文件见
`scripts/action_tokenizer_eval/README.md`。评估流程不接收 `.npy` 数据。

命名混合及权重定义在 `scripts/action_vqvae/rlds/oxe/mixtures.py`。可用的典型配置包括 `rtx`、`oxe_magic_soup` 和 `oxe_magic_soup_plus`；也可以在该文件中加入自己的组合。训练输出会额外保存 `dataset_statistics.json`，记录每个数据集实际采用的动作统计量。

RLDS/OXE 代码由 VQ-VLA 的 MIT-licensed `prismatic/vla/datasets/rlds` 数据管道复制并改为本项目内的相对导入；原许可证保存在 `scripts/action_vqvae/rlds/LICENSE.VQ-VLA`。本地 wrapper 只输出现有模型所需的 action tensor。
