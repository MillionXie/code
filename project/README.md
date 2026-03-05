# Electronic Baseline ConvVAE (MNIST + FashionMNIST + CIFAR-10)

一个可直接运行的卷积 VAE 基线工程，支持：
- 训练（recon + beta * KL）
- 重建可视化
- 先验随机采样生成
- 潜空间插值生成
- 潜空间统计分析（均值/方差直方图、协方差谱、KL 分布、PCA、可选 t-SNE）
- 统一结果保存（`results.json` / `summary.csv` 或 `summary.json`）

## 目录结构

```text
project/
  README.md
  requirements.txt
  train_vae.py
  eval_vae.py
  sample_vae.py
  analyze_latent.py
  models/
    __init__.py
    conv_vae.py
  data/
    datasets.py
  latent/
    __init__.py
    providers.py
  utils/
    seed.py
    io.py
    metrics.py
    viz.py
    logger.py
```

## 环境安装

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练参数说明（核心）

- `--dataset {mnist,fashionmnist,cifar10}`
- `--latent_dim`（默认 100）
- `--model_size {tiny,small}`
- `--beta`
- `--epochs --batch_size --lr`
- `--recon_loss {auto,bce,mse}`
  - `auto`: MNIST/FashionMNIST->BCE, CIFAR-10->MSE
- `--out_range {zero_one,neg_one_one}`
  - `zero_one`：输出 sigmoid，输入在 `[0,1]`
  - `neg_one_one`：输出 tanh，输入在 `[-1,1]`

> 注意：当使用 `BCE` 时必须使用 `--out_range zero_one`。

## 命令示例

1. MNIST tiny 训练

```bash
python train_vae.py \
  --dataset mnist \
  --model_size tiny \
  --latent_dim 64 \
  --beta 1.0 \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --recon_loss auto \
  --out_range zero_one \
  --data_root ./data \
  --outdir ./outputs/mnist_tiny
```

2. CIFAR-10 small 训练

```bash
python train_vae.py --dataset cifar10 --model_size small --latent_dim 64 --beta 1.0 --epochs 50 --batch_size 128 --lr 1e-3 --recon_loss auto --out_range zero_one --data_root ./data --outdir ./outputs/cifar10_small
```

3. FashionMNIST tiny 训练

```bash
python train_vae.py --dataset fashionmnist --model_size tiny --latent_dim 64 --beta 1.0 --epochs 30 --batch_size 128 --lr 1e-3 --recon_loss auto --out_range zero_one --data_root ./data --outdir ./outputs/fashionmnist_tiny
```

4. 评估（test MSE/PSNR + 重建图）

```bash
python eval_vae.py \
  --checkpoint ./outputs/mnist_tiny/checkpoints/best.pt \
  --data_root ./data \
  --outdir ./outputs/mnist_tiny/eval
```

5. 随机采样生成

```bash
python sample_vae.py \
  --checkpoint ./outputs/mnist_tiny/checkpoints/best.pt \
  --n_samples 64 \
  --grid_size 8 \
  --outdir ./outputs/mnist_tiny/sample
```

6. 潜空间分析

```bash
python analyze_latent.py \
  --checkpoint ./outputs/mnist_tiny/checkpoints/best.pt \
  --data_root ./data \
  --run_tsne \
  --outdir ./outputs/mnist_tiny/analyze
```

## 输出内容

训练目录（`train_vae.py --outdir`）下包含：
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `reconstructions/epoch_xxx_recon.png`
- `samples/epoch_xxx_sample.png`
- `interpolations/epoch_xxx_interp.png`
- `latent_viz/epoch_xxx_mu_heatmap.png`
- `latent_viz/epoch_xxx_logvar_heatmap.png`
- `latent_viz/epoch_xxx_kl_heatmap.png`
- `logs/train.log`
- `history.csv`
- `results.json`
- `summary.csv`

分析目录（`analyze_latent.py --outdir`）下包含：
- `mu_dim_mean_hist.png`
- `mu_dim_var_hist.png`
- `mu_cov_eigenspectrum.png`
- `kl_distribution_hist.png`
- `pca_2d.png`
- `tsne_2d.png`（若开启且可用）
- `summary.json`
- `summary.csv`

## 电子基线说明

- 当前电子基线为纯电子 ConvVAE（不依赖 `LatentProvider` 抽象）。
- 默认公平配置为 `model_size=tiny, latent_dim=64`，参数量约 `24k`（MNIST/CIFAR 都接近）。

## 新增：Map-Latent 双入口（共享 encoder/decoder）

在不改动原有 `train_vae.py` 的前提下，新增两条完全分离入口：

- `train_map_electronic.py`：Electronic OLS，`z_map` 直接进 decoder
- `train_map_optical.py`：Optical OLS，`z_map -> 光学链路 -> decoder`

二者共享同一核心模型 `models/vae_map_core.py`，仅中间 adapter 不同：

- 电子：`IdentityAdapter`
- 光学：`OpticalOLSAdapter`

### Map-Latent 训练

电子版（MNIST）：

```bash
python train_map_electronic.py \
  --config ./configs/map_electronic_mnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_elec_mnist
```

光学版（MNIST）：

```bash
python train_map_optical.py \
  --config ./configs/map_optical_mnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_opt_mnist
```

电子版（FashionMNIST）：

```bash
python train_map_electronic.py \
  --config ./configs/map_electronic_fashionmnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_elec_fashionmnist
```

光学版（FashionMNIST）：

```bash
python train_map_optical.py \
  --config ./configs/map_optical_fashionmnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_opt_fashionmnist
```

### Map-Latent 采样

电子版采样：

```bash
python sample_map_electronic.py \
  --config ./configs/map_electronic_mnist.yaml \
  --checkpoint ./outputs/map_elec_mnist/checkpoints/best.pt \
  --outdir ./outputs/map_elec_mnist/sample
```

光学版采样：

```bash
python sample_map_optical.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --outdir ./outputs/map_opt_mnist/sample
```

### Map-Latent 评估与分析

评估：

```bash
python eval_map.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --mode optical \
  --data_root ./data \
  --outdir ./outputs/map_opt_mnist/eval
```

分析：

```bash
python analyze_map.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --data_root ./data \
  --run_tsne \
  --outdir ./outputs/map_opt_mnist/analyze
```

### Optical Sensor/Pooling 物理约定

- `pad/unpad` 仅在 `angular_spectrum_propagate` 内部处理。
- 光学编码链路统一为：`输入图像(上采样到200x200) -> (传播+相位层)x2 -> 传播到散射介质 -> 静态散射 -> 短距离传播 -> 强度探测 -> pooling -> mu_w`。
- 当 `model.arch: pure_optical` 时，decoder 也为纯光学：`z -> 复数场(由强度开根构造) -> 4层(传播+相位层) -> 强度探测 -> 重建图`。
- 训练时采用 `q(w|x)=N(mu_w, sigma_post^2 I)`，其中 `sigma_post` 为常数（不学习），采样得到 decoder 输入。
- KL 在 `mu_w` 上按 biased Gaussian prior 计算（`m0`,`prior_sigma0` 由配置指定）。
- 默认不做 ROI 裁剪、不做额外插值缩放；建议只用一次 `pool_kernel == pool_stride` 的 pooling 直接把 `resize_hw` 降到 `model.latent_hw`（微透镜阵列规则分块汇聚）。
- 若不想池化，设 `optics.sensor.pool_type: none`，并让 `model.latent_hw` 与解 padding 后的光场尺寸一致。
- 每个 epoch 的 `optics/epoch_xxx_optics.png` 包含 5 行：`原图`、`散射前光场`、`散射后传播到探测面的强度(池化前)`、`潜空间均值mu_w(池化后)`、`重建图`。

### 光学记忆效应测试（新）

新增脚本：`optics/test_scattering_memory.py`，用于固定流程
`propagate(a) -> scatter -> propagate(b)` 比较三类散射建模（IID/Correlated/Angle-limited）在不同相关长度或 NA 下的“记忆效应”。

```bash
python optics/test_scattering_memory.py \
  --outdir ./outputs/optics_memory_demo \
  --distance_a_mm 20 \
  --distance_b_mm 5 \
  --corr_lens_px 1.0,3.0,6.0 \
  --na_values 0.08,0.15,0.25
```

输出包括：
- `*_intensity_grid.png`：各 tilt 输入下的探测强度图
- `memory_peak_corr_vs_tilt.png`：峰值相关系数曲线
- `memory_peak_shift_x_vs_tilt.png`：峰值位移曲线
- `summary.json` / `summary.csv`：记忆范围统计（`memory_width_cpp`）
