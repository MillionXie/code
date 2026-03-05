# Electronic Baseline ConvVAE (MNIST + CIFAR-10)

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

- `--dataset {mnist,cifar10}`
- `--latent_dim`（默认 100）
- `--model_size {tiny,small}`
- `--beta`
- `--epochs --batch_size --lr`
- `--recon_loss {auto,bce,mse}`
  - `auto`: MNIST->BCE, CIFAR-10->MSE
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
  --latent_dim 100 \
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
python train_vae.py --dataset cifar10 --model_size small --latent_dim 100 --beta 1.0 --epochs 50 --batch_size 128 --lr 1e-3 --recon_loss auto --out_range zero_one --data_root ./data --outdir ./outputs/cifar10_small
```

3. 评估（test MSE/PSNR + 重建图）

```bash
python eval_vae.py \
  --checkpoint ./outputs/mnist_tiny/checkpoints/best.pt \
  --data_root ./data \
  --outdir ./outputs/mnist_tiny/eval
```

4. 随机采样生成

```bash
python sample_vae.py \
  --checkpoint ./outputs/mnist_tiny/checkpoints/best.pt \
  --n_samples 64 \
  --grid_size 8 \
  --outdir ./outputs/mnist_tiny/sample
```

5. 潜空间分析

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

## LatentProvider 统一接口

- `GaussianPriorProvider`: 先验采样 `z ~ N(0, I)`
- `EncoderPosteriorProvider`: 编码器后验 `q(z|x)`（可采样或直接取 `mu`）

后续新增 `ScatteringProvider` 时可直接复用现有训练/生成主流程。

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
- 训练时采用 `q(w|x)=N(mu_w, sigma_post^2 I)`，其中 `sigma_post` 为常数（不学习），采样得到 decoder 输入。
- KL 在 `mu_w` 上按 biased Gaussian prior 计算（`m0`,`prior_sigma0` 由配置指定）。
- 默认不做 ROI 裁剪、不做额外插值缩放；建议只用一次 `pool_kernel == pool_stride` 的 pooling 直接把 `resize_hw` 降到 `model.latent_hw`（微透镜阵列规则分块汇聚）。
- 若不想池化，设 `optics.sensor.pool_type: none`，并让 `model.latent_hw` 与解 padding 后的光场尺寸一致。
- 每个 epoch 的 `optics/epoch_xxx_optics.png` 包含 5 行：`原图`、`散射前光场`、`散射后传播到探测面的强度(池化前)`、`潜空间均值mu_w(池化后)`、`重建图`。
