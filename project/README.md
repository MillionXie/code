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
python train_vae.py \
  --dataset cifar10 \
  --model_size small \
  --latent_dim 100 \
  --beta 1.0 \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --recon_loss auto \
  --out_range zero_one \
  --data_root ./data \
  --outdir ./outputs/cifar10_small
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
