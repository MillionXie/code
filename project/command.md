# Command Cookbook

> 目的：把常用训练/评估/采样/分析命令整理成可直接粘贴执行的版本。
> 说明：以下命令默认在 `project/` 目录执行。

## 0) 环境准备（首次）

```bash
# 进入项目
cd /Users/million/Library/CloudStorage/OneDrive-个人/2026OpticsGen/code/project

# 可选：创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt
```

## 1) 电子向量基线（原 train_vae.py，不受 map-optical 改动影响）

```bash
# MNIST tiny 训练（公平电子基线，约 24k 参数）
python train_vae.py \
  --dataset mnist \
  --model_size tiny \
  --latent_dim 64 \
  --beta 1.0 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --recon_loss auto \
  --out_range zero_one \
  --data_root ./data \
  --outdir ./outputs/vae_mnist_tiny
```

```bash
# CIFAR-10 small 训练
python train_vae.py \
  --dataset cifar10 \
  --model_size small \
  --latent_dim 64 \
  --beta 1.0 \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --recon_loss auto \
  --out_range zero_one \
  --data_root ./data \
  --outdir ./outputs/vae_cifar10_small
```

```bash
# FashionMNIST tiny 训练
python train_vae.py \
  --dataset fashionmnist \
  --model_size tiny \
  --latent_dim 64 \
  --beta 1.0 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --recon_loss auto \
  --out_range zero_one \
  --data_root ./data \
  --outdir ./outputs/vae_fashionmnist_tiny
```

```bash
# 评估（test MSE/PSNR + 重建图）
python eval_vae.py \
  --checkpoint ./outputs/vae_mnist_tiny/checkpoints/best.pt \
  --data_root ./data \
  --outdir ./outputs/vae_mnist_tiny/eval
```

```bash
# 随机采样
python sample_vae.py \
  --checkpoint ./outputs/vae_mnist_tiny/checkpoints/best.pt \
  --n_samples 64 \
  --grid_size 8 \
  --outdir ./outputs/vae_mnist_tiny/sample
```

```bash
# 潜空间分析（可选 t-SNE）
python analyze_latent.py \
  --checkpoint ./outputs/vae_mnist_tiny/checkpoints/best.pt \
  --data_root ./data \
  --run_tsne \
  --outdir ./outputs/vae_mnist_tiny/analyze
```

## 2) Map-Latent 电子版（Electronic OLS）

```bash
# MNIST map-electronic 训练（共享 encoder/decoder，identity adapter）
python train_map_electronic.py \
  --config ./configs/map_electronic_mnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_elec_mnist
```

```bash
# CIFAR-10 map-electronic 训练
python train_map_electronic.py \
  --config ./configs/map_electronic_cifar10.yaml \
  --data_root ./data \
  --outdir ./outputs/map_elec_cifar10
```

```bash
# FashionMNIST map-electronic 训练
python train_map_electronic.py \
  --config ./configs/map_electronic_fashionmnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_elec_fashionmnist
```

```bash
# map-electronic 采样
python sample_map_electronic.py \
  --config ./configs/map_electronic_mnist.yaml \
  --checkpoint ./outputs/map_elec_mnist/checkpoints/best.pt \
  --outdir ./outputs/map_elec_mnist/sample
```

## 3) Map-Latent 光学版（Optical OLS）

```bash
# MNIST map-optical 训练（两层200x200衍射层 -> 静态散射 -> mu_w + 固定sigma采样）
python train_map_optical.py \
  --config ./configs/map_optical_mnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_opt_mnist
```

```bash
# CIFAR-10 map-optical 训练
python train_map_optical.py \
  --config ./configs/map_optical_cifar10.yaml \
  --data_root ./data \
  --outdir ./outputs/map_opt_cifar10
```

```bash
# FashionMNIST map-optical 训练
python train_map_optical.py \
  --config ./configs/map_optical_fashionmnist.yaml \
  --data_root ./data \
  --outdir ./outputs/map_opt_fashionmnist
```

```bash
# map-optical 采样
python sample_map_optical.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --outdir ./outputs/map_opt_mnist/sample \
  --n_samples 64 \
  --grid_size 8
```

```bash
# map 模型统一评估（mode: electronic / optical）
python eval_map.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --mode optical \
  --data_root ./data \
  --outdir ./outputs/map_opt_mnist/eval
```

```bash
# map 潜空间分析（光学链路会统计 latent_intensity_map 的 KL_w）
python analyze_map.py \
  --config ./configs/map_optical_mnist.yaml \
  --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt \
  --data_root ./data \
  --run_tsne \
  --outdir ./outputs/map_opt_mnist/analyze
```

## 4) 常用覆盖参数（不改 YAML 快速试验）

```bash
# 覆盖 epochs / batch_size / lr / seed / outdir（其余仍走 YAML）
python train_map_optical.py \
  --config ./configs/map_optical_mnist.yaml \
  --data_root ./data \
  --epochs 10 \
  --batch_size 64 \
  --lr 5e-4 \
  --seed 123 \
  --outdir ./outputs/map_opt_mnist_quick
```

## 5) 结果目录速查

```bash
# 查看最近输出
ls -lt ./outputs | head

# 训练日志
ls ./outputs/map_opt_mnist/logs

# 核心结果文件
ls ./outputs/map_opt_mnist/{history.csv,results.json,summary.csv}
```

## 6) 一键顺序跑（示例：MNIST optical train -> eval -> sample -> analyze）

```bash
python train_map_optical.py --config ./configs/map_optical_mnist.yaml --data_root ./data --outdir ./outputs/map_opt_mnist && \
python eval_map.py --config ./configs/map_optical_mnist.yaml --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt --mode optical --data_root ./data --outdir ./outputs/map_opt_mnist/eval && \
python sample_map_optical.py --config ./configs/map_optical_mnist.yaml --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt --outdir ./outputs/map_opt_mnist/sample && \
python analyze_map.py --config ./configs/map_optical_mnist.yaml --checkpoint ./outputs/map_opt_mnist/checkpoints/best.pt --data_root ./data --outdir ./outputs/map_opt_mnist/analyze
```

## 7) 备注

- `pad/unpad` 在 `angular_spectrum_propagate` 内部完成。
- 当前默认光学 encoder：`(prop+phase)x2` + `scatter(static)` + `sensor pooling`。
- 当前 `map_optical_*.yaml` 默认已设 `model.arch: pure_optical`，decoder 为 4 层衍射相位链路（无卷积）。
- KL 作用在散射后潜空间均值 `mu_w`，posterior 方差由 `loss.posterior_sigma` 固定。
- `sample_map_optical.py` 直接从先验 `P(z)=N(m0, prior_sigma0^2)` 采样后送 decoder（不再走光学链路）。
- 推荐优先用 `pool_kernel == pool_stride` 直接把 `resize_hw` 降到 `latent_hw`，不做额外数字后处理缩放。
- 若要关闭池化，设 `pool_type=none`，并把 `latent_hw` 设成与解 padding 后光场一致的尺寸。

## 8) 散射介质记忆效应测试（新）

```bash
# 传播距离 a -> 散射介质 -> 传播距离 b，比较三种散射建模与相关长度/NA 对记忆效应的影响
python optics/test_scattering_memory.py \
  --outdir ./outputs/optics_memory_demo \
  --distance_a_mm 20 \
  --distance_b_mm 5 \
  --corr_lens_px 1.0,3.0,6.0 \
  --na_values 0.08,0.15,0.25 \
  --n_tilts 11 \
  --tilt_max_cpp 0.03
```

```bash
# 关闭 bandlimit 或改成动态散射（ablation）
python optics/test_scattering_memory.py \
  --outdir ./outputs/optics_memory_ablation \
  --no_bandlimit \
  --dynamic_scatter \
  --corr_lens_px 0.5,2.0,5.0
```
