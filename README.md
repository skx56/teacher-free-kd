# Teacher-free knowledge distillation

Self-distillation for CNNs. There is no external teacher. The network is trained from scratch, and an exponential moving average of its own weights supplies the soft targets.

The training objective has four parts:

- **Label smoothing** on the hard labels, which is a virtual teacher with a fixed smooth distribution.
- **EMA self-teacher**, updated every step from the student. This is the self-training signal.
- **Logit standardization**, a per-sample z-score of the student and teacher logits before the soft loss.
- **Decoupled KL**, split into the labeled class versus the distribution over the other classes.

Supported models are ResNet-18, ResNet-50, ResNeXt-50, MobileNetV2, and ShuffleNetV2. Supported datasets are CIFAR-100 and ImageNet. CIFAR uses a 3x3 stride-1 stem so the 32x32 input is not downsampled away. ImageNet uses the standard torchvision stem. Weights start randomly.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

CIFAR-100, ResNet-18, 200 epochs:

```bash
python train.py --dataset cifar100 --model resnet18 --data data
```

The same command accepts `resnet50`, `resnext50_32x4d`, `mobilenet_v2`, and `shufflenet_v2_x1_0`.

ImageNet expects an ImageFolder layout at `train/` and `val/`:

```bash
python train.py --dataset imagenet --model resnet50 --data /path/to/imagenet --batch-size 256
```

Check that a model can take one optimizer step without downloading data:

```bash
python train.py --dry-run --model resnet18
```

Loss checks, no dataset:

```bash
python -m unittest tests/test_losses.py
```

## Defaults

CIFAR-100 uses SGD at learning rate 0.1, momentum 0.9, weight decay `5e-4`, label smoothing 0.1, temperature 4, target-class weight 1, non-target weight 8, and a 20-epoch warmup before the self-teacher loss turns on. ImageNet uses weight decay `1e-4`, temperature 1, non-target weight 0.5, and a 10-epoch warmup. The EMA decay is 0.999. The best top-1 checkpoint is written to `checkpoints/best.pt`.

Top-1 depends on the run. Compare against plain cross-entropy by training the same model with `--kd-weight 0`.
