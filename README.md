# LSFMD

PyTorch implementation of **[Long Short-Term Fusion by Multi-Scale Distillation for Screen Content Video Quality Enhancement](https://doi.org/10.1109/TCSVT.2025.3544314)**, published in IEEE Transactions on Circuits and Systems for Video Technology, 2025.

The model is implemented in `LSFMv2.py`. This repository includes the QP37 pretrained model at iteration 300000, training and evaluation scripts, and experiment logs.

## Installation

Use Python with PyTorch and an NVIDIA CUDA GPU.

```bash
pip install -r requirements.txt
```

## Project structure

```text
LSFMv2.py                   # Model
train.py                    # Training
test_fyp.py                 # Full-frame PSNR evaluation
test_HAWTssim.py            # Full-frame SSIM evaluation
create_lmdb_mfqev2.py       # Training data preparation
LSFMv2_QP37.yml             # Training configuration
LSFMv2_QP37_test.yml        # Evaluation configuration
dataset/                   # Data loaders
utils/                     # Utilities
exp/                       # Pretrained model and logs
```

## Data preparation

Set the absolute dataset paths in `LSFMv2_QP37.yml` and `LSFMv2_QP37_test.yml`, replacing `/path/to/dataset/` with your data directory.

```text
/path/to/dataset/
├── train/raw/
├── train/QP37/
├── scc_LD_HAWTgt37.lmdb/
├── scc_LD_HAWTlq37.lmdb/
├── eval/raw/
├── eval/QP37/
├── test/raw/
└── test/QP37/
```

Prepare paired raw and QP37-compressed training videos, then build the LMDB databases:

```bash
python create_lmdb_mfqev2.py --opt_path LSFMv2_QP37.yml
```

The data preparation script reads 8-bit YUV420p; validation and evaluation read 8-bit YUV444p. Training videos are paired by sorted filename order, using up to 300 frames per video.

For validation and evaluation, compressed filenames use the prefix `rec`:

```text
Raw:        seq_1920x1080_30_8bit_300_444.yuv
Compressed: recseq_1920x1080_30_8bit_300_444.yuv
```

## Training

```bash
python train.py --opt_path LSFMv2_QP37.yml
```

The configuration uses 128×128 crops, Adam with a learning rate of `1e-4`, Charbonnier loss, and 300000 iterations. The batch size is 4 per GPU. Checkpoints and training logs are saved in `exp/SCC_LD_LSFMv2_QP37_train/`.

## Evaluation

The QP37 checkpoint is located at:

```text
exp/SCC_LD_LSFMv2_QP37_enlarg300x/ckp_300000.pt
```

Run full-frame evaluation:

```bash
# PSNR
python test_fyp.py --opt_path LSFMv2_QP37_test.yml

# SSIM
python test_HAWTssim.py --opt_path LSFMv2_QP37_test.yml
```

Results are saved as `log_test_psnr.log` and `log_test_ssim.log` in the checkpoint directory. PSNR is measured in dB; SSIM is dimensionless.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{huang2025lsfmd,
  author  = {Ziyin Huang and Yui-Lam Chan and Ngai-Wing Kwong and Sik-Ho Tsang and Kin-Man Lam and Wing-Kuen Ling},
  title   = {Long Short-Term Fusion by Multi-Scale Distillation for Screen Content Video Quality Enhancement},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2025},
  volume  = {35},
  number  = {8},
  pages   = {7762--7777},
  month   = aug,
  doi     = {10.1109/TCSVT.2025.3544314}
}
```

## Acknowledgments

The training and data-processing utilities build on [STDF-PyTorch](https://github.com/ryanxingql/stdf-pytorch). We thank its authors for sharing their code.
