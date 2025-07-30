<div align="center">
<h1>DFA-Net: Dual Frequency-Aware Vision Mamba U-Net for Medical Image Segmentation</h1>
</div>

## Abstract

Currently, edge blurring and the loss of structural details remain key challenges that limit the performance of medical image segmentation.Although Vision Mamba demonstrates strong long-range dependency modeling capabilities, it still shows limited sensitivity to high-frequency details in the frequency domain, which often results in weakened fine-grained features and inaccurate boundary reconstruction.To address these issues, we propose a Dual Frequency-Aware Vision Mamba Network (DFA-Net), which introduces two frequency-aware modules into the feature fusion and decoding stages, respectively, to preserve detail features and reconstruct boundaries, forming a cross-stage frequency modeling pathway. Specifically, we design an Adaptive Frequency-Aware Interaction (AFAI) module in the fusion stage, which employs a learnable Laplacian kernel to decouple high- and low-frequency components and enhances frequency-domain representation through a dynamic frequency interaction mechanism.In the decoding stage, we incorporate a Frequency Wavelet Vision State Space (FWVSS) block, which leverages wavelet transform to enhance frequency-band detail perception and integrates multi-scale convolutional aggregation to capture local context for fine-grained reconstruction of blurred boundaries.Extensive experiments on two public medical image segmentation datasets (Synapse and ACDC) demonstrate that DFA-Net outperforms previous state-of-the-art methods, achieving up to 1.31\% improvement in average Dice score, and reaching SOTA performance, showcasing its effectiveness in preserving details and accurately reconstructing blurred boundaries.

## Installation

We recommend the following platforms
In addition, you need to install the necessary packages using the following instructions
And install a runtime environment that supports Mamba: 

```
Python 3.8 / Pytorch 2.0.0 / NVIDIA GeForce RTX 4090 / CUDA 11.8.0 / Ubuntu
pip install -r requirements.txt
cd ./kernels/selective_scan
pip install -e .
```

## Prepare data & Pretrained model

#### Dataset:

- **Synapse Multi-Organ Dataset**: Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset , save in the `dataset/synapse/` folder.
- **ACDC Dataset**: Download the preprocessed ACDC dataset from [TransUNet](https://github.com/Beckschen/TransUNet/tree/main) and move into `dataset/acdc/` folder.

#### ImageNet pretrained model:

You should download the pretrained VMamba-Tiny V2 model (vssm_tiny_0230_ckpt_epoch_262) from [VMamba](https://github.com/MzeroMiko/VMamba/releases/), and then put it in the `model/pretrain/` folder for initialization.

## Training

Using the following command to train & evaluate MSVM-UNet:

```python
# Synapse Multi-Organ Dataset
python train_synapse.py
# ACDC Dataset
python train_acdc.py
```
## test

Please put the trained checkpoints file into the inference.py file for testing:

```python
python inference.py
```

