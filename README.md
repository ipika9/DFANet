<div align="center">
<h1>CFAI-NETt</h1>
</div>

## Installation

We recommend the following platforms.
In addition, you need to install the necessary packages using the following instructions.
Please leave a message for any questions.
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

You should download the pretrained VMamba-Tiny V2 model (vssm1_tiny_0230s_ckpt_epoch_264) from [VMamba](https://github.com/MzeroMiko/VMamba/releases/), and then put it in the `model/pretrain/` folder for initialization.

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

