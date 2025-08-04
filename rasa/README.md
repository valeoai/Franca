# RASA: Removal of Absolute Spatial Attributes

**RASA** is a lightweight, iterative post-processing module that removes absolute spatial bias from Vision Transformer (ViT) patch embeddings. It enhances semantic understanding by disentangling positional information—especially useful in self-supervised learning setups where location labels are absent.

---

## Highlights

- Works **post-hoc** on pretrained ViTs, like Franca
- **Iterative debiasing** using Gram-Schmidt orthogonal projections
- Modular `RASAHead` integrates into any ViT pipeline
- Boosts **semantic clustering** and **spatial entropy**
- Compatible with [FRANCA](https://github.com/valeoai/franca) training framework
- Built with PyTorch & PyTorch Lightning

---

## Method Overview

RASA learns to predict normalized 2D patch coordinates from ViT embeddings. It then removes the components in feature space that are most predictive of position.

1. Train a simple linear regressor to predict 2D patch coordinates.
2. Orthonormalize its weight vectors using Gram-Schmidt.
3. Project patch embeddings onto this 2D subspace and subtract it.
4. Repeat iteratively until position-predictive signal vanishes.

```math
Z^{(t)}_s = Z^{(t-1)}_s - \left\langle Z_s, u_r \right\rangle u_r - \left\langle Z_s, u_c \right\rangle u_c
```

---

## Installation

To install RASA seperately from Franca's repo,

```bash
 pip install -e .
```

---

## Setup

### Configuration

To set up training, create or adapt a configuration file.
An example is provided at:

```
./rasa/experiments/configs/rasa_franca.yml
```

Modify parameters such as:
- when `only_load_weights` is `True`:
    - `train.checkpoint`: path to pretrained franca backbone
- when `only_load_weights` is `False`:
    - `train.checkpoint`: path to checkpoint of RASA lightning module to continue training
    - In this case `start_pos_layers` can also be adapted to continue training of the rest of linear layers within the RASAhead.
- Learning rate, weight decay, epochs, etc.

### Dataset Preparation

To prepare your dataset for training RASA and evaluation, refer to [dataset_rasa.md](rasa/dataset_rasa.md).

---

## Training

Once your config and dataset are ready, start training:

```bash
cd rasa
python experiments/train_rasa.py --config_path ./rasa/experiments/configs/rasa_franca.yml
```

---

## Loading Trained RASA Heads

### Use PyTorch Lightning wrapper

```python
from src.rasa import RASA

model = RASA.load_from_checkpoint("/path/to/pytorch_lightning_chkpt/rasa_model.ckpt", map_location="cpu")
# Load Images, here we Create Random Ones
batch = torch.randn(32, 3, 224, 224)
# Apply FRANCA First to get the patch embeddings
x = model.backbone.forward_features(batch)["x_norm_patchtokens"]
# Apply RASA Second to remove the positional information
x_debiased = model.head(x, use_pos_pred=True, return_pos_info=False)
```

---

## Evaluation

RASA includes multiple evaluation strategies:

- **How well can embeddings predict position?**
- **How position-dependent are features before/after debiasing?**
- **Unsupervised segmentation (mIoU) on patch clustering**
- **Entropy of patch activations across spatial locations**


Evaluation is handled inside the `RASA` LightningModule.

---

## Project Structure

```
rasa/
├── src/
│   ├── rasa.py             # LightningModule
│   ├── transforms.py       # Data Pipeline Transformations
│   └── rasa_head.py        # RASAHead torch Module
├── experiments/
│   ├── train_rasa.py       # Training entrypoint
│   ├── utils.py            # Training Utils
│   └── configs/
├── data/                   # Setting Datasets Modules
│   ├── coco/               # COCO
│   ├── imagenet/           # Imagenet 1k and 100
│   ├── VOCdevkit/          # Pascal VOC
│   └── utils.py            # Data Handling Utils
└── dataset_rasa.md         # Dataset setup instructions
```

---