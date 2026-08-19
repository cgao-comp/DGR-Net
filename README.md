# DGR-Net

Official implementation for **DGR-Net: Dynamic Gating and Retrieval-Enhanced Network for Multimodal Sentiment Analysis**.

## Directory structure

```text
DGR01-13C1_submission/
├── code/                         # Source code and training configuration
├── dataset/
│   └── MOSI/Processed/
│       └── aligned_50.pkl         # CMU-MOSI aligned features
└── README.md
```

## Datasets

- [CMU-MOSI](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)
- [CMU-MOSEI](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)

## Environment

- Python 3.9
- PyTorch >= 1.13 with a CUDA build appropriate for the local GPU/driver
- Other dependencies are listed in `code/requirements.txt`

Create an environment and install dependencies:

```bash
conda create -n dgr python=3.9 -y
conda activate dgr
pip install -r code/requirements.txt
```

The model uses `bert-base-uncased`. With Internet access, it will be downloaded automatically on the first run. For offline use, provide a complete local model directory:

```bash
export DGR_BERT_PATH=/path/to/bert-base-uncased
```

## Training

Run from the `code` directory:

```bash
cd code
python train.py
```

The default configuration trains DGR-Net on MOSI with seed 68. Checkpoints, logs, and results are written to `code/pt`, `code/log`, and `code/result`.

## Citation

```bibtex
@inproceedings{dgrnet2026,
  title     = {DGR-Net: Dynamic Gating and Retrieval-Enhanced Network for Multimodal Sentiment Analysis},
  author    = {Liu, Xiangfeng and Jiang, Shenjie and Wang, Zhuoyu and Li, Xianghua and Zhang, Huixiang and Gao, Chao},
  booktitle = {Proceedings of thPre 35th ACM International Conference on Information and Knowledge Management (CIKM26)},
  publisher = {Association for Computing Machinery},
  pages     = {1--12},
  year      = {2026}
}
```
