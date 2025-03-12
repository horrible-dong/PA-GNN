基于链路聚合的图欺诈检测 (Path Aggregation-Based Graph Fraud Detection)
========

[![journal](https://img.shields.io/badge/Journal-软件学报_(Journal_of_Software)-ff69b4)](https://www.jos.org.cn)
&emsp;[![code-framework](https://img.shields.io/badge/Code_Framework-QTClassification_v0.9.1-brightgreen)](https://github.com/horrible-dong/QTClassification)
&emsp;[![doc](https://img.shields.io/badge/Docs-Latest-orange)](README.md)
&emsp;[![license](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

> Authors: Tian Qiu, Lingxiang Jia, Yang Gao, Zunlei Feng, Yi Gao, Mingli Song  
> Affiliation: Zhejiang University

## Installation

The development environment of this project is `python 3.8 & pytorch 1.13.1+cu117 & dgl 1.1.3+cu117`.

1. Create your conda environment.

```bash
conda create -n qtcls python==3.8 -y
```

2. Enter your conda environment.

```bash
conda activate qtcls
```

3. Install PyTorch.

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Or you can refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) to install newer or older versions.
Please note that if pytorch ≥ 1.13, then python ≥ 3.7.2 is required.

4. Install DGL.

```bash
pip install dgl==1.1.3+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-geometric
```

5. Install necessary dependencies.

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the zip file from [[百度网盘]](https://pan.baidu.com/s/1N0CBnaA1ygQOu3yGmu6YWw?pwd=xqk9) / [[Google Drive]](https://drive.google.com/file/d/1X8wjxn2-ebW7jud21vuawk7UtCOW5KtD/view?usp=sharing) and put the file into `data/raw`.

2. Unzip the file.

```bash
cd data/raw
unzip amazon_elliptic_tfinance_tsocial_yelpchi.zip
cd ../..
```

## Training

Import the config file (.py) from [configs](configs).

```bash
python main.py --config /path/to/config.py
```

or

```bash
python main.py -c /path/to/config.py
```

During training, the config file, checkpoints (.pth), logs, and other outputs will be stored in `--output_dir`.

## Evaluation

```bash
python main.py --config /path/to/config.py --resume /path/to/checkpoint.pth --eval
```

or

```bash
python main.py -c /path/to/config.py -r /path/to/checkpoint.pth --eval
```

## License

Our code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

Copyright (c) QIU Tian and ZJU-VIPA Lab. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with
the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

## Citation

If you find the paper useful in your research, please consider citing:

```bibtex
Coming soon...
```
