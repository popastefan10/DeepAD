# DeepAD

A PyTorch implementation of "Anomaly Detection using Deep Learning based Image Completion". This paper was published at the 17th IEEE International Conference on Machine Learning and Applications (ICMLA). You can find it on [arXiv](https://arxiv.org/abs/1811.06861).

## Run locally

First step is to install the dependencies. On my machine I have used a conda environment, but the project can be run 
with venv or without any environment at all.

```bash
# Use only one option from below

# Install packages within a Conda environment
$ conda create -n deep-ad -f environment.yml

# Install packages within a virtual environment
$ pip install virtualenv
$ virtualenv deep-ad
$ source deep-ad/bin/activate
(deep-ad) $ pip install -r requirements.txt

# Install packages globally (not recommended)
$ pip install -r requirements.txt
```

In order to be able to use modules from `src/deep_ad` inside notebooks we need to install the project. For development
purposes use `--editable/-e`.

```bash
python -m pip install .
# OR
python -m pip install -e .
```