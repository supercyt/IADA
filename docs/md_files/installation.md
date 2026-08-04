# Installation

* [__System/Hardware Requirements__](#requirements)
* [__Installation__](#installation)
    * [__1. Dependency Installation__](#1-dependency-installation)
    * [__2. Install Pytorch__](#2-pytorch-installation-18)
    * [__3. Install Spconv__](#3-spconv-121-requred)




---
## System/Hardware Requirements
To get started, the following requirements should be fulfilled.
* __System requirements.__ OpenCOOD is tested under Ubuntu 18.04
* __Adequate GPU.__ A minimum of 6GB gpu is recommended.
* __Disk Space.__ Estimate 100GB of space is recommended for data downoading.
* __Python__ Python3.7 is required.


---
## Installation
### 1. Install uv and the project
First, download OpenCOOD github to your local folder if you haven't done it yet.
```sh
git clone https://github.com/DerrickXuNu/OpenCOOD.git
cd OpenCOOD
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the Python 3.8 environment and install the locked project dependencies:

```bash
uv python install 3.8
uv sync
```

`uv sync` creates `.venv`, installs this project in editable mode, and installs
PyTorch 2.4.1/torchvision 0.19.1/torchaudio 2.4.1 from the CUDA 12.1 wheel
index configured in `pyproject.toml`. Run project commands without activating
the environment:

```bash
uv run python your_script.py
```

To activate it in the current shell instead:

```bash
source .venv/bin/activate
```

### 2. PyTorch and CUDA

The default environment uses the official PyTorch CUDA 12.1 wheels. The NVIDIA
driver must support CUDA 12.1. A system CUDA toolkit is only required when
compiling CUDA extensions; check it with `nvcc --version`.

### 3. Spconv 2.x

The default dependency is `spconv-cu120`, which is the spconv build for CUDA
12.x. If your machine uses another CUDA major version, replace this package in
`pyproject.toml` with the matching package from the
[spconv installation table](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library),
then regenerate the lockfile with `uv lock`.

### 4. Bbx IOU cuda version compile
Install bbx nms calculation cuda version
  
```bash
uv run python opencood/utils/setup.py build_ext --inplace
```

After changing dependencies, run `uv lock`. On another machine, `uv sync
--frozen` installs exactly the versions recorded in `uv.lock`.

