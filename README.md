<div align="center">

# Differentiable Computational Lithogrpahy Framework


<img src="./misc/figs/torchlitho.png" width="80%">


[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
<br>

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
[![contributors](https://img.shields.io/github/contributors/TorchOPC/TorchLitho.svg)](https://github.com/TorchOPC/TorchLitho/graphs/contributors)


</div>

## TL;DR

- **First** open-source differentiable Abbe lithography imaging model. (With Hopkins / TCC model included)
- **First** open-source analytical resist development model.
- **First** open-source vector lithography implementation (work in progress).


## Description


The rapid evolution of the electronics industry, driven by Moore's law and the proliferation of integrated circuits, has led to significant advancements in modern society, including the Internet, wireless communication, and artificial intelligence (AI). Central to this progress is optical lithography, a critical technology in semiconductor manufacturing that accounts for approximately 30\% to 40\% of production costs. As semiconductor nodes shrink and transistor numbers increase, optical lithography becomes increasingly vital in current integrated circuit (IC) fabrication technology. This repo introduces an open-source differentiable lithography imaging framework that leverages the principles of differentiable programming and the computational power of GPUs to enhance the precision of lithography modeling and simplify the optimization of resolution enhancement techniques (RETs). The framework models the core components of lithography as differentiable segments, allowing for the implementation of standard scalar imaging models, including the Abbe and Hopkins models, as well as their approximation models.

[**Introduction slides**](./misc/slides/torchlitho.pdf)

## Notes

This repo is still under construction. 🚧

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/TorchOPC/TorchLitho
cd TorchLitho

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

<details><summary>Using Conda</summary>

```bash
# clone project
git clone https://github.com/TorchOPC/TorchLitho
cd TorchLitho

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```
</details>




## How to run

Train model with default configuration

```bash
# Test the imaging
cd src/models/litho
python ImagingModel.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```



## Cited as



```txt
@inproceedings{SPIE24_difflitho,
  title     = {Open-Source Differentiable Lithography Imaging Framework},
  author    = {Chen, Guojin and Geng, Hao and Yu, Bei and Pan, David Z.},
  booktitle = {Advanced Lithography + Patterning},
  organization = {International Society for Optics and Photonics},
  publisher    = {SPIE},
  year      = {2024},
  month     = {3},
}
```




## Releated work

[Guojin Chen](https://gjchen.me/), Hao Geng, Bei Yu, David Z. Pan, “[Open-Source Differentiable Lithography Imaging Framework](https://www.cse.cuhk.edu.hk/~byu/papers/C201-SPIE2024-OpenLitho.pdf)”, SPIE Advanced Lithography + Patterning, San Jose, Feb. 25–29, 2024. ([paper](https://www.cse.cuhk.edu.hk/~byu/papers/C201-SPIE2024-OpenLitho.pdf)) ([slides](https://www.cse.cuhk.edu.hk/~byu/papers/C201-SPIE2024-OpenLitho-slides.pdf)) 

[Guojin Chen](https://gjchen.me/), Hongquan He, Peng Xu, Hao Geng, Bei Yu, “Efficient Bilevel Source Mask Optimization”, ACM/IEEE Design Automation Conference (DAC), San Francisco, Jun. 23–27, 2024.

```txt
@inproceedings{DAC24_BiSMO,
  title     = {Efficient Bilevel Source Mask Optimization},
  author    = {Chen, Guojin and He, Hongquan and Xu, Peng and Geng, Hao and Yu, Bei},
  booktitle = {2024 61th ACM/IEEE Design Automation Conference (DAC)},
  pages     = {1-6},
  year      = {2024},
  month     = {7},
  organization = {IEEE},
}
```

[Guojin Chen](https://gjchen.me/), Zixiao Wang, Bei Yu, David Z. Pan, Martin D.F. Wong, “[Ultra-Fast Source Mask Optimization via Conditional Discrete Diffusion](https://ieeexplore.ieee.org/document/10419017)”, accepted by IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD).


```txt
@ARTICLE{DiffSMO-chen-2024,
  author={Chen, Guojin and Wang, Zixiao and Yu, Bei and Pan, David Z. and Wong, Martin D.F.},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={Ultra-Fast Source Mask Optimization via Conditional Discrete Diffusion},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Optimization;Lithography;Lighting;Integrated circuits;Graphics processing units;Optical imaging;Resists},
  doi={10.1109/TCAD.2024.3361400}
}
```
<br>



## Resources

OPC : [OpenILT](https://github.com/OpenOPC/OpenILT)

<br>

## License

TorchLitho is licensed under the [GPL3.0 License](./LICENSE).