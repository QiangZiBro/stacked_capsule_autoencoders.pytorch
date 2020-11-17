# Stacked  Capsule Autoencoders pytorch implementation

Unofficial pytorch implementation of paper:[stacked capsule autoencoders](https://arxiv.org/abs/1906.06818). Still at actively developing, welcome to join to discuss!
This repository aims to:
- reproducing original paper in pytorch with recent reversion(1.7)
- writing structured and readable code
- to be closed to original implementation as much as possible

There are also some other pytorch implementations such as [this](https://github.com/phanideepgampa/stacked-capsule-networks),
[this](https://github.com/MuhammadMomin93/Stacked-Capsule-Autoencoders-PyTorch) and
[this](https://github.com/Axquaris/StackedCapsuleAutoencoders).

## Progress
- [x] CCAE model
- [x] CCAE loss stuff
- [x] CCAE pytorch dataloader
- [ ] CCAE training script
- [ ] PCAE
- [ ] OCAE
- [ ] SCAE loss stuff
- [ ] SCAE training script
- [ ] possible visulization for  CCAE and SCAE

## Installation
Based on pytorch 1.7 but you can use it older than this version, recommend 1.2~1.7.
```bash
git clone https://github.com/QiangZiBro/stacked_capsule_autoencoders.pytorch
pip install requirements.txt
pip install requirements-dev.txt # for debug or coding quality stuff
```



## Reference
- [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template) for deep learning project template, and this project is built on it.
- [stacked_capsule_autoencoders](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders) original tensorflow version