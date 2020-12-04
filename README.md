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
Based on pytorch 1.7 but sure you can use  older  version, 1.2~1.7 recommended.
```bash
git clone https://github.com/QiangZiBro/stacked_capsule_autoencoders.pytorch
pip install requirements.txt
pip install requirements-dev.txt # for debug or coding quality stuff
```


## LICENSE
MIT License

Copyright (c) 2020 qiangzibro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reference
- [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template) for deep learning project template, and this project is built on it.
- [stacked_capsule_autoencoders](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders) original tensorflow version
