# CartoonGAN
a simple PyTorch implementation of CartoonGAN (CVPR 2018)

[![](https://img.shields.io/badge/Python-3.7-yellow)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-1.3.1-brightgreen)](https://github.com/pytorch/pytorch)
[![](https://img.shields.io/badge/Numpy-1.15.1-red)](https://github.com/numpy/numpy/)
[![](https://img.shields.io/badge/Cv2-4.1.2-blue)](https://github.com/opencv/opencv)
[![](https://img.shields.io/badge/CUDA-8.0-orange)](https://developer.nvidia.com/cuda-downloads)

## Usage

### 1.prepare dataset

The folder structure is as follows
```
├── data
│   ├── train # train real images
│   ├── test  # test real images 
│   ├── ShinKai # cartoon images
│   |── ...
│
├── outputs # generated images
│
├── main.py # training code
├── utils.py
├── config.py # some configs
```

### 2.preprocess cartoon images

```bash
python utils.py your_cartoon_image
```

### 3.train

```bash
python main.py --cartoon_name your_cartoon_image # yon can see more arguments in config.py
```

## Resutls

### 1.

<img src = './asserts/real/42.jpg' height = '320' width = '640'>

<center><h5>input</h5></center>

<img src = './asserts/Demon_Slayer/5.png' height = '320' width = '640'>

<center><h5>ufotable</h5></center>

<img src = './asserts/Kyo_Ani/46.png' height = '320' width = '640'>

<center><h5>Kyoto Animation</h5></center>

### 2.

<img src = './asserts/real/2.jpg' height = '320' width = '640'>

<center><h5>input</h5></center>

<img src = './asserts/Demon_Slayer/43.png' height = '320' width = '640'>

<center><h5>ufotable</h5></center>

<img src = './asserts/Kyo_Ani/10.png' height = '320' width = '640'>

<center><h5>Kyoto Animation</h5></center>

### 3.

<img src = './asserts/real/22.jpg' height = '320' width = '640'>

<center><h5>input</h5></center>

<img src = './asserts/Demon_Slayer/42.png' height = '320' width = '640'>

<center><h5>ufotable</h5></center>

<img src = './asserts/Kyo_Ani/24.png' height = '320' width = '640'>

<center><h5>Kyoto Animation</h5></center>

### 4.

<img src = './asserts/real/4.jpg' height = '320' width = '640'>

<center><h5>input</h5></center>

<img src = './asserts/Demon_Slayer/34.png' height = '320' width = '640'>

<center><h5>ufotable</h5></center>

<img src = './asserts/Kyo_Ani/5.png' height = '320' width = '640'>

<center><h5>Kyoto Animation</h5></center>

## Reference

[CartoonGAN-CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

[pytorch-CartoonGAN by znxlwm](https://github.com/znxlwm/pytorch-CartoonGAN)