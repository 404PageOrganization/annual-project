# 中文手写字体风格迁移

## 简介

本项目的目的是使用深度学习，实现不同手写字体之间的风格迁移。

> 注：由于拖延症等缘故，本项目正处于并可能长期处于测试阶段，模型还有待调整。

## 安装

> 注：本项目仅在 Python 3.6.8 下测试通过。

安装项目所需的环境：

```bash
pip install pillow colorama
```

如果你使用 CPU 进行训练（不推荐），可以安装 CPU 版本的 PyTorch：

```bash
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```

如果你的电脑有较新的 NVIDIA 显卡，可以安装 CUDA10.0 与 CUDNN 来使用 GPU 加速，并且安装以下版本的 PyTorch：

```bash
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```

## 使用流程

> 注：当前项目处于测试阶段，最终应用阶段使用流程会与目前不同。

1. 创建目录结构

```bash
python3 make_dirs.py
```

2. 选择要使用的字体

将源字体文件放在 `/raw_fonts` 目录下，将目标字体文件放在 `/target_fonts` 目录下。

3. 生成目标字体

> 注：此步骤仅在测试阶段使用，应用阶段将变为读取用户手写字体。

```bash
python3 [test]target_font_to_png.py
```

4. 训练模型

```bash
python3 train_model.py
```

## 目录结构

-   fake_img 存储训练过程中的生成图像
-   fake_img_mosaicking 存储拼合产生的进度图
-   fonts_reserve 存储所有备用字体文件
-   model_data 存储风格识别网络模型
-   output_img 保存生成的图片结果
-   raw_fonts 存储训练所用的源字体文件
-   target_fonts 存储目标字体（注：此目录仅在测试阶段使用）
-   target_img 存储目标字体图像
-   target_img_origin 存储未切分的目标字体图像


## 项目各流程所使用的文件

### 预处理环节

-   [test]target_font_to_png.py 调用 target_fonts 文件夹下的第一个字体，在 target_img 子目录下生成所需要的目标字体图像（注：此脚本仅在测试阶段使用）
-   fonts_sifting.py 筛选字符完整的字体
-   image_preprocessing.py 进行字体图像的切分（注：此脚本在应用阶段使用）
-   make_dirs.py 创建项目所需要的子目录

### 网络训练环节

-   characters.txt 存放最常用的 3500 个汉字
-   train_model.py 训练网络
-   lib/custom_layers.py 自定义向量层
-   lib/image_mosaicking.py 在网络生成训练结果后，制作一张训练进度图

### 生成环节

-   generate_words.py 生成文章图像
-   test.txt 存放需要生成的文章

## 贡献者

[@cometeme](https://github.com/cometeme)

[@mzzx](https://github.com/mzzx)