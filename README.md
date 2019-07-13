### 识别用户的手写字体风格，并进行风格迁移

使用生成网络来完成字体风格迁移（注：本项目仍处于测试阶段，模型结构还有待调整）

#### 需要的环境

本项目仅在 python 3.6.8 版本下测试通过

安装项目所需的环境：

```bash
pip install pillow colorama
```

如果你使用 CPU 进行训练（不推荐），可以安装 CPU 版本的 pytorch：

```bash
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```

如果你的电脑有较新的 NVIDIA 显卡，可以安装 CUDA9.0 与 CUDNN 来使用 GPU 加速，并且安装以下版本的 pytorch：

```bash
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```

#### 目录结构

-   fake_img 存储训练过程中的生成图像
-   fake_img_mosaicking 存储拼合产生的进度图
-   fonts_reserve 存储所有备用字体文件
-   model_data 存储风格识别网络模型
-   output_img 保存生成的图片结果
-   raw_fonts 存储训练所用的源字体文件
-   target_fonts 存储目标字体（注：此目录仅在测试阶段使用）
-   target_img 存储目标字体图像
-   target_img_origin 存储未切分的目标字体图像

#### 项目各流程文件

##### 预处理环节

-   [test]target_font_to_png.py 调用 target_fonts 文件夹下的第一个字体，在 target_img 子目录下生成所需要的目标字体图像（注：此脚本仅在测试阶段使用）
-   fonts_sifting.py 筛选字符完整的字体
-   image_preprocessing.py 进行字体图像的切分（注：此脚本在应用阶段使用）
-   make_dirs.py 创建项目所需要的子目录
-   requirements.txt 依赖包

##### 网络训练

-   characters.txt 存放最常用的 3500 个汉字
-   train_model.py 训练网络

##### 生成环节

-   generate_words.py 生成文章图像
-   test.txt 存放需要生成的文章

##### lib

-   custom_layers.py 自定义向量层
-   image_mosaicking.py 在网络生成训练结果后，制作一张训练进度图