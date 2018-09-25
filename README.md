## 中文

### 识别用户的手写字体风格，并进行风格迁移

使用纯 CNN 网络来完成风格迁移

#### 需要的环境

-   conda

```bash
conda install keras tensorflow-gpu pillow colorama
```

或者

-   pip

```bash
pip install keras tensorflow tensorflow-gpu pillow colorama
```

如果你的电脑有较新的 NVIDIA 显卡，可以安装 CUDA9.0 与 CUDNN 来使用 GPU 加速。

#### 目录结构

-   characters.txt 存放最常用的3500个汉字
-   custom_layers.py 自定义向量层
-   font_reference.md 字体引用目录
-   image_mosaicking.py 在风格迁移网络生成训练结果后，制作一张进度图
-   image_preprocessing.py 进行字体图像的切分
-   list_fonts.py 生成字体引用文档
-   make_all_characters.py 套用 fonts 文件夹下的所有字体，生成所需要的所有原字符
-   make_dirs.py 创建项目所需要的目录
-   real_font_to_png.py 套用 fonts 文件夹下的第一个字体，生成测试用图片
-   style_discrimination.py 风格识别网络训练
-   style_prediction.py 风格识别网络判别
-   training_generator.py 训练风格迁移网络
-   translate_unknow_characters.py 迁移未知字符
