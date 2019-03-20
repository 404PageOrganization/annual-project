## 中文

### 识别用户的手写字体风格，并进行风格迁移

使用生成网络来完成字体风格迁移（注：本项目仍处于测试阶段，模型结构还有待调整）

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

-   fake_img 存储训练过程中的生成图像
-   fake_img_mosaicking 存储拼合产生的进度图
-   fonts_reserve 存储所有备用字体文件
-   model_data 存储风格识别网络模型
-   raw_fonts 存储训练所用的源字体文件
-   target_fonts 存储目标字体
-   target_img 存储目标字体图像
-   target_img_origin 存储未切分的目标字体图像

#### 项目各流程文件

##### 预处理

-   characters.txt 存放最常用的3500个汉字
-   font_reference.md 字体引用文档
-   fonts_sifting.py 筛选字符完整的字体
-   image_preprocessing.py 进行字体图像的切分（注：此脚本在应用阶段使用）
-   list_fonts.py 生成字体引用文档
-   make_dirs.py 创建项目所需要的子目录
-   raw_font_to_png.py 调用 raw_fonts 目录下的字体，在 raw_img 子目录下生成所需要的源字体图像
-   target_font_to_png.py 调用 target_fonts 文件夹下的第一个字体，在 target_img 子目录下生成所需要的目标字体图像（注：此脚本在测试阶段使用）

##### 网络及其训练

-   custom_layers.py 自定义向量层
-   fonts_name.dat 以二进制格式存储风格判别网络训练时读入的字体名称
-   style_discrimination.py 风格判别网络训练
-   style_prediction.py 使用风格判别网络预测字体风格
-   training_generator.py 训练风格迁移网络
-   transfer_unknow_characters.py 迁移未知字符

##### 后续工作

-   image_mosaicking.py 在风格迁移网络生成训练结果后，制作一张训练进度图
