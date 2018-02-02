# Finger-Cartoon
检测手指是否指向卡通，准备收集更多训练数据，获得更好效果。
- 前半部分工作在：https://github.com/ZZZZZch/ImgThracker


## 模块一：使用YOLO基于tensorflow的实现——Darkflow
-	官方github：https://github.com/thtrieu/darkflow
-	所需配置  Python3, tensorflow 1.0+, numpy, opencv 3。
初始化程序：
    ```shell
    python3 setup.py build_ext --inplace
    pip install -e .
    pip install .
    ```
- 训练需要下载权值文件与配置cfg文件
- 我上传的文件中自带一个/Finger-Cartoon/darkflow/bin/tiny-yolo-voc.weights权值文件，是小型yolo模型，训练速度快，识别效果较差。与该权值文件相对应的配置文件为/Finger-Cartoon/darkflow/cfg/tiny-yolo-4c.cfg。
- 更多权值文件与配置文件：https://pjreddie.com/darknet/yolo/
- 若自行下载模型权值和配置文件，需要在配置文件中将最后一层卷积的filter数改为35，最后class数量改为2。
- 创建两个文件夹train/Images与train/Annotations，分别放置训练图片与训练标注文件（为模块二中生成的文件）。
```
flow --model cfg/tiny-yolo-voc-4c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images
```
- 或
```
./flow --model cfg/tiny-yolo-voc-4c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images
```
- 生成模型文件保存在ckpt文件下中。
- 测试命令：
```
flow --imgdir ValImgs/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0
```
- 将在测试图片文件夹下创建ValiImgs/out 放置检测结果。
- 已经修改过源代码，添加手指与卡通双目标重叠时进行提示的功能。

## 模块二：使用Label-Image工具进行图片标注
- 官方Git：https://github.com/tzutalin/labelImg
- 运行label_image.py，进入可视化标注工具。
- 标注结果会保存为一个XML文件，该文件需要在模块一中配合使用。
