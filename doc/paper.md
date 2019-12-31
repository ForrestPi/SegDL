https://blog.csdn.net/weixin_38957591/article/details/84558193 医疗图像分割的损失函数
https://blog.csdn.net/m0_37477175/article/details/83004746 从loss处理图像分割中类别极度不均衡的状况---keras

https://arxiv.org/pdf/1707.03237.pdf  Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations

https://blog.csdn.net/CaiDaoqing/article/details/90457197 语义分割常用loss介绍及pytorch实现

https://arxiv.org/abs/1708.02551 Semantic Instance Segmentation with a Discriminative Loss Function

https://github.com/ShawnBIT/UNet-family


https://github.com/wasidennis/AdaptSegNet  Learning to Adapt Structured Output Space for Semantic Segmentation, CVPR 2018 (spotlight)

https://arxiv.org/pdf/1812.07032.pdf  Boundary loss for highly unbalanced segmentation
https://zhuanlan.zhihu.com/p/72783363 一票难求的MIDL 2019 Day 1-Boundary loss
作者的论文中在两个label不平衡的数据集上做了实验，结果表明：只用本文提出的boundary loss不work，和Dice loss组合在一起效果最好。但在今天的presentation中作者右提到，在心脏数据集上boundary loss可以单独使用。由此可以看出，数据集的选择对本文提出的loss影响很大。

https://zhuanlan.zhihu.com/p/50539347 图像分割中从Loss上解决数据集imbalance的方法

https://zhuanlan.zhihu.com/p/44958351 研习U-Net

https://niftynet.readthedocs.io/en/dev/  NiftyNet is a TensorFlow-based open-source convolutional neural networks platform for research in medical image analysis and image-guided therapy


https://zhuanlan.zhihu.com/p/85897057  【ICCV 2019】图像分割论文

https://github.com/xiaoketongxue/AI-News

https://blog.csdn.net/baidu_27643275/article/details/101524444 

https://blog.csdn.net/baidu_27643275/article/details/99683222 Portrait分割】BANet:Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation

https://researchcode.com/code/2908487688/learning-a-discriminative-feature-network-for-semantic-segmentation/

https://github.com/lxtGH/GALD-Net

https://github.com/lxtGH/Fast_Seg

https://github.com/speedinghzl/CCNet