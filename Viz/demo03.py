#pytorch中语义分割最终的one hot结果转为color image
#https://blog.csdn.net/weixin_39610043/article/details/93851359
import numpy as np
import torch

def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1

def create_pascal_label_colormap(class_num):
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((class_num, 3), dtype=int)
    ind = np.arange(class_num, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


n_classes = 10
color = create_pascal_label_colormap(n_classes)
print(color)

#众所周知，pytorch的tensor结构为[batch,channel,height,width],因此而我们日常rgb等结构为[height,width,channel]，因此我们首先要进行维度转换再做color化
def to_color_img(img):
    #下面的0代表batch的第0个元素
    score_i = img[0,...]
    score_i = score_i.cpu().numpy()
    #转换通道
    score_i = np.transpose(score_i,(1,2,0))
    # one hot转一个channel
    score_i = np.argmax(score_i,axis=2)
    #color为上面生成的color list
    color_img = color[score_i]
    return color_img
#最终调用方式，很简单了
predict = model(input_img)
to_color_img(predict)

#自己用的类
import numpy as np

class label2color(object):
    def __init__(self,class_num):
        self.class_num = class_num

        self.colors = self.create_pascal_label_colormap(self.class_num)

    def to_color_img(self,imgs):
        # img:bs,3,height,width
        color_imgs = []
        for i in range(imgs.shape[0]):
            score_i = imgs[i,...]
            score_i = score_i.cpu().numpy()
            score_i = np.transpose(score_i,(1,2,0))
            # np.save('pre.npy',score_i)
            score_i = np.argmax(score_i,axis=2)
            color_imgs.append(self.colors[score_i])
        return color_imgs
    def single_img_color(self,img):
        score_i = img
        score_i = score_i.cpu().numpy()
        score_i = np.transpose(score_i,(1,2,0))
        # np.save('pre.npy',score_i)
        score_i = np.argmax(score_i,axis=2)
        return self.colors[score_i]


    def bit_get(self,val, idx):
        """Gets the bit value.
        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.
        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    def create_pascal_label_colormap(self,class_num):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
          A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((class_num, 3), dtype=int)
        ind = np.arange(class_num, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= self.bit_get(ind, channel) << shift
            ind >>= 3

        return colormap

#调用方法:
l = label2color(5)
label_color = l.single_img_color(c)
