#将语义分割标注图像显示为彩色图像
#https://blog.csdn.net/u010103202/article/details/82858592


import numpy as np

def create_visual_anno(anno):
    """
    语义分割的标注图像其灰度值表示其分类类别，因此存储下来以后不能直观查看（灰度值都很小，如1， 2，3，…）。为了直观地查看标注图像，可生成一张对应的彩色图像，函数如下：
    """
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno