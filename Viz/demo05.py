#语义分割之图片和 mask 的可视化
#https://www.aiuai.cn/aifarm276.html
import cv2
import matplotlib.pyplot as plt
imgfile = 'image.jpg'
pngfile = 'mask.png'

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

img = img[:, :, ::-1]
img[..., 2] = np.where(mask == 1, 255, img[..., 2])

plt.imshow(img)
plt.show()