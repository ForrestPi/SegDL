#可视化语义分割结果
#https://blog.csdn.net/fanzonghao/article/details/85160499
def test_visul_label():
    path='./train_annot'
    output='./train_annot_out'
    if not os.path.exists(output):
        os.mkdir(output)
    label_path=[os.path.join(path,i) for i in os.listdir(path)]
    for label_path_ in label_path:
        label = cv2.imread(label_path_)
        print(label.shape)
        print(type(label))
        label =label[:, :, 0]
        cmap = np.array([[0, 0, 0],
        [128, 0, 0],
        [128, 128, 0],
        [0, 128, 0],
        [0, 0, 128]]
        )
        y = label
        r = y.copy()
        g = y.copy()
        b = y.copy()
        # print('r=',r)
        for l in range(0, len(cmap)):
            r[y == l] = cmap[l, 0]
            g[y == l] = cmap[l, 1]
            b[y == l] = cmap[l, 2]
        label=np.concatenate((np.expand_dims(b,axis=-1),np.expand_dims(g,axis=-1),
                              np.expand_dims(r,axis=-1)),axis=-1)
 
        cv2.imwrite(output+'/'+label_path_.split('/')[-1],label)