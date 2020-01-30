#NYUDv2语义分割可视化结果的colormap设置
#https://blog.csdn.net/u012455577/article/details/86317253

def show_all(gt, pred, index, i_iter):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
 
    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes
 
    classes = np.array(('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                        'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
                        'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
                        'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand',
                        'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
                        'background'))
 
    colormap = [
                (127, 20, 22), (9, 128, 64), (127, 128, 51), (40, 41, 115), (125, 39, 125), (0, 128, 128),
                (127, 127, 127), (57, 16, 18),
                (191, 32, 38), (65, 128, 61), (191, 128, 43), (67, 41, 122), (192, 27, 128), (64, 128, 127),
                (191, 127, 127), (28, 64, 28),
                (127, 66, 28), (47, 180, 74), (127, 192, 66), (29, 67, 126), (128, 64, 127), (47, 183, 127),
                (127, 192, 127), (65, 65, 25),
                (191, 67, 38), (75, 183, 73), (190, 192, 49), (64, 64, 127), (193, 65, 128), (74, 187, 127),
                (192, 192, 127), (11, 17, 60),
                (127, 21, 66), (0, 128, 65), (127, 127, 63), (47, 65, 154), (117, 64, 153), (8, 127, 191),
                (127, 127, 189), (63, 9, 63),
                (0, 0, 0)]
    colormap = list(colormap)
    item = []
    for colormap_item in range(len(colormap)):
        tmp = [j / 255. for j in colormap[colormap_item]]
        item.append(tmp)
    colormap = tuple(item)
    cmap = colors.ListedColormap(colormap)
 
    # bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    #           31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
 
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=40)
 
    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)
 
    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)
 
    # plt.show()
    if not os.path.exists('./NYUresult/'):
        os.makedirs('./NYUresult/')
 
    plt.savefig(os.path.join('NYUresult', str(i_iter) + '_' + str(index) + '.png'))