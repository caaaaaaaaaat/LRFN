import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#mean, cov = [0, 0], [(1, .6), (.6, 1)]
#x, y = np.random.multivariate_normal(mean, cov, 100).T
#y += x + 1

#x = np.random.random(50)
#y = np.random.random(50)
root_dir_x = 'F:\paper\my paper\paper2\images\Plot\SCDCN.csv'
root_dir_y = 'F:\paper\my paper\paper2\images\Plot\LRFN.csv'


x = pd.read_csv(root_dir_x, sep='\t',header = None,
                       engine='python')
y = pd.read_csv(root_dir_y, sep='\t',header = None,
                       engine='python')


(root_x, dataset_name_x) = os.path.split(os.path.splitext(root_dir_x)[0])
(root_y, dataset_name_y) = os.path.split(os.path.splitext(root_dir_y)[0])
#os.path.splitext(root_dir_x)[0]: 去除文件后缀名

f, ax = plt.subplots(figsize=(5, 5)) # 画布大小

plt.fill_between(ax.get_xlim(), 0, ax.get_ylim(), facecolor='#F27405', alpha=0.3)

plt.xlabel(dataset_name_x)
plt.ylabel(dataset_name_y)

plt.text(x=0.2,  # 文本x轴坐标
         y=0.7,  # 文本y轴坐标
         s= dataset_name_y+'\n'+'Better Here',  # 文本内容
         rotation=0,  # 文字旋转
         ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=14, color='black',
                       family='Times New Roman',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                       weight='bold',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                       )  # 字体属性设置
         )

plt.text(x=0.5,  # 文本x轴坐标
         y=0.2,  # 文本y轴坐标
         s= dataset_name_x+'\n'+'Better Here',  # 文本内容
         rotation=0,  # 文字旋转
         ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=14, color='black',
                       family='Times New Roman',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                       weight='bold',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                       )  # 字体属性设置
         )

for i in range(x.shape[0]):
    if x[0][i]>y[0][i]:
        C = "red"
        ax.scatter(x[0][i], y[0][i], c=C, s=50, marker='<', clip_on=False)  # s = size , clip_on=False使点不被坐标轴遮挡
        ax.set(xlim=(0, 1), ylim=(0, 1))  # scatter绘制散点图
    elif x[0][i]<y[0][i]:
        C = "blue"
        ax.scatter(x[0][i], y[0][i], c=C, s=50, marker='o', clip_on=False) #s = size
        ax.set(xlim=(0, 1), ylim=(0, 1)) # scatter绘制散点图

# Plot your initial diagonal line based on the starting
# xlims and ylims.
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # 对角线

#plt.fill_between(ax.get_xlim(), 0, ax.get_ylim(), facecolor='#F27405', alpha=0.3)
#填充画布下半部分为绿色 #A65179


def on_change(axes):
    # When this function is called it checks the current
    # values of xlim and ylim and modifies diag_line
    # accordingly.
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    diag_line.set_data(x_lims, y_lims)

# Connect two callbacks to your axis instance.
# These will call the function "on_change" whenever
# xlim or ylim is changed.
ax.callbacks.connect('xlim_changed', on_change)
ax.callbacks.connect('ylim_changed', on_change)

plt.savefig('F:\paper\my paper\paper2\paper\Latex\elsarticle\Paper_image\Plot' + '/'+dataset_name_x, bbox_inches='tight')

plt.show()