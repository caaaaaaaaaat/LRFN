import matplotlib.pyplot as plt

# 指定字体路径
font_path = '/path/to/arial.ttf'

# 设置字体样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['font.cursive'] = ['Arial', 'cursive']

# 进行后续的绘图操作