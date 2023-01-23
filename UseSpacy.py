# import spacy
# 
# import usenltk
# 
# path = 'TextE.txt'
# text = usenltk.read_text(path)
# print(text)

'''
#下载英文数据库模型
en=spacy.load('en')
text=en(text)
#获得所有句子
for word in list(text.sents)[0]:
print(word.word.tag)
'''

'''

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()  # 获取鸢尾花数据集
Y = iris.target  # 数据集标签 ['setosa', 'versicolor', 'virginica']，山鸢尾、变色鸢尾、维吉尼亚鸢尾
X = iris.data  # 数据集特征 四维，花瓣的长度、宽度，花萼的长度、宽度
# Y
# X
# X.shape
# pd.DataFrame(X)
# 调用PCA
pca = PCA(n_components=2)  # 实例化 n_components:降维后需要的维度，即需要保留的特征数量，可视化一般取值2
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取新矩阵
# X_dr
# 也可以fit_transform一步到位
# X_dr = PCA(2).fit_transform(X)
# 要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标（两个特征向量）应该是三种鸢尾花降维后的
# X_dr[Y == 0, 0] #这里是布尔索引，即取出Y=0的行的第0列

# 对三种鸢尾花分别绘图
colors = ['red', 'black', 'orange']
# iris.target_names
plt.figure()  # 画布
for i in [0, 1, 2]:
    plt.scatter(X_dr[Y == i, 0]  # x轴
                , X_dr[Y == i, 1]  # y轴
                , alpha=1  # 图表的填充不透明度(0到1之间)
                , c=colors[i]  # 颜色
                , label=iris.target_names[i]  # 标签
                )
plt.legend()  # 显示图例
plt.title('PCA of IRIS dataset')  # 设置标题
plt.show()  # 画图

'''

# a=123
# c=a+10
#
# print(a)
# print(c)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()  # 获取鸢尾花数据集
Y = iris.target  # 数据集标签 ['setosa', 'versicolor', 'virginica']，山鸢尾、变色鸢尾、维吉尼亚鸢尾
X = iris.data  # 数据集特征 四维，花瓣的长度、宽度，花萼的长度、宽度
# Y
# X
# X.shape
# pd.DataFrame(X)
# 调用PCA
pca = PCA(n_components=2)  # 实例化 n_components:降维后需要的维度，即需要保留的特征数量，可视化一般取值2
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取新矩阵
# X_dr
# 也可以fit_transform一步到位
# X_dr = PCA(2).fit_transform(X)
# 要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标（两个特征向量）应该是三种鸢尾花降维后的
# X_dr[Y == 0, 0] #这里是布尔索引，即取出Y=0的行的第0列

# 对三种鸢尾花分别绘图
colors = ['red', 'black', 'orange']
# iris.target_names
plt.figure()  # 画布
for i in [0, 1, 2]:
    plt.scatter(X_dr[Y == i, 0]  # x轴
                , X_dr[Y == i, 1]  # y轴
                , alpha=1  # 图表的填充不透明度(0到1之间)
                , c=colors[i]  # 颜色
                , label=iris.target_names[i]  # 标签
                )
plt.legend()  # 显示图例
plt.title('PCA of IRIS dataset')  # 设置标题
plt.show()  # 画图

import math
# def add(x,y):
#    return x+y
# def subtract(x,y):
#      return x-y
# def multiply(x,y):
#       return x*y
# def divide(x,y):
#    if y==0:
#     print("Cannot divide by 0!")
#    else:
#      return x/y
# print(add(4,5))
# print(multiply(1,2))
# print(divide(4,5))
# print(subtract(4,5))
