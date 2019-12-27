import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆句子
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import matplotlib.pyplot as plt  # 图形处理包
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns  # 画图工具包


def corr_plt(data, feats, start=0, end=20, png_savename=0):
    """
    功能: 画相关系数热力图
    data: 数据集（df型）
    feats: 特征集（list性/一般是去掉id和label），可用该方法生成 feats = [x for x in data.columns if x not in ['id','label']]
    start: 用以画相关系数特征的开始点，默认0（int型）
    end: 用以画相关系数特征的结束点，默认30，-1为最终点（int型）
    png_savename: 保存图片的名字，默认不保存（str型）
    reture:
        输出相关系数图，可返回图片
    """
    corr = data[feats[start:end]].corr()  # 获取相关系数值 
    plt.figure(figsize=(20,10))    # 画布大小
    sns.heatmap(corr, annot=True)   ### 画出热力图， annot是否在方格中置入值
    if png_savename:
        plt.savefig("%s_相关系数图.png" % png_savename)  # 保存相关系数图