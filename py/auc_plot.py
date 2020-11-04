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


def auc_plot(y, y_pro, thre=0.5, png_savename=0):
    """
    功能: 画出AUC图
    why: 能够知道模型的效果，AUC越高，则模型分辨正负样本的能力越好。
    y: 实际正样本
    y_pro：预测概率
    clf: 已训练过的最佳lgb模型
    png_savename: 保存图片的名字，默认不保存
    return: AUC图
    """
    y_pre = y_pro > thre
    fpr, tpr, thres = roc_curve(y, y_pro)
    roc_auc = auc(fpr, tpr)  # 计算AUC
    # 画出AUC
    plt.plot(fpr, tpr, label="AUC = {0:.4f}".format(roc_auc), ms=100)
    plt.title("ROC曲线", fontsize=20)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('误判率', fontsize=15)
    plt.ylabel('命中率', fontsize=15)
    if png_savename != 0:
        plt.savefig("%s_AUC图.png" % png_savename)  # 保存AUC图
    plt.show()
    print("Accuracy: {0:.2f}".format(accuracy_score(y, y_pre)))  # 准确率
 