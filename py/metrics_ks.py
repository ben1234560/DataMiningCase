import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆矩阵
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西


def metrics_ks(y, y_predicted):
    """
    功能: 计算模型性能指标：ks， 基于ks找到最佳threshold值
    why: ks值越高，则模型效果越好
    y: 数据y（标签/df型）
    y_predicted: 概率值， 公式为：= clf.predict_proba(X)[:, 1]
    return:
        ks值
        thres_ks值
    """
    fpr, tpr, thres = metrics.roc_curve(y, y_predicted, pos_label=1)
    ks = abs(fpr - tpr).max()  # abs:返回数字绝对值的函数
    tmp = abs(fpr - tpr)
    index_ks = np.where(tmp=ks)  # np.where: 返回符合条件的下标函数
    thres_ks = thres[index_ks]
    return ks, thres_ks