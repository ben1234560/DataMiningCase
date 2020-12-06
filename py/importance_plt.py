import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
import matplotlib.pyplot as plt  # 图形处理包
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False


def importance_plt(X, clf, png_savename=0):
    """
    功能:打印特征重要图
    why: 能看出哪个特征更重要，继而对特征做相关衍生，也可以讲特征使用次数为0的特征去掉，防止冗余。
    X: 数据X（无标签/df型）
    clf: 已训练过的最佳lgb模型
    png_savename: 保存图片的名字，默认不保存
    return: 打印出特征重要性图
    """
    feats_list = X.columns.values  # 获取全部特征
    feats_importance = clf.feature_importances_  # 获取特征使用次数
    sorted_idx = np.argsort(feats_importance)
    
    plt.figure(figsize=(10, 55))
    # 下面是画图操作
    plt.barh(range(len(sorted_idx)), feats_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), feats_list[sorted_idx], align='center')
    plt.xlabel("Importance")
    plt.title("Feature importances")
    if png_savename:
        plt.savefig("特征重要性.png", dpi=500, bbox_inches='tight')  # 由于特征过多图片过大，所以需要这些处理才能让图片全部保存下来
    plt.show()
    
    
def fit_importance_plt(feats_list, feats_importance, png_savename=0):
    """
    功能:打印特征重要图
    why: 能看出哪个特征更重要，继而对特征做相关衍生，也可以讲特征使用次数为0的特征去掉，防止冗余。
    feats_list: 特征名，list类型，可以用如下方法获取X.columns.values  # 获取全部特征
    feats_importance: 已训练过的模型的特征重要性
        xgb和lgb可以用如下方法获得:
        feats_importance = clf.feature_importances_  # 获取特征使用次数
        长这样：array([0.00567917, 0.00615975,],dtype=float32)
    png_savename: 保存图片的名字，默认不保存
    return: 打印出特征重要性图
    """
    sorted_idx = np.argsort(feats_importance)
    
    plt.figure(figsize=(10, 55))
    # 下面是画图操作
    plt.barh(range(len(sorted_idx)), feats_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feats_list)[sorted_idx])
    plt.xlabel("Importance")
    plt.title("Feature importances")
    if png_savename:
        plt.savefig("特征重要性.png", dpi=500, bbox_inches='tight')  # 由于特征过多图片过大，所以需要这些处理才能让图片全部保存下来
    plt.show()