import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西



def just_num_leaves(X, y, start_num=10, end_num=101, step=10):
    """
    功能: 找到最优num_leaves参数，以此类推找出全部的最优参
    why: 最优参数组能让模型效果更好，一般提升在0~5%左右，如果提升超过5%，那么就要考虑特征是否选取正确，是否有过多的噪音数据。
    X: 数据X（无标签/df型）
    y: 数据y（标签/df型）
    start_num: 开始值
    end_num: 最大值
    step: 步数
    return: 最佳num_leaves
    """
    param_dic = {'num_leaves': range(start_num, end_num, step)}
    gscv = GridSearchCV(estimator=lgb.LGBMClassifier(max_depth=20, min_data_in_bin=5, max_bin=200,
                                                     min_child_samples=90, n_estimators=20000,
                                                     objective='binary', boosting_type='gbdt', learning_rate=0.02,
                                                     lambda_l2=5),
                       param_grid=param_dic, scoring='f1', cv=5)
    gscv.fit(X, y)
    print("best_params:{0}".format(gscv.best_params_))
    print("best_score:{0}".format(gscv.best_score_))