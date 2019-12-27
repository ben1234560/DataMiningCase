import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆句子
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
from sklearn.feature_selection import RFE, RFECV  # 递归消除选特征，前者是自己选优化到多少位，后者是自动cv优化到最佳
from imblearn.under_sampling import RandomUnderSampler  # 朴素随机过采样，由于是比较旧的这里不做例子
from imblearn.over_sampling import SMOTE, ADASYN  # 目前流行的过采样
# SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本;
# ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本;


def over_smote_(X, y, num):
    """
    功能: 二分类过采样，以smote举例。
    X: 数据X（df型/无label）
    y: 数据y（df型/label）
    num: 过采样的个数
    """
    ss = pd.Series(y).value_counts()
    smote = SMOTE(sampling_strategy={0:ss[0],1:ss[1]+num},random_state=2019)  # radom_state为随机值种子，1:ss[1]+表示label为1的数据增加多少个
    # adasyn = ADASYN(sampling_strategy={0:ss[0],1:ss[1]+800},random_state=2019) # 改变正样本数量参数
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("过采样个数为：", num)
    check_num_X = X_resampled.shape[0] - X.shape[0]
    check_num_y = y_resampled.shape[0] - y.shape[0]
    if (check_num_X == check_num_y) and (check_num_X == num):
        print("过采样校验：成功")
        return X_resampled, y_resampled
    else:
        print("过采样校验：失败")