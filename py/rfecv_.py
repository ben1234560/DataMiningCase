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


def rfecv_(X, y, feats, lgb_model, cv=5, scoring='roc_auc',verbose=1):
    """
    功能: 减少特征，递归消除选特征，输出结果最优最少的特征组。基于lgb模型
    why: 防止特征冗余，该方法有一定的正反性，即最佳的特征组可能是当前数据的最近，以后数据变化了可能就不是了，建议多测几次。
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    feats: 特征集（list性/一般是去掉id和label），可用该方法生成 feats = [x for x in data.columns if x not in ['id','label']]
    lgb_model: 模型参数
    reture:
        rfe_cv_model: 特征相关信息对象
        selected_feat: 当前数据消除后特征组
    """
    lgb_model = lgb.LGBMClassifier(**lgb_params)  # 传入参数字典
    rfe_cv_model = RFECV(lgb_model, cv=5, scoring='roc_auc', verbose=1) # 自动选择特定的特征数量,cv为多少折，scoring为评分标准，verbose为信息显示
    rfe_cv_model.fit(X, y)  # 开始训练
    selected_feat = np.array(feats)[rfe_cv_model.support_].tolist()  # 拿出特征
    print("剩余特征：", len(selected_feat))
    return rfe_cv_model, selected_feat