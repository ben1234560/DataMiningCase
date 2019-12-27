import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆句子
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西


def preprocessing(df_train, df_pre, feats_list=0, label='label', id_1='id'):
    """
    功能:将训练集和预测集做处理，并输出
    why: 数据处理一般在SQL里面做完，能在SQL里面做完的一定要在SQL里面做。
    df_train: 未处理过的训练集（df型/有label）
    df_pre: 未处理过的预测集（df型/有label）
    feats_list: 特征表，默认使用全部特征，可传入list
    label: 标签，判断0/1
    id: id
    return: 
        dt_train: 处理过的训练集
        df_pre: 处理过的预测集
        X: 训练数据X（df型/无label）
        y: 训练数据y（df型/label）
        X_test_v1: 预测数据X（df型/无label）
        y_test_v1: 预测数据y（df型/label）
    """
    print("=============开始处理数据集===============")
    print("未处理训练集数据大小：", df_train.shape)
    print("未处理预测集数据大小：", df_pre.shape)
    # 和if不同，try的代码即使错了，也不会报错中断代码
    try:
        df_train = df_train.dropna(subset=['A字段']).reset_index(drop=True)  # 删掉指定列为null值的行
        df_train = df_train[df_train['B字段'] >= 0].reset_index(drop=True)  # 去掉指定列非数值的行
        df_pre = df_pre.dropna(subset=['A字段']).reset_index(drop=True)  # 预测集做同样的操作
        df_pre = df_pre[df_pre['B字段'] >= 0].reset_index(drop=True)
    except KeyError:
        print("数据集里没有相应字段")
        
    print("处理后训练集数据大小：", df_train.shape)
    print("处理后预测集数据大小：", df_pre.shape)    
    print("=============数据集处理完成===============")
    
    print("==============开始切分数据=================")
    if feats_list==0:
        print("使用全部特征")
        X = df_train[df_train.columns.drop([id_1, label])]
        X_test_v1 = df_pre[df_pre.columns.drop([id_1, label])]
    elif type(feats_list) == list:
        print("使用列表特征，长度为：", len(feats_list))
        X = df_train[feats_list]
        X_test_v1 = df_pre[feats_list]
    else:
        print("feats_list输入有误")
    y = df_train[label]
    y_test_v1 = df_pre[label]
    X = X.fillna(0)  # null值补0，这是经过模型做出来的最好处理方式
    X_test_v1 = X_test_v1.fillna(0)
    print("训练集正负样本情况：")
    print(pd.value_counts(df_train[label]))
    print("预测集正负样本情况：")
    print(pd.value_counts(df_pre[label]))
    print("==============数据切分完成=================")
    return df_train, df_pre, X, y, X_test_v1, y_test_v1
