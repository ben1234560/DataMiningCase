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


def train_5_cross(df_pre, X,y, X_test_v1,y_test_v1, thresholds=0.45, id_1='id', csv_name=0):
    """
    功能: 五折训练并输出名单
    df_pre：原始预测数据
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    X_test_v1: 预测数据X（无标签/df型）
    y_test_v1: 预测数据y（无标签/df型）
    thresholds: 阈值选择，默认0.45高精确率
    csv_name: 保存csv的名称，默认不保存
    returen:
        模型，客户名单及情况
    """
    vali_auc_num=0  # 验证集AUC
    vali_recall_num=0  # 验证集召回率
    vali_precision_num=0  # 验证集精确率
    test_auc_num=0  # 预测集AUC
    test_recall_num=0  # 预测集召回率
    test_precision_num=0  # 预测集精确率
    y_pred_input = np.zeros(len(X_test_v1))  # 相应大小的零矩阵
    print("=============开始训练================")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)  # 分层采样, n_splits为几折
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("第 {} 次训练...".format(fold_+1))
        train_x, trai_y = X.loc[trn_idx], y.loc[trn_idx]
        vali_x, vali_y = X.loc[val_idx], y.loc[val_idx]
        
        # 以下为调过参的lgb模型
        clf = lgb.LGBMClassifier(max_depth=20, min_data_in_bin=5, max_bin=200,
                                min_child_samples=90, num_leaves=20, n_estimators=20000,
                                objective='binary', boosting_type='gbdt', learning_rate=0.02,
                                lambda_l2=5)
        clf.fit(train_x, trai_y, eval_set=[(train_x, trai_y), (vali_x, vali_y)], verbose=0,
               early_stopping_rounds=100, eval_metric='f1')
        
        # 不懂的去GitHub看搜LightGBM的参数解释
        
        # ===============验证集AUC操作===================
        y_prb = clf.predict_proba(vali_x)[:,1]  # 获取预测概率
        # fpr:在实际为正的样本中，被正确判断为正的比例。tpr:在实际为负的样本中，被正确判断为负的比例。thres为阈值
        fpr, tpr, thres = roc_curve(vali_y, y_prb)
        vali_roc_auc = auc(fpr, tpr)  # 获取验证集auc
        vali_auc_num += vali_roc_auc  # 将本次auc加入总值里
        print("vali auc = {0:.4}".format(vali_roc_auc))  # 本次auc的值
        # ===============预测集AUC操作===================
        y_prb_test = clf.predict_proba(X_test_v1)[:,1]  # 获取预测概率
        fpr, tpr, thres = roc_curve(y_test_v1, y_prb_test)
        test_roc_auc = auc(fpr, tpr)
        test_auc_num += test_roc_auc
        print("test auc = {0:.4}".format(test_roc_auc))
        
        # ===============验证metric操作===================
        y_pre_proba = clf.predict_proba(vali_x.values)
        y_predictions = y_pre_proba[:, 1]>thresholds  # 取阈值多少以上的为True
        cnf_matrix = confusion_matrix(vali_y, y_predictions)  # 建立矩阵
        np.set_printoptions(precision=2)  # 控制在两位数
        vali_recall = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))  # 召回率
        vali_precision = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))  # 精确率
        print("vali_metric: ", vali_recall, vali_precision)
        vali_recall_num += float(vali_recall)  # 将本次召回率加入总值里
        vali_precision_num += float(vali_precision)  # 将本次精确率加入总值里
        # ===============预测metric操作===================
        y_pre_proba_test = clf.predict_proba(X_test_v1.values)
        y_predictions_test = y_pre_proba_test[:, 1]>thresholds  # 取阈值多少以上的为True
        cnf_matrix_test = confusion_matrix(y_test_v1, y_predictions_test)  # 建立矩阵
        np.set_printoptions(precision=2)  # 控制在两位数
        test_recall = '{0:.3f}'.format(cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1]))  # 召回率
        test_precision = '{0:.3f}'.format(cnf_matrix_test[1,1]/(cnf_matrix_test[0,1]+cnf_matrix_test[1,1]))  # 精确率
        print("test_metric: ", test_recall, test_precision)
        test_recall_num += float(test_recall)  # 将本次召回率加入总值里
        test_precision_num += float(test_precision)  # 将本次精确率加入总值里
        y_pred_input += y_pre_proba_test[:, 1]  # 将每次的预测的结果写入数组中
        
    print("5折泛化，验证集AUC：{0:.3f}".format(vali_auc_num/5))  # 前面是做了5次相加，所以这次要除以5
    print("5折泛化，预测集AUC：{0:.3f}".format(test_auc_num/5))
    
    print("5折泛化，验证集recall：{0:.3f}".format(vali_recall_num/5))
    print("5折泛化，验证集precision：{0:.3f}".format(vali_recall_num/5))
    
    print("5折泛化，预测集recall：{0:.3f}".format(test_recall_num/5))
    print("5折泛化，预测集precision：{0:.3f}".format(test_recall_num/5))
    
    print("================开始输出名单==================")
    y_pred_input_end = y_pred_input / 5  # 前面是做了5次相加，所以这次要除以5
    y_pred_input_precision = y_pred_input_end > thresholds  # 获取高精确率的标签
    submission = pd.DataFrame({"id": df_pre[id_1],
                              "概率": y_pred_input_end,
                              "高精确": y_pred_input_precision})
    if csv_name != 0:
        submission.to_csv("%s预测名单.csv" % csv_name, index=False)  # 保存
    print("================输出名单名单==================")
    print(submission.head(5))
    return clf
    
    
def metrics_ks(y, y_predicted):
    """
    功能: 计算模型性能指标：ks， 基于ks找到最佳threshold值
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
    
    
def train_2_cross(df_pre,X,y, X_test_v1,y_test_v1, thresholds=0.45, id_1='id', csv_name=0):
    """
    功能: 五折训练并输出名单
    df_pre：原始预测数据
    X: 训练数据X（无标签/df型）
    y: 训练数据y（标签/df型）
    X_test_v1: 预测数据X（无标签/df型）
    y_test_v1: 预测数据y（无标签/df型）
    thresholds: 阈值选择，默认0.45高精确率
    csv_name: 保存csv的名称，默认不保存
    returen:
        模型，客户名单及情况
    """
    y_pred_input = np.zeros(len(X_test_v1))  # 相应大小的零矩阵
    train_x, vali_x, train_y,vali_y = train_test_split(X, y, test_size=0.33, random_state=1234)
    clf = lgb.LGBMClassifier(max_depth=20, min_data_in_bin=5, max_bin=200,
                            min_child_samples=90, num_leaves=20, n_estimators=20000,
                            objective='binary', boosting_type='gbdt', learning_rate=0.02,
                            lambda_l2=5)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (vali_x, vali_y)], verbose=0,
           early_stopping_rounds=100, eval_metric='f1')
    # 这里的参数不懂的去GitHub搜LightGBM的参数解释

    # ===============验证集AUC操作===================
    y_prb = clf.predict_proba(vali_x)[:,1]  # 获取预测概率
    # fpr:在实际为正的样本中，被正确判断为正的比例。tpr:在实际为负的样本中，被正确判断为负的比例。thres为阈值
    fpr, tpr, thres = roc_curve(vali_y, y_prb)
    vali_roc_auc = auc(fpr, tpr)  # 获取验证集auc
    print("vali auc = {0:.4}".format(vali_roc_auc))  # 本次auc的值
    # ===============预测集AUC操作===================
    y_prb_test = clf.predict_proba(X_test_v1)[:,1]  # 获取预测概率
    fpr, tpr, thres = roc_curve(y_test_v1, y_prb_test)
    test_roc_auc = auc(fpr, tpr)
    print("test auc = {0:.4}".format(test_roc_auc))
    
    # ===============训练metric操作===================
    y_pre_proba = clf.predict_proba(vali_x.values)
    y_predictions = y_pre_proba[:, 1]>thresholds  # 取阈值多少以上的为True
    cnf_matrix = confusion_matrix(vali_y, y_predictions)  # 建立矩阵
    np.set_printoptions(precision=2)  # 控制在两位数
    vali_recall = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))  # 召回率
    vali_precision = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))  # 精确率
    print("vali_metric: ", vali_recall, vali_precision)
    # ===============预测metric操作===================
    y_pre_proba_test = clf.predict_proba(X_test_v1.values)
    y_predictions_test = y_pre_proba_test[:, 1]>thresholds  # 取阈值多少以上的为True
    cnf_matrix_test = confusion_matrix(y_test_v1, y_predictions_test)  # 建立矩阵
    np.set_printoptions(precision=2)  # 控制在两位数
    test_recall = '{0:.3f}'.format(cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1]))  # 召回率
    test_precision = '{0:.3f}'.format(cnf_matrix_test[1,1]/(cnf_matrix_test[0,1]+cnf_matrix_test[1,1]))  # 精确率
    print("test_metric: ", test_recall, test_precision)

    print("================开始输出名单==================") 
    y_pred_input_precision = y_pre_proba_test[:, 1] > thresholds  # 获取高精确率的标签
    submission = pd.DataFrame({"id": df_pre[id_1],
                              "概率": y_pre_proba_test[:, 1],
                              "高精确": y_pred_input_precision})
    if csv_name != 0:
        submission.to_csv("%s预测名单.csv" % csv_name, index=False)  # 保存
    print("================输出名单名单==================")
    print(submission.head(5))
    return clf


# 测试用lgb_params 参数
lgb_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.06,
    'num_leaves': 31,
    'max_depth': -1,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.8
}


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


def over_smote_(X, y, num):
    """
    功能: 二分类过采样，以smote举例。
    why: 当正负样本比例相差过大时，一般为1：20以内。举例：如果正负样本为1：99，那么相当于告诉模型只要判断为负，则正确率就为99%，那么模型就会这么做。
    X: 数据X（df型/无label）
    y: 数据y（df型/label）
    num: 过采样的个数
    reture: 
        X_resampled: 过采样后的X
        y_resampled: 过采样后的y
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