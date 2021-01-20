import lightgbm as lgb  # 模型
import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆句子
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split  # 分层五折验证包、寻找最优参函数、切分数据
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')


def train_5_cross(df_pre, X,y, X_test_v1,y_test_v1, thresholds=0.45, id_1='id', csv_name=0):
    """
    功能: 五折训练并输出名单
    why: 5折一般是效果比较稳定的，用于线下做的。
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