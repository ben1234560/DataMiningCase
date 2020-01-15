import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn.metrics import roc_auc_score  # roc
from sklearn.model_selection import StratifiedKFold  # 数据切分、分层五折验证包
import lightgbm as lgb  # lgb模型 ,安装的方法是在anaconda promote里，直接pip install lightgbm 即可，做第一层1模型
import xgboost as xgb  # xgb模型，安装的方法是在anaconda promote里，直接pip install xgboost 即可，和lightgbm一样，做第一层2模型
from sklearn.linear_model import LogisticRegression  # 载入lr模型，这里lr模型用作第二层模型

"""
为什么需要模型融合：多个学习器进行结合，常可获得比单一学习器显著优越的泛化性能。 --《机器学习 周志华》

stacking模型融合策略：我们的第一层模型是lgb和xgb，第二层模型是lr
第一层模型的作用：将数据训练并预测，并将训练结果和预测结果作为特征
第二层模型的作用：将第一层的两个模型的训练结果和预测结果作为训练集和预测集，对数据进行训练和预测
"""


def get_leaderboard_score(test_df,prediction):
    """
    定义评分函数
    test_df: 测试集
    prediction: 预测结果
    reture: 输出结果分数
    """
    label = test_df['label'].values  # 拿出真实样本
    assert len(prediction) == len(label)  # 断言其长度相等
    print('stacking auc score: ', roc_auc_score(label, prediction))  # 计算评分


train = pd.read_csv('../data/train.csv')  # 导入训练集数据
test = pd.read_csv('../data/train.csv')  # 导入预测集数据
print("数据集大小：", train.shape, test.shape)
train = train.fillna(-999) # 填充特殊值
test = test.fillna(-999)

feats = [x for x in train.columns if x not in ['id','label']]  # 拿出特征
X = train[feats].values  # 训练X
y = train['label'].values  # 训练y
X_test = test[feats].values  # 测试X
y_test  = test['label'].values  # 测试y

# 设置skf
data_seed = 2020
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_seed)
# lgb和xgb的参数
lgb_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.06,
    'num_leaves': 31,
    'max_depth': -1,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 300, ### 特征抽样的随机种子
    'bagging_seed': 3, ### 数据抽样的随机种子,取10个不同的，然后对结果求平均,todo:求出10个结果，然后求平均
    #'is_unbalance': True   #### 第一种方法：设置is_unbalance为True，表明传入的数据集是类别不平衡的
    #'scale_pos_weight': 98145/1855###负样本数量/正样本数量 -> scale_pos_weight * 正样本 == 负样本
}
xgb_params = {
    'booster': 'gbtree',  ##提升类型
    'objective': 'binary:logistic',  ###目标函数
    'eval_metric': 'auc',  ##评价函数
    'eta': 0.1,  ### 学习率 ，一般0.0几
    'max_depth': 6,  ###树最大深度
    'min_child_weight': 1,  ###最小样本二阶梯度权重, 取值是整数
    'subsample': 1.0,  ###训练数据采样 ,取值0.0~1.0之间
    'colsample_bytree': 1.0,  ###训练特征采样，取值0.0~1.0之间
    'lambda': 1,  ## l2正则，取值是整数
    'alpha': 0,   ### l1正则，取值整数
    'silent': 1   ### 取值1控制xgboost训练信息不输出
}

blend_train = pd.DataFrame()  # 定义df数据接收验证集结果，作为特征
blend_test = pd.DataFrame()  # 定义df数据接收测试集结果，作为特征
# 训练lgb，用作第一层模型中的其中一个
test_pred_lgb = 0  # 预测结果存放对象
cv_score_lgb = []  # 存放每次auc的对象
train_feats = np.zeros(X.shape[0])  # 整体训练的样本数量
for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print('training fold: ', idx + 1)  # 遍历的第几次
    train_x, valid_x = X[train_idx], X[test_idx]  # 拆分成训练集和验证集
    train_y, valid_y = y[train_idx], y[test_idx]  # 拆分成训练集和验证集
    dtrain = lgb.Dataset(train_x, train_y, feature_name=feats)  # 组成训练集
    dvalid = lgb.Dataset(valid_x, valid_y, feature_name=feats)  # 组成验证集
    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=dvalid, early_stopping_rounds=50, verbose_eval=50)  # 定义lgb模型

    valid_pred = model.predict(valid_x, num_iteration=model.best_iteration)  # 当前模型最佳参数并预测，num_iteration：选择最优的lgb
    train_feats[test_idx] = valid_pred  # 每次把验证集的结果填入，做训练的结果集，由于是5折，所以每次都是1/5的数据，把它们当作lgb训练集特征
    auc_score = roc_auc_score(valid_y, valid_pred)  # 计算auc
    print('auc score: ', auc_score)
    cv_score_lgb.append(auc_score)  # 存放验证集auc值
    test_pred_lgb += model.predict(X_test, num_iteration=model.best_iteration)  # 预测结果并累加，做预测的结果集，把它们当作lgb测试集当作特征
    
print("训练的结果:",train_feats)
test_pred_lgb /= 5
print("测试的结果", test_pred_lgb)  # 由于测试的结果是5折每次的累加，所以需要除于5
# 将训练结果和预测结果加入到blend数据集
blend_train['lgb_feat'] = train_feats
blend_test['lgb_feat'] = test_pred_lgb

# 训练xgb，用作第一层模型中的其中一个
test_pred_xgb = 0 # 预测结果存放对象
cv_score_xgb = []  # 存放每次auc的对象
train_feats_xgb = np.zeros(X.shape[0])  # 整体训练的样本数量
for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print('training fold: ', idx + 1) # 遍历的第几次
    train_x, valid_x = X[train_idx], X[test_idx]  # 拆分成训练集和验证集
    train_y, valid_y = y[train_idx], y[test_idx]  # 拆分成训练集和验证集
    dtrain = xgb.DMatrix(train_x, train_y, feature_names=feats)  # 组成训练集
    dvalid = xgb.DMatrix(valid_x, valid_y, feature_names=feats)  # 组成验证集
    watchlist = [(dvalid, 'eval')]
    model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=watchlist, early_stopping_rounds=50, verbose_eval=50)  # 定义xgb模型

    valid_pred = model.predict(dvalid, ntree_limit=model.best_iteration)  # 当前模型最佳参数并预测，ntree_limit：选择最优的xgb
    train_feats_xgb[test_idx] = valid_pred  # 每次把验证集的结果填入，做训练的结果集，由于是5折，所以每次都是1/5的数据
    auc_score = roc_auc_score(valid_y, valid_pred)  # 计算auc
    print('auc score: ', auc_score)
    cv_score_xgb.append(auc_score)  # 存放验证集auc值
    dtest = xgb.DMatrix(X_test,feature_names=feats)  ##同时指定特征名字
    test_pred_xgb += model.predict(dtest, ntree_limit=model.best_iteration)  # 预测结果并累加，做预测的结果集
    
print("训练的结果:", train_feats_xgb)
train_feats_xgb /= 5  # 由于测试的结果是5折每次的累加，所以需要除于5
pirnt("测试的结果:", train_feats_xgb)
# 将训练结果和预测结果加入到blend数据集
blend_train['xgb_feat'] = train_feats_xgb
blend_test['xgb_feat'] = test_pred_xgb

print(blend_train.head(5))  # 查看训练集作为特征的情况
print(blend_test.head(5))  #  查看测试集作为特征的情况
# 第二层模型lr训练
lr_model = LogisticRegression()  # 实例化
lr_model.fit(blend_train.values, y)  # 训练
print("特征权重:", lr_model.coef_)
test_pred_lr = lr_model.predict_proba(blend_test.values)[:,1]  # 第二层模型预测结果

# 各个模型的评分情况
get_leaderboard_score(test,test_pred_lr)  # stacking模型的分数