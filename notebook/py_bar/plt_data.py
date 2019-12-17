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


def auc_plot(X, y, clf, png_savename=0):
    """
    功能: 画出AUC图
    X: 数据X（无标签/df型）
    y: 数据y（标签/df型）
    clf: 已训练过的最佳lgb模型
    png_savename: 保存图片的名字，默认不保存
    return: AUC图
    """
    y_pre = clf.predict(X)
    y_prb_1 = clf.predict_proba(X)[:,1]  # 输出预测的概率
    fpr, tpr, thres = roc_curve(y, y_prb_1)
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
    

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    混淆矩阵画图函数
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment='center', color='white' if cm[i, j]> thresh else 'black')
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        
        
def metrics_plot(X, y, clf, thres=0.45, png_savename=0):
    """
    功能: 画出混淆矩阵图
    X: 数据X（无标签/df型）
    y: 数据y（标签/df型）
    clf: 已训练过的最佳lgb模型
    thres: 阈值，多少以上为预测正确
    png_savename: 保存图片的名字，默认不保存
    return: 输出混淆矩阵图
    """
    y_pred_proba = clf.predict_proba(X.values)  # 获取概率
    y_prediction = y_pred_proba[:, 1] > thres  # 多少以上的概率判定为正
    cnf_matrix = confusion_matrix(y, y_prediction)  # 形成混淆矩阵
    np.set_printoptions(precision=2)  # 设置浮点精度
    vali_recall = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))  # 召回率
    vali_precision = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))  # 精确率
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='召回率=%s%% \n 精确率=%s%%' %('{0:.1f}'.format(float(vali_recall)*100),
                                                                                         '{0:.1f}'.format(float(vali_precision)*100)))
    if png_savename!=0:
        plt.savefig("%s_混淆矩阵.png" % png_savename)  # 保存混淆矩阵图
        

def importance_plt(X, clf, png_savename=0):
    """
    功能:打印特征重要图
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