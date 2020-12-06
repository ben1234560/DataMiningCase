import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
from sklearn import metrics  # 混淆矩阵
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix  # 准确率、roc计算、auc计算、混淆矩阵
import matplotlib.pyplot as plt  # 图形处理包
import itertools  # 处理混淆矩阵
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns  # 画图工具包


def auc_plot(y, y_prob, thres=0.5, png_savename=0):
    """
    why: 能够知道模型的效果，AUC越高，则模型分辨正负样本的能力越好。
    功能: 画出AUC图
    y: 真实结果（标签/df型）
    y_prob：预测概率（浮点数）
    png_savename: 保存图片的名字，默认不保存
    return: AUC图
    """
    y_prediction = y_prob > thres  # 输出预测的结果
    fpr, tpr, thres = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)  # 计算AUC
    # 画出AUC
    plt.plot(fpr, tpr, label="AUC = {0:.4f}".format(roc_auc), ms=100)
    plt.title("ROC曲线", fontsize=20)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('误判率', fontsize=15)
    plt.ylabel('命中率', fontsize=15)
    if png_savename != 0:
        plt.savefig("%s_AUC图.png" % png_savename, dpi=300)  # 保存AUC图, dpi清晰度，越大越清晰
    plt.show()
    print("Accuracy: {0:.2f}".format(accuracy_score(y, y_prediction)))  # 准确率
    

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
        
        
def metrics_plot(y, y_prob, thres=0.45, png_savename=0):
    """
    why: 能选择是召回率高，还是精确率高，也能从一定层面看出模型的效果。
    功能: 画出混淆矩阵图
    y: 真实值y（标签/df型）
    y_prob：预测概率
    thres: 阈值，多少以上为预测正确
    png_savename: 保存图片的名字，默认不保存
    return: 输出混淆矩阵图
    """
    y_prediction = y_prob > thres  # 多少以上的概率判定为正
    cnf_matrix = confusion_matrix(y, y_prediction)  # 形成混淆矩阵
    np.set_printoptions(precision=2)  # 设置浮点精度
    vali_recall = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))  # 召回率
    vali_precision = '{0:.3f}'.format(cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))  # 精确率
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='召回率=%s%% \n 精确率=%s%%' %('{0:.1f}'.format(float(vali_recall)*100),
                                                                                         '{0:.1f}'.format(float(vali_precision)*100)))
    if png_savename!=0:
        plt.savefig("%s_混淆矩阵.png" % png_savename,dpi=300)  # 保存混淆矩阵图
        

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
    plt.barh(range(len(sorted_idx)), feats_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), feats_list[sorted_idx])
    plt.xlabel("Importance")
    plt.title("Feature importances")
    if png_savename:
        plt.savefig("特征重要性.png", dpi=500, bbox_inches='tight')  # 由于特征过多图片过大，需要处理才能让图片全部保存下来,dpi清晰度
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
    
    
def corr_plt(data, feats, start=0, end=20, png_savename=0):
    """
    功能: 画相关系数热力图
    why: 大于0.75的特征只留一个，不然会造成特征冗余模型效果差，但是现实情况中，一般去掉其中一个就会导致模型效果变差，请慎用。
    data: 数据集（df型）
    feats: 特征集（list性/一般是去掉id和label），可用该方法生成 feats = [x for x in data.columns if x not in ['id','label']]
    start: 用以画相关系数特征的开始点，默认0（int型）
    end: 用以画相关系数特征的结束点，默认30，-1为最终点（int型）
    png_savename: 保存图片的名字，默认不保存（str型）
    reture:
        输出相关系数图，可返回图片
    """
    corr = data[feats[start:end]].corr()  # 获取相关系数值 
    plt.figure(figsize=(20,10))    # 画布大小
    sns.heatmap(corr, annot=True)   ### 画出热力图， annot是否在方格中置入值
    if png_savename:
        plt.savefig("%s_相关系数图.png" % png_savename, dpi=300)  # 保存相关系数图
        
        
def kde_plt(data, feat, label,png_savename=0):
    """
    功能: 画二分类密度线图
    why: 通过该图能够明显的看出正负样本在不同区间的差异，更能找到特征。
    data: 数据集（df型）
    feat: 单个特征名（str型）
    label: 标签名（str型）
    png_savename: 保存图片，以feat为名字，默认不保存
    return:
        返回二分类图，可保存图片
    """
    # 对于区分不易的特征，如资产，有很多连续值且大小不一，可以先放大再log以便区分
    # data['asset_log'] = data['asset'].apply(lambda x:int(math.log(1+x*x))) 这里的math需要导入
    sns.kdeplot(data[data[label]==0][feat], label='label_0', shade=True)  # feat是取的特征，0/1是正负样本，label是命名，shade为阴影
    sns.kdeplot(data[data[label]==1][feat], label='label_1', shade=True)
    plt.title(feat)
    if png_savename:
        plt.savefig("%s_二分类密度线图.png" % feat, dpi=300)  # 保存二分类图，以feat为名字
        
        
def bar_plt(label, feat, data, png_savename=0):
    """
    功能:画二分类柱状图
    why: 通过该图能够明显的看出正负样本在不同区间的差异，更能找到特征。
    data: 数据集（df型）
    feat: 单个特征名（str型）
    label: 标签名（str型）
    png_savename: 保存图片，以feat为名字，默认不保存
    return:
        返回二分类图，可保存图片
    """
    label0 = data[feat][data[label]==0].value_counts()
    label1 = data[feat][data[label]==1].value_counts()
    df_test = pd.DataFrame({'0':label0/(sum(label0)), '1':label1/(sum(label1))})  # 换成百分比，因为正负样本差异大，不除以sum就是数值
    df_test.plot(kind='bar', stacked=False,color=['red','blue'])
    plt.title(feat)
    plt.ylabel('precent')
    if png_savename:
        plt.savefig("%s_二分类柱状图.png" % feat, dpi=300)  # 保存二分类图，以feat为名字


def scatter_plt(data, feat_1,feat_2, label, png_savename=0):
    """
    功能:画双特征二分类散点图
    why: 通过该图能够明显的看出正负样本在不同区间的差异，更能找到特征。
    data: 数据集（df型）
    feat_1: 横轴单个特征名（str型）
    feat_2: 纵轴单个特征名（str型）
    label: 标签名（str型）
    png_savename: 保存图片，以feat为名字，默认不保存
    return:
        返回二分类图，可保存图片
    """
    data_label_1 = data[data[label]==1]
    plt.scatter(data_label_1[feat_1], data_label_1[feat_2], color='red')
    data_label_0 = data[data[label]==0]
    plt.scatter(data_label_0[feat_1], data_label_0[feat_2], color='green')
    plt.title('%s and %s' % (feat_1,feat_2)) 
    plt.xlabel(feat_1)
    plt.ylabel(feat_2)
    if png_savename:
        plt.savefig("%s_%s_二分类柱状图.png" % (feat_1, feat_2), dpi=300)  # 保存二分类图，以feat为名字, dpi清晰度