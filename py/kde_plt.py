import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
import matplotlib.pyplot as plt  # 图形处理包
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns  # 画图工具包


def kde_plt(data, feat, label, png_savename=0):
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