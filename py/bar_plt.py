import pandas as pd  # 数据处理包
import numpy as np  # 数据处理包
import matplotlib.pyplot as plt  # 图形处理包
import gc  # 处理缓存，有兴趣的可以搜搜怎么使用
import warnings  # 忽略普通警告，不打印太多东西
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文
plt.rcParams['axes.unicode_minus']=False


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
