{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd  # 数据处理包\n",
    "import numpy as np  # 数据处理包\n",
    "import matplotlib.pyplot as plt  # 图形处理包\n",
    "import gc  # 处理缓存，有兴趣的可以搜搜怎么使用\n",
    "import warnings  # 忽略普通警告，不打印太多东西\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.rcParams['font.sans-serif']=['SimHei']  # 让图形可以显示中文\n",
    "plt.rcParams['axes.unicode_minus']=False\n",
    "import seaborn as sns  # 画图工具包\n",
    "\n",
    "\n",
    "def scatter_plt(data, feat_1,feat_2, label, png_savename=0):\n",
    "    \"\"\"\n",
    "    功能:画双特征二分类散点图\n",
    "    why: 通过该图能够明显的看出正负样本在不同区间的差异，更能找到特征。\n",
    "    data: 数据集（df型）\n",
    "    feat_1: 横轴单个特征名（str型）\n",
    "    feat_2: 纵轴单个特征名（str型）\n",
    "    label: 标签名（str型）\n",
    "    png_savename: 保存图片，以feat为名字，默认不保存\n",
    "    return:\n",
    "        返回二分类图，可保存图片\n",
    "    \"\"\"\n",
    "    data_label_1 = data[data[label]==1]\n",
    "    plt.scatter(data_label_1[feat_1], data_label_1[feat_2], color='red')\n",
    "    data_label_0 = data[data[label]==0]\n",
    "    plt.scatter(data_label_0[feat_1], data_label_0[feat_2], color='green')\n",
    "    plt.title('%s and %s' % (feat_1,feat_2)) \n",
    "    plt.xlabel(feat_1)\n",
    "    plt.ylabel(feat_2)\n",
    "    if png_savename:\n",
    "        plt.savefig(\"%s_%s_二分类柱状图.png\" % (feat_1, feat_2), dpi=300)  # 保存二分类图，以feat为名字, dpi清晰度"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
