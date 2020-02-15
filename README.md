# DataMiningCase
<a href="https://pypi.org/project/lightgbm" rel="nofollow"><img src="https://camo.githubusercontent.com/34244ae628b4cb096fa26305abc1304e5d1b5e33/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f6c6967687467626d2e7376673f6c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python Versions" data-canonical-src="https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&amp;logoColor=white" style="max-width:100%;"></a>
<a href="https://pypi.org/project/lightgbm" rel="nofollow"><img src="https://camo.githubusercontent.com/e78e5fa3a797f79dfb9179ae5d4c34f5409d45b9/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6c6967687467626d2e7376673f6c6f676f3d70797069266c6f676f436f6c6f723d7768697465" alt="PyPI Version" data-canonical-src="https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&amp;logoColor=white" style="max-width:100%;"></a>
[![image](https://img.shields.io/badge/conda-jupyter-deepgreen.svg)](https://www.anaconda.com/)

<p> 流失预警模型（二分类），代码原型为本人在某银行做的流失模型，AUC：83%、召回率（覆盖率）：19.4%，精确率：85%（数据是外部数据/代码已脱敏）
<p>你将习得：数据的处理、LightGBM、sklearn包（里面含有：GridSearchCV寻找最优参、StratifiedKFold分层5折切分、train_test_split单次数据切分等）、stacking模型融合、画AUC图、画混淆矩阵图，并输出预测名单。
<p>并告诉你：是什么（WHAT）、怎么做(HOW)、为什么这么做(WHY)。




### 注释覆盖率为80%左右，旨在帮助快速入门，新手级，持续更新，提供免费支持，只需要一颗star

## 该项目涉及的如下：

<ul>
  <li>商业理解
    <ul>
      <li><a href='https://github.com/ben1234560/DataMiningCase/blob/master/doc/%E4%B8%9A%E5%8A%A1%E9%9C%80%E6%B1%82%E5%88%86%E6%9E%90.md'>业务需求分析（实战）</a>
    </ul>
  </li>
  <li>数据理解
    <ul>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/doc/%E6%95%B0%E6%8D%AE%E7%90%86%E8%A7%A3.md'>数据质量探查</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/doc/%E6%95%B0%E6%8D%AE%E7%90%86%E8%A7%A3.md'>重要特征探查</a>
    </ul>
  </li>
  <li> 数据处理（数据准备）
    <ul>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/preprocessing.py'>Preprocessing函数（切分训练数据和预测数据）</a>
      </li>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/preprocessing.py'>Dropna（处理异常值字段）</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/preprocessing.py'>Pd.value_counts（正负样本情况）</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/preprocessing.py'>Fillna（null值填充）</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/over_smote_.py'>Smote（过采样）</a>
    </ul>
  <li> 特征工程（数据准备）
      <ul>
          <li>
              <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/corr_plt.py'>Corr（特征相关系数图）</a>
          </li>
          <li>
              <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/bar_plt.py'>正负样本特征柱状图</a>
          </li>
          <li>
              <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/kde_plt.py'>正负样本特征线性图</a>
          </li>
          <li>
              <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/rfecv_.py'>RFECV（特征五折递归消除）</a>
          </li>
          <li>
              <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/importance_plt.py'>Importance（基于模型的特征重要性）</a>
          </li>
      </ul>
  <li> 建立模型
    <ul>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/just_num_leaves.py'>GridSearchCV（寻找最优参）</a>
      <li> <a href='https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst'>LightGBM参数详解</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/train_5_cross.py'>StratifiedKFold（分层5折模型训练）</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/train_2_cross.py'>Train_test_split（单次切分模型训练）</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/train_5_cross.py'>输出名单</a>
      <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/metrics_ks.py'>Ks值及优Threshold</a>
    </ul>
  <li> 模型融合
      <ul>
          <li><a href='https://github.com/ben1234560/DataMiningCase/blob/master/notebook/%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88.ipynb'>Stacking模型融合_note版(含简单加权融合)</a>
          <li><a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/stacking_fusion.py'>Stacking模型融合_py版</a></li>
          </li>
      </ul>
  <li> 模型评估及实验
    <ul>
      <li> 画图
        <ul>
          <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/auc_plot.py'>Plot/auc_plot（画AUC图）</a>
          <li> <a href='https://github.com/ben1234560/DataMiningCase/blob/master/py/metrics_plot.py'>Confusion_matrix/plot_confusion_matrix（画混淆矩阵图）</a>
        </ul>
    </ul>
    <ul>
      <li><a href='https://github.com/ben1234560/DataMiningCase/tree/master/doc'>实验模板</a>
    </ul> 
</ul>
<img src="assets/数据挖掘流程图.png" alt="数据挖掘流程图" title="数据挖掘流程图" width="330"  height = "300" />

## 说明
<p> 本专题并不用于商业用途，转载请注明本专题地址，如有侵权，请务必邮件通知作者。
<p> 本人水平有限，代码搬到外部环境难免有遗漏错误的地方，望不吝赐教，万分感谢。
<p> 有代码疑惑的地方也请找我。
<p> Email：909336740@qq.com
<p> QQ：909336740
<p> PS：如你尝试有效并喜欢，欢迎点赞，如你尝试失败请联系我。

