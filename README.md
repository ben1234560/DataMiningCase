# DataMiningCase
<a href="https://pypi.org/project/lightgbm" rel="nofollow"><img src="https://camo.githubusercontent.com/34244ae628b4cb096fa26305abc1304e5d1b5e33/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f6c6967687467626d2e7376673f6c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python Versions" data-canonical-src="https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&amp;logoColor=white" style="max-width:100%;"></a>
<a href="https://pypi.org/project/lightgbm" rel="nofollow"><img src="https://camo.githubusercontent.com/e78e5fa3a797f79dfb9179ae5d4c34f5409d45b9/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6c6967687467626d2e7376673f6c6f676f3d70797069266c6f676f436f6c6f723d7768697465" alt="PyPI Version" data-canonical-src="https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&amp;logoColor=white" style="max-width:100%;"></a>

你将习得：数据的处理、LightGBM、sklearn包（里面含有：GridSearchCV寻找最优参、StratifiedKFold分层5折切分、train_test_split单次数据切分等）、画AUC图、画混淆矩阵图，并输出预测名单。
### 注释覆盖率为80%左右，旨在帮助快速入门，新手级，持续更新

## 该项目涉及的如下：
<ul>
  <li> 数据处理（preprocessing函数）
    <ul>
      <li> 数据切分成训练用数据X，和校对数据y
      <li> dropna（处理异常值字段）
      <li> pd.value_counts（正负样本情况）
      <li> fillna（null值填充）
    </ul>
  <li> 模型训练
    <ul>
      <li> GridSearchCV（寻找最优参）
      <li> StratifiedKFold（分层5折模型训练）
      <li> train_test_split（单次切分模型训练）
      <li> 输出名单
    </ul>
  <li> 画图
    <ul>
      <li> plot/auc_plot（画AUC图）
      <li> confusion_matrix/plot_confusion_matrix（画混淆矩阵图）
      <li> importance_plt（画特征重要性图）
    </ul>
</ul>


## 说明
<p> 项目代码原型为本人在某行做的流失模型，AUC：85%、召回率：19.4%，精确率：85%（已脱敏）
<p> 本人水平有限，代码搬到外部环境可能有遗漏错误的地方，望不吝赐教，万分感谢。
<p> Email：909336740@qq.com
<p> QQ：909336740
