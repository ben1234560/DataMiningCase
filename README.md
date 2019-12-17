# DataMiningCase
学习该项目，你将习得：数据的处理、LightGBM、sklearn包（里面含有：GridSearchCV寻找最优参、StratifiedKFold分层5折切分、train_test_split单次数据切分等）、画AUC图、画混淆矩阵图，并输出预测名单。
### 注释覆盖率为80%左右，旨在帮助快速入门，代码简单

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
