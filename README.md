# MAGIC-Gamma-Telescope-Dataset
# Assignment1-task1
# 数据源：https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope 

# 数据概要：
# 这些数据是一个叫Monte-Carlo的人生成的，用于使用成像技术在地面大气契伦科夫伽马望远镜中模拟高能伽马粒子的配准。数   据集包含19,020条记录。
# 这些记录有10个特征属性和一个类属性。

# 此任务目标：
# 基于MAGIC Gamma望远镜数据集训练决策树，预测采集到的模式是由primary Gamma(信号)引起的还是由hadron(背景)引起的，并比较分类器的性能

# 任务要求：
# 1.实现四个决策树，每个决策树分别使用不同的分类标准(ig,gr,va,gi)
# 2.实现一个组装决策树，这个组织决策树分类标准是由ig,gr,va这四个决策树投票决定。
# 3.实现10折交叉验证来评估组装树和MGI树
# 4.MGI树实现一个后剪枝，并用1/3数据集用于训练，1/3用于剪枝，1/3用于评估，并打出TP,FP.
