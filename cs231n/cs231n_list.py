# coding=utf-8
# -*- coding:cp936 -*-
""""""

''' Assignment #1 '''
# 1. understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
#    理解基本的图像分类流程和数据驱动方法(训练/预测 阶段)

# 图像分类流程
# 图像分类就是输入一个元素为像素值的数组, 然后给它分配一个分类标签. 完整流程如下:
# 输入: 输入是包含N个图像的集合, 每个图像的标签是K种分类标签中的一种. 这个集合称为训练集.
# 学习: 使用训练集来学习每个类到底长什么样. 一般该步骤叫做训练分类器或者学习一个模型.
# 评价: 让分类器来预测它未曾见过的图像的分类标签, 并以此来评价分类器的质量.
#      我们会把分类器预测的标签和图像真正的分类标签对比.
#      毫无疑问, 分类器预测的分类标签和图像真正的分类标签如果一致, 那就是好事, 这样的情况越多越好.


# 2. understand the train/val/test splits and the use of validation data for hyperparameter tuning.
#    理解 训练/验证/测试 分块, 学会使用验证数据来进行超参数调优.


# 3. develop proficiency in writing efficient vectorized code with numpy
#    熟悉使用numpy来编写向量化代码


# 4. implement and apply a k-Nearest Neighbor (kNN) classifier
#    实现并应用k-NN分类器


# 5. implement and apply a Multiclass Support Vector Machine (SVM) classifier
#    实现并应用多类支持向量机(SVM)分类器


# 6. implement and apply a Softmax classifier
#    实现并应用Softmax分类器


# 7. implement and apply a Two layer neural network classifier
#    实现并应用一个两层神经网络分类器


# 8. understand the differences and tradeoffs between these classifiers
#    理解以上分类器的差异和权衡之处


# 9. get a basic understanding of performance improvements from using
#    higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)
#    基本理解使用更高层次表达相较于使用原始图像像素对算法性能的提升(例如: 色彩直方图和梯度直方图HOG)


''' Assignment #2 '''
# 1. understand Neural Networks and how they are arranged in layered architectures
#    理解神经网络及其分层结构


# 2. understand and be able to implement (vectorized) backpropagation
#    理解并实现(向量化)反向传播


# 3. implement various update rules used to optimize Neural Networks
#    实现多个用于神经网络最优化的更新方法


# 4. implement batch normalization for training deep networks
#    实现用于训练深度网络的批量归一化(batch normalization )


# 5. implement dropout to regularize networks
#    实现随机失活(dropout)


# 6. effectively cross-validate and find the best hyperparameters for Neural Network architecture
#    进行高效的交叉验证并为神经网络结构找到最好的超参数


# 7. understand the architecture of Convolutional Neural Networks
#    and train gain experience with training these models on data
#    理解卷积神经网络的结构，并积累在数据集上训练此类模型的经验


''' Assignment #3 '''
# 1. Understand the architecture of recurrent neural networks (RNNs)
#    and how they operate on sequences by sharing weights over time
#    理解循环神经网络(RNN)的结构, 知道它们是如何随时间共享权重来对序列进行操作的


# 2. Understand the difference between vanilla RNNs and Long-Short Term Memory (LSTM) RNNs
#    理解普通循环神经网络和长短记忆(Long-Short Term Memory)循环神经网络之间的差异


# 3. Understand how to sample from an RNN at test-time
#    理解在测试时如何从RNN生成序列


# 4. Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
#    理解如何将卷积神经网络和循环神经网络结合在一起来实现图像标注


# 5. Understand how a trained convolutional network can be used to compute gradients with respect to the input image
#    理解一个训练过的卷积神经网络是如何用来从输入图像中计算梯度的

# 6. Implement and different applications of image gradients,
#    including saliency maps, fooling images, class visualizations, feature inversion, and DeepDream.
#    实现图像梯度的不同应用，比如显著图，搞笑图像，类别可视化，特征反演和DeepDream
