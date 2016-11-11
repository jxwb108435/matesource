# coding=utf-8
# -*- coding:cp936 -*-
import numpy as np
import matplotlib.pyplot as plt
from cs231n_util import load_CIFAR10, KNearestNeighbor

''' k-Nearest Neighbor分类器 '''
# Nearest Neighbor算法将会拿着测试图片和训练集中每一张图片去比较, 然后将它认为最相似的那个训练集图片的标签赋给这张测试图片
# 将两张图片先转化为两个向量I_1和I_2, 然后计算他们的L1距离:
# distances L1 = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
# distances L2 = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
# 面对两个向量之间的差异时, L2比L1更加不能容忍这些差异, 也就是说, 相对于1个巨大的差异, L2距离更倾向于接受多个中等程度的差异
# 逐个像素求差值,然后将所有差值的绝对值加起来得到一个数值
# 如果两张图片一模一样, 那么L1距离为0, 如果两张图片很是不同, 那L1值将会非常大


# 与其只找最相近的那1个图片的标签, 我们找最相似的k个图片的标签, 然后让他们针对测试图片进行投票, 最后把票数最高的标签作为对测试图片的预测
# 更高的k值可以让分类的效果更平滑, 使得分类器对于异常值更有抵抗力
# 在实际中, 大多使用k-NN分类器

# k-NN分类器需要设定k值, 那么选择哪个k值最合适的呢? 我们可以选择不同的距离函数, 比如L1范数和L2范数等,
# 那么选哪个好? 还有不少选择我们甚至连考虑都没有考虑到(比如:点积)
# 所有这些选择，被称为超参数(hyperparameter)

# 建议尝试不同的值, 看哪个值表现最好就选哪个
# 决不能使用测试集来进行调优,
# 危险在于: 算法实际部署后, 性能可能会远低于预期, 这种情况, 称之为算法对测试集过拟合
# 实际上就是把测试集当做训练集, 由测试集训练出来的算法再跑测试集, 自然性能看起来会很好, 这其实过于乐观了，实际部署起来效果就会差很多
# 测试数据集只使用一次, 即在训练完成后评价最终的模型时使用, 最终测试的时候再使用测试集, 可以很好地近似度量分类器的泛化性能

# 从训练集中取出一部分数据用来调优, 我们称之为验证集(validation set)


# find hyperparameters that work best on the validation set


# 程序结束后, 我们会作图分析出哪个k值表现最好, 然后用这个k值来跑真正的测试集, 并作出对算法的评价
# 把训练集分成训练集和验证集, 使用验证集来对所有超参数调优,最后只在测试集上跑一次并报告结果

# 交叉验证, 有时候训练集数量较小, 会使用一种被称为交叉验证的方法
# 将训练集平均分成5份, 其中4份用来训练, 1份用来验证
# 循环取其中4份来训练, 1份验证, 最后取所有5次验证结果的平均值作为算法验证结果

# 在实际情况下, 人们不是很喜欢用交叉验证，主要是因为它会耗费较多的计算资源
# 一般直接把训练集按照50%-90%的比例分成训练集和验证集。
# 但这也是根据具体情况来定的: 如果超参数数量多, 你可能就想用更大的验证集, 而验证集的数量不够, 那么最好还是用交叉验证吧
# 至于分成几份比较好, 一般都是分成3,5和10份

# Nearest Neighbor分类器的优劣
# 首先, Nearest Neighbor分类器易于理解, 实现简单
# 其次, 算法的训练不需要花时间, 因为其训练过程只是将训练集数据存储起来
# 然而测试要花费大量时间计算, 因为每个测试图像需要和所有存储的训练图像进行比较, 这显然是一个缺点
# 在实际应用中, 我们关注测试效率远远高于训练效率
# 其实, 我们后续要学习的卷积神经网络在这个权衡上走到了另一个极端: 虽然训练花费很多时间, 但是一旦训练完成, 对新的测试数据进行分类非常快
# 这样的模式就符合实际使用需求

# Nearest Neighbor分类器在某些特定情况(比如数据维度较低)下, 可能是不错的选择
# 但是在实际的图像分类工作中, 很少使用, 因为图像都是高维度数据(他们通常包含很多像素), 而高维度向量之间的距离通常是反直觉的


# Load the raw CIFAR-10 data.
X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance.
# training a kNN classifier is a noop: simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# compute_distances_two_loops.
# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# We can visualize the distance matrix: each row is a single test example and its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)
# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference,))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference,))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


# Let's compare how fast the implementations are
def time_function(f, *args):
    """  Call a function f with args and return the time (in seconds) that it took to execute.  """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation


''' Cross-validation '''

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array(np.array_split(X_train, num_folds))
y_train_folds = np.array(np.array_split(y_train, num_folds))
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################

for k_alter in k_choices:
    k_to_accuracies[k_alter] = []
    for i in range(num_folds):
        X_train_folds_4 = np.concatenate(X_train_folds[np.where(np.arange(num_folds) != i)[0]])
        y_train_folds_4 = np.concatenate(y_train_folds[np.where(np.arange(num_folds) != i)[0]])
        X_validation = np.concatenate(X_train_folds[np.where(np.arange(num_folds) == i)[0]])
        y_validation = np.concatenate(y_train_folds[np.where(np.arange(num_folds) == i)[0]])
        classifier.train(X_train_folds_4, y_train_folds_4)
        y_validation_pred = classifier.predict(X_validation, k_alter)
        num_correct = np.sum(y_validation_pred == y_validation)
        accuracy = float(num_correct) / y_validation.shape[0]
        k_to_accuracies[k_alter].append(accuracy)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


''' 从图像到标签分值的参数化映射 '''

# 一种更强大的方法来解决图像分类问题, 这种方法主要有两部分组成:
# 一个是评分函数(score function), 它是原始图像数据到类别分值的映射
# 另一个是损失函数(loss function), 它是用来量化预测分类标签的得分与真实标签之间一致性的
# 该方法可转化为一个最优化问题, 在最优化过程中, 将通过更新评分函数的参数来最小化损失函数值

# 从图像到标签分值的参数化映射
# 定义一个评分函数, 函数将图像的像素值映射为各个分类类别的得分, 得分高低代表图像属于该类别的可能性高低

# 假设有一个包含很多图像的训练集{xi, yi}, 这里i=1,2,...,N并且yi = 1,...,K
# 就是说, 我们有N个图像样例,每个图像的维度是D, 共有K种不同的分类

# 举例来说, 在CIFAR-10中, 我们有一个N=50000的训练集, 每个图像有D=32x32x3=3072个像素
# 而K=10, 因为图片被分为10个不同的类别(狗，猫，汽车等)
# 现在定义评分函数为: f:R^D -> R^K, 该函数是原始图像像素到分类分值的映射
