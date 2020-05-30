# student_name:Yongheng Hou
# student_NO:5556661
# login:yh790
import pandas as pd
from numpy import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def calAccuracy(pred, label):
    acc = 0.0
    for i in range(len(pred)):
        if (pred[i] == label[i][-1]):
            acc += 1
    return acc / len(pred)


def classify2(node, dataSet):
    if not isinstance(node, dict):
        return np.argmax(node)

    if (dataSet[node['spInd']] < node['spVal']):
        C = classify2(node['left'], dataSet)
    else:
        C = classify2(node['right'], dataSet)

    return C


def classify1(node, dataSet):
    result = np.zeros(len(dataSet))
    for i in range(len(dataSet)):
        sample = dataSet[i]
        result[i] = classify2(node, sample)

    return result


def getMean(node):
    if isinstance(node['left'], dict):
        node['left'] = getMean(node['left'])

    if isinstance(node['right'], dict):
        node['right'] = getMean(node['right'])

    return (node['right'] + node['left']) / (2.0)


def Gini(dataSet):
    numOfG = dataSet[:, -1].sum()
    pG = numOfG / len(dataSet)
    pH = 1 - pG

    giniG = 1 - pG * pG
    giniH = 1 - pH * pH

    return giniG + giniH


def ShannoEnt(dataSet):
    numOfG = dataSet[:, -1].sum()
    pG = numOfG / len(dataSet)
    pH = 1 - pG

    if (pH == 0 or pG == 0):
        return 0
    shannonEnt = -pH * np.log2(pH) - pG * np.log2(pG)

    return shannonEnt


def BinarySplit(dataSet, featIndex, splitVal):
    subData0 = dataSet[dataSet[:, featIndex] < splitVal]
    subData1 = dataSet[dataSet[:, featIndex] >= splitVal]

    return subData0, subData1


# new entropy
def NewS(subData0, subData1, criteria):
    if (len(subData0) == 0 or len(subData1) == 0):
        return inf

    total = len(subData0) + len(subData1)
    p0 = len(subData0) / total
    p1 = len(subData1) / total

    if criteria == 'VA':
        newS = np.var(subData0[:, -1]) * p0 + np.var(subData1[:, -1]) * p1

    elif criteria == 'GI':
        newS = Gini(subData0) * p0 + Gini(subData1) * p1
    else:
        newS = ShannoEnt(subData0) * p0 + ShannoEnt(subData1) * p1

    return newS


def BaseS(dataSet, criteria):
    if criteria == 'VA':
        S = np.var(dataSet[:, -1])
    elif criteria == 'GI':
        S = Gini(dataSet)
    else:
        S = ShannoEnt(dataSet)

    return S


def chooseBestSplit_GR(dataSet, ops=(0.05, 4)):
    tolS = ops[0]  # early stoping for information gain but didint implement
    tolN = ops[1]  # early stoping for number of coloumn
    n = dataSet.shape[1]
    baseS = ShannoEnt(dataSet)  # base entropy

    bestGR = -inf;
    bestIndex = 0;
    bestSplitVal = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):

            subDS0, subDS1 = BinarySplit(dataSet, featIndex, splitVal)

            if (np.shape(subDS0)[0] < tolN) or (np.shape(subDS1)[0] < tolN):
                continue

            total = len(subDS0) + len(subDS1)
            p0 = len(subDS0) / len(dataSet)
            p1 = len(subDS1) / len(dataSet)

            newS = ShannoEnt(subDS0) * p0 + ShannoEnt(subDS1) * p1

            splitInfo = - p0 * np.log2(p0) - p1 * np.log2(p1)

            GR = (baseS - newS) / splitInfo

            if (GR > bestGR):
                bestGR = GR
                bestIndex = featIndex
                bestSplitVal = splitVal

    # check valid split again and return bestIndex, bestSplitValue

    subDS0, subDS1 = BinarySplit(dataSet, bestIndex, bestSplitVal)

    if (len(subDS0) == 0 or len(subDS1) == 0):
        return None, 0

    return bestIndex, bestSplitVal


def chooseBestSplit_MIG_MVA_MGI(dataSet, criteria, ops=(0.05, 4)):
    tolS = ops[0]  # early stoping for information gain but didint implement
    tolN = ops[1]  # early stoping for number of coloumn
    n = dataSet.shape[1]
    S = BaseS(dataSet, criteria)  # base entropy
    bestS = inf;
    bestIndex = 0;
    bestSplitVal = 0
    # call for lops to iterate each available split condition,then calculate thier entropy,
    # then calculate thier criterion，such asinformaton gain,finally contrast thier criterion to choose best splition
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            # call binary split to choose what sub data
            subDS0, subDS1 = BinarySplit(dataSet, featIndex, splitVal)

            newS = NewS(subDS0, subDS1, criteria)
            # contrast
            if (newS < bestS):
                bestS = newS
                bestIndex = featIndex
                bestSplitVal = splitVal

    if (S - bestS) < tolS:
        return None, 0

    # check valid split again and return bestIndex, bestSplitValue

    subDS0, subDS1 = BinarySplit(dataSet, bestIndex, bestSplitVal)

    if (np.shape(subDS0)[0] < tolN) or (np.shape(subDS1)[0] < tolN):
        return None, 0

    return bestIndex, bestSplitVal


def induction(dataSet, criteria):
    n = {}
    total = dataSet[:, -1].sum()
    freqClasses = np.array([len(dataSet) - total, total])
    # If it has only one category, the split is stopped
    if len(set(dataSet[:, -1])) == 1:
        return dataSet[0, -1]
    # judge which criteria it belongs to
    # the data divided may not all belong to a class, at this time, the classification of the sub-data set needs to be determined according to the majority voting rule.
    if criteria == 'GR':
        bestIndex, bestSplitVal = chooseBestSplit_GR(dataSet)
    else:
        bestIndex, bestSplitVal = chooseBestSplit_MIG_MVA_MGI(dataSet, criteria)
    # no feature ,so stop slit
    if (bestIndex == None):
        return freqClasses

    # call binary split to choose what sub data
    subDataLeft, subDataRight = BinarySplit(dataSet, bestIndex, bestSplitVal)

    n['spInd'] = bestIndex
    n['spVal'] = bestSplitVal

    n_chaild_l = induction(subDataLeft, criteria)
    n['left'] = n_chaild_l

    n_chaild_r = induction(subDataRight, criteria)
    n['right'] = n_chaild_r

    return n


def postPrune(node, prune_data):
    # if it is leaf .return
    if not isinstance(node, dict):
        return node

    if (len(prune_data) == 0):
        return getMean(node)
    # prune
    subLeft, subRight = BinarySplit(prune_data, node['spInd'], node['spVal'])
    # if it is node,continue prune
    if (isinstance(node['left'], dict)):
        node['left'] = postPrune(node['left'], subLeft)
    # if it is node,continue prune
    if (isinstance(node['right'], dict)):
        node['right'] = postPrune(node['right'], subRight)
    # if it is leaf,do classify
    if not isinstance(node['left'], dict) and not isinstance(node['right'], dict):

        predictNoMerge = classify1(node, prune_data)
        noMergeAcc = calAccuracy(predictNoMerge, prune_data)

        mergeNode = getMean(node)
        predictMerge = classify1(mergeNode, prune_data)
        mergeAcc = calAccuracy(predictMerge, prune_data)
        # do contrast betwwen merge accacuy and no merge accurracy
        if (mergeAcc > noMergeAcc):
            return mergeNode

    return node


class DecisionTree:
    def __init__(self, train_data, criteria):
        self.train_data = train_data
        self.criteria = criteria
        self.root = induction(train_data, criteria)

    def prune(self, prune_data):
        self.root = postPrune(self.root, prune_data)

    def predict(self, test_data):
        return classify1(self.root, test_data)

    def calAcc(self, test_data):
        predict = self.predict(test_data)
        return calAccuracy(predict, test_data)


class DT_Ensembled:
    # intialize three kinds of tree
    def __init__(self, train_data):
        self.M_IG = DecisionTree(train_data, 'IG')
        self.M_GR = DecisionTree(train_data, 'GR')
        self.M_VA = DecisionTree(train_data, 'VA')

    # preict is to preduct the class of data by calculation resul.the calculation between three trees
    def predict(self, test_data):
        predict_IG = self.M_IG.predict(test_data)
        predict_GR = self.M_GR.predict(test_data)
        predict_VA = self.M_VA.predict(test_data)

        result = predict_IG + predict_GR + predict_VA
        result[result < 2] = 0
        result[result >= 2] = 1

        return result

    # call prune function of decsion tree calss to prune.
    def prune(self, prune_data):
        self.M_IG.prune(prune_data)
        self.M_GR.prune(prune_data)
        self.M_VA.prune(prune_data)

    # calculate the accurracy
    def calAcc(self, test_data):
        predict = self.predict(test_data)
        return 1 - calAccuracy(predict, test_data)


def ten_fold_cross_validation():
    # define the number of folds
    n_folds = 10

    fold_len = int(len(dataSet) / n_folds)
    maskFold = np.ones(dataSet.shape[0], dtype=bool)

    startIdx = 0
    endIdx = fold_len
    # return nflods 0_array
    x = np.zeros(n_folds)
    y = np.zeros(n_folds)
    # for loop for 10 times
    for i in range(n_folds):

        if (i == n_folds - 1):
            endIdx = len(dataSet)

        maskFold[startIdx:endIdx] = False
        # use mask to get training and test set
        train_data = dataSet[maskFold]
        test_data = dataSet[~maskFold]

        M_star = DT_Ensembled(train_data)

        Acc_M_star = M_star.calAcc(test_data)

        M_GI = DecisionTree(train_data, 'GI')

        Acc_M_GI = M_GI.calAcc(test_data)

        x[i] = Acc_M_star
        y[i] = Acc_M_GI

        maskFold[startIdx:endIdx] = True
        startIdx = endIdx
        endIdx += fold_len


def postPruneFor_MGI():
    # stratify split train_data to 1/3,test_data to 2/3
    train_data, test_data = train_test_split(df, stratify=df['class'], test_size=2 / 3)
    # stratify split test data to 1/3,prune data to 1/3
    prune_data, test_data = train_test_split(test_data, stratify=test_data['class'], test_size=1 / 2)
    # tranfer df to array
    train_data = train_data.values
    prune_data = prune_data.values
    test_data = test_data.values

    M_GI = DecisionTree(train_data, 'GI')

    label = test_data[:, -1]

    pred = M_GI.predict(test_data)

    print("before pruning : ")

    conf_mat = np.zeros([2, 2])
    for i in range(len(pred)):
        row = int(1 - label[i])
        col = int(1 - pred[i])
        conf_mat[row][col] += 1

    TP = conf_mat[0][0]
    FP = conf_mat[1][0]
    FN = conf_mat[0][1]
    TN = conf_mat[1][1]
    P = conf_mat[0].sum()
    N = conf_mat[1].sum()
    All = P + N

    print("\t", conf_mat[0])
    print("\t", conf_mat[1])
    print("\tTPR: ", TP / P)
    print("\tFPR: ", FP / N)
    print("\tAcc: ", (TP + TN) / All)
    print("-------------------------")
    print()

    M_GI.prune(prune_data)

    pred = M_GI.predict(test_data)

    print("after pruning : ")
    conf_mat = np.zeros([2, 2])
    for i in range(len(pred)):
        row = int(1 - label[i])
        col = int(1 - pred[i])
        conf_mat[row][col] += 1

    TP = conf_mat[0][0]
    FP = conf_mat[1][0]
    FN = conf_mat[0][1]
    TN = conf_mat[1][1]
    P = conf_mat[0].sum()
    N = conf_mat[1].sum()
    All = P + N

    print("\t", conf_mat[0])
    print("\t", conf_mat[1])
    print("\tTPR: ", TP / P)
    print("\tFPR: ", FP / N)
    print("\tAcc: ", (TP + TN) / All)
    print("-------------------------")
    print()


# -----------------------------main-----------------------------
headers = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans',
           'fAlpha', 'fDist', 'class']

df = pd.read_csv("magic04.data", header=None, names=headers)

# preprocess
df.hist(column=headers)
plt.show()
print(df.describe())

# change value of class, g to 0,g to 1.
df.loc[df["class"] == "h", 'class'] = 0
df.loc[df["class"] == "g", 'class'] = 1

# normaination for column 'fAlpha','fLength','fWidth'
from sklearn import preprocessing

x = df[['fAlpha', 'fLength', 'fWidth']].values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df[['fAlpha', 'fLength', 'fWidth']] = pd.DataFrame(x_scaled)
df.hist(column=headers)
plt.show()

# reduce set,can set ratio,0.01 mean extract 1/10 of whole dataset,but I used random select,tried best to reduce influence
ratio = 0.1
shuffled_indices = np.random.permutation(len(df))
test_set_size = int(len(df) * ratio)
test_indices = shuffled_indices[:test_set_size]
new_df = df.loc[test_indices]

# convert to numpy type
dataSet = new_df.values

ten_fold_cross_validation()

postPruneFor_MGI()