import io
import math
import numpy as np
import random

def load_database():#type: ()->(list,list)
    features = []; labels = []
    fr = io.open("data/ch05/testSet.txt")
    for line in fr.readlines():
        line_array = line.strip().split();
        features.append( [1,float(line_array[0]),float(line_array[1]) ] )
        labels.append( float(line_array[2]) )
    return features,labels

def sigmold(in_x):
    return 1 / (1 + np.exp(-in_x))


def grad_ascent(features,lables):#type:(list,list)->list
    """
    :param features: dataset features
    :param lables: dataset lable
    :return: param for liner sigmold.
    """
    alpha = 0.001
    max_cycle = 500
    feature_mat = np.matrix(features)
    lable_mat = np.matrix(lables).transpose()
    #m is count of sample. n is the count of feature
    m,n = np.shape(feature_mat)
    weights = np.ones((n,1)) #n*1
    #weights = np.zeros((n ,1)) # use ones,zeros ,get different result
    for i in range(max_cycle):
        h = sigmold(feature_mat * weights)  # m*1
        error = lable_mat - h #m*1
        weights = weights + alpha * (feature_mat.transpose()*error)
    return weights


def test_grad_ascent():
    features, labels = load_database()

    train_label = labels
    train_features = features
    test_label = []
    test_features = []
    test_count = int(len(train_label) * 0.2)
    for i in range(test_count):
        index = random.randrange(0,len(train_label))
        test_label.append(train_label[index])
        test_features.append(train_features[index])
        del train_label[index]
        del train_features[index]

    weights_mat = grad_ascent(train_features, train_label)  # n*1

    #weights_mat = np.matrix([4.12414349,0.48007329,-0.6168482]).transpose()
    features_mat = np.matrix(test_features) #m*n
    ret = features_mat * weights_mat
    ret = sigmold(ret) # m*1
    ret = np.where(ret>=0.5,1,0).flatten() #type:list

    ret_len = len(ret)
    error_count = 0
    for i_ret in range(ret_len):
        if test_label[i_ret] != ret[i_ret]:
            error_count += 1
    print "%.2f%% error" % (float(error_count) *100 /float(ret_len))


def plot_best_fit():
    import matplotlib.pyplot as plt

    features, labels = load_database()
    weight = grad_ascent(features[:],labels[:])
    #print features
    xcoord_1 = []; ycoord_1 = []
    xcoord_2 = []; ycoord_2 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            xcoord_1.append(features[i][1])
            ycoord_1.append(features[i][2])
        else:
            xcoord_2.append(features[i][1])
            ycoord_2.append(features[i][2])

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.scatter(xcoord_1,ycoord_1,c='red')
    ax.scatter(xcoord_2, ycoord_2, c='green')

    x_line = np.arange(-3.0,3.0,0.1)
    y_line = (-weight[0]-weight[1]*x_line)/weight[2]
    y_line = np.asarray(y_line).flatten()
    ax.plot(x_line,y_line.flatten())

    plt.show()


if __name__ == '__main__':
    #features, labels = load_database()
    #grad_ascent(features,labels)
    #test_grad_ascent()
    plot_best_fit()


