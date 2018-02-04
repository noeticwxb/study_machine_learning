import numpy as np
import math
import random

def load_dataset(filename:str)->(list, list):
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_array = line.strip().split("\t")
        data_mat.append([float(line_array[0]),float(line_array[1])])
        label_mat.append(float(line_array[2]))
    return data_mat,label_mat


def select_rand_J(i,n):
    """
    return a random integer J smaller than n and not equal i
    :param i:
    :param n:
    :return:
    """
    j = i
    while (j==i):
        j = random.uniform(0,n)
    return j


def clamp(val,low,high):
    temp = [low,high]
    low =min(temp)
    high = max(temp)
    if(val < low):
        val = low
    if(val > high):
        val = high
    return val


def simple_svm(train_features, train_lables, C , tolerance, max_iter) -> (np.matrix,float):
    feature_mat = np.mat(train_features)
    sample_count, feaure_count = np.shape(feature_mat)
    label_mat = np.mat(train_lables).transpose() # sharp: sample_count * 1
    b = 0
    alphas = np.mat(np.ones((sample_count,1)))
    itor = 0
    while(itor < max_iter):
        alpha_pair_changed_count = 0
        for i in range(sample_count):
            x_i = feature_mat[i:].transpose()
            y_i = float(label_mat[i])
            f_x_i = float( np.multiply(label_mat,alphas).transpose() * (feature_mat * x_i) ) +  b
            error_i = f_x_i - y_i
            if ( ( (y_i*error_i < -tolerance) and  (alphas[i] < C) ) or ( ( y_i*error_i > tolerance ) and (alphas[i] > 0) ) ):
                j = select_rand_J(i,sample_count)
                x_j = feature_mat[j:].transpose()
                y_j = float(label_mat[j])
                f_x_j = float( np.multiply(label_mat,alphas).transpose() * (feature_mat * x_j) ) +  b
                error_j =  f_x_j - y_j
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if(y_i != y_j):
                    L = max(0,alpha_j_old - alpha_i_old)
                    H = min(C,C + alpha_j_old-alpha_i_old)
                else:
                    L = max(0,alpha_j_old+alpha_i_old-C)
                    H = min(C, alpha_j_old+alpha_i_old)
                if L==H:
                    print("L==H")
                    continue
                k_i_i = x_i * x_i.transpose()
                k_j_j = x_j * x_j.transpose()
                k_i_j = x_i * x_j.transpose()
                eta = k_i_i + k_j_j - 2 * k_i_j
                if(eta <= 0 ):
                    print("eta <= 0")
                    continue
                else:
                    alpha_j_new = alpha_j_old + y_j * (error_i-error_j) / eta
                    alpha_j_new = clamp(alpha_j_new,H,L)
                if(abs(alpha_j_new-alpha_j_old) < 0.0001):
                    print("alpha_j_new equal alpha_j_old")
                    continue
                alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
                b_i = b - error_i - y_i * k_i_i * (alpha_i_new - alpha_i_old) - y_j * k_i_j * (alpha_j_new - alpha_j_old)
                b_j = b - error_j - y_i * k_i_j * (alpha_i_new - alpha_i_old) - y_j * k_j_j * (alpha_j_new - alpha_j_old)
                if( (alpha_i_new > 0 ) and (alpha_i_new < C)):
                    b = b_i
                elif( ( alpha_j_new > 0 ) and (alpha_j_new < C) ):
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new
                alpha_pair_changed_count += 1

        if(alpha_pair_changed_count == 0):
            itor += 1
        else:
            itor = 0
        print("iteration numeber: %d" % itor)
    return  alphas,b


def plot(data_mat:list, label_mat:list, weights:list):
    import matplotlib.pyplot as plt
    xcoord_1 = [];
    ycoord_1 = []
    xcoord_2 = [];
    ycoord_2 = []
    for i in range(len(label_mat)):
        if math.isclose(label_mat[i],1):
            xcoord_1.append(data_mat[i][0])
            ycoord_1.append(data_mat[i][1])
        else:
            xcoord_2.append(data_mat[i][0])
            ycoord_2.append(data_mat[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord_1, ycoord_1, c='red')
    ax.scatter(xcoord_2, ycoord_2, c='green')

    if weights != None:
        x_line = np.arange(-3.0, 3.0, 0.1)
        y_line = (-weights[0] - weights[1] * x_line) / weights[2]
        y_line = np.asarray(y_line).flatten()
        ax.plot(x_line, y_line.flatten())

    plt.show()


if __name__ == "__main__":
    features, lables = load_dataset("data/ch06/testSet.txt")
    #plot(features,lables,None)
    alphas,b = simple_svm(features,lables,0.6,0.001,40)
