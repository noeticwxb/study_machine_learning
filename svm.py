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
        j = int(random.uniform(0,n))
    return int(j)


def clamp(val,low,high):
    temp = [low,high]
    low =min(temp)
    high = max(temp)
    if(val < low):
        val = low
    if(val > high):
        val = high
    return val


def simple_svm(train_features, train_lables, C , tolerance, max_iter) -> (np.matrix,float,np.matrix):
    feature_mat = np.mat(train_features)
    sample_count, feaure_count = np.shape(feature_mat)
    label_mat = np.mat(train_lables).transpose() # sharp: sample_count * 1
    b = 0
    alphas = np.mat(np.zeros((sample_count,1)))
    itor = 0
    while(itor < max_iter):
        alpha_pair_changed_count = 0
        for i in range(sample_count):
            x_i = feature_mat[i,:].transpose()
            y_i = float(label_mat[i])
            f_x_i = float( np.multiply(label_mat,alphas).transpose() * (feature_mat * x_i) ) +  b
            error_i = f_x_i - y_i
            if ( ( (y_i*error_i < -tolerance) and  (alphas[i] < C) ) or ( ( y_i*error_i > tolerance ) and (alphas[i] > 0) ) ):
                j = select_rand_J(i,sample_count)
                x_j = feature_mat[j,:].transpose()
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
                    #print("L==H")
                    continue
                k_i_i = float(x_i.transpose() * x_i)
                k_j_j = float(x_j.transpose() * x_j)
                k_i_j = float(x_j.transpose() * x_i)
                eta = k_i_i + k_j_j - 2 * k_i_j
                if(eta <= 0 ):
                    #print("eta <= 0")
                    continue
                else:
                    alpha_j_new = alpha_j_old + y_j * (error_i-error_j) / eta
                    alpha_j_new = clamp(alpha_j_new,H,L)
                if(abs(alpha_j_new-alpha_j_old) < 0.0001):
                    #print("alpha_j_new equal alpha_j_old")
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
            print("alpha_pair_changed_count numeber: %d" % alpha_pair_changed_count)
        print("iteration numeber: %d" % itor)

    w = np.multiply(alphas, label_mat)
    w = np.multiply(w,feature_mat)
    w = np.sum(w,axis=0)
    return  alphas, b, w


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
        x_line = np.arange(0.0, 10.0, 0.1)
        y_line = (-weights[0] - weights[1] * x_line) / weights[2]
        y_line = np.asarray(y_line).flatten()
        ax.plot(x_line, y_line.flatten())

    plt.show()


class PlattSVM:
    def __init__(self,data_set,class_label,C,tolerance):
        self.feature_mat = np.mat(data_set)
        self.label_mat = np.mat(class_label).transpose()
        self.C = C
        self.tolerance = tolerance
        self.sample_count = np.shape(self.feature_mat)[0]
        self.alphas = np.mat(np.zeros((self.sample_count,1)))
        self.b = 0
        self.cache = np.mat(np.zeros((self.sample_count,2)))

    def calc_error(self,i):
        x_i = self.feature_mat[i, :].transpose()
        y_i = float(self.label_mat[i])
        f_x_i = float(np.multiply(self.label_mat, self.alphas).transpose() * (self.feature_mat * x_i)) + self.b

        error_i = f_x_i - y_i
        return error_i

    def select_rand_J(self,i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.sample_count))
        return j

    def select_J(self,i,error_i):
        max_index = -1
        max_delta_error = 0
        max_error = 0
        # 这个cache的设置过程很难理解。
        self.cache[i] = [1,error_i]
        valid_cache_index_array = np.nonzero(self.cache[:,0].A)[0]
        if len(valid_cache_index_array) > 1 :
            for k in valid_cache_index_array:
                if k == i:
                    continue
                error_k = self.calc_error(k)
                delta_error = abs(error_i-error_k)
                if delta_error > max_delta_error:
                    max_index = k
                    max_delta_error = delta_error
                    max_error = error_k
        else:
            max_index = self.select_rand_J(i)
            max_error = self.calc_error(max_index)
        return max_index,max_error

    def update_cache(self,i):
        error_i = self.calc_error(i)
        self.cache[i] = [1,error_i]

    def examine_example(self,i):
        x_i = self.feature_mat[i, :].transpose()
        y_i = float(self.label_mat[i])
        error_i = self.calc_error(i)
        if (((y_i * error_i < - self.tolerance) and (self.alphas[i] < self.C)) or ((y_i * error_i > self.tolerance) and (self.alphas[i] > 0))):
            j,error_j = self.select_J(i, error_i)
            x_j = self.feature_mat[j, :].transpose()
            y_j = float(self.label_mat[j])
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            if (y_i != y_j):
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_j_old + alpha_i_old - self.C)
                H = min(self.C, alpha_j_old + alpha_i_old)
            if L == H:
                #print("L==H")
                return 0
            k_i_i = float(x_i.transpose() * x_i)
            k_j_j = float(x_j.transpose() * x_j)
            k_i_j = float(x_j.transpose() * x_i)
            eta = k_i_i + k_j_j - 2 * k_i_j
            if (eta <= 0):
                #print("eta <= 0")
                return 0
            else:
                alpha_j_new = alpha_j_old + y_j * (error_i - error_j) / eta
                alpha_j_new = clamp(alpha_j_new, H, L)
            if (abs(alpha_j_new - alpha_j_old) < 0.0001):
                #print("alpha_j_new equal alpha_j_old")
                return 0
            alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
            b_i = self.b - error_i - y_i * k_i_i * (alpha_i_new - alpha_i_old) - y_j * k_i_j * (alpha_j_new - alpha_j_old)
            b_j = self.b - error_j - y_i * k_i_j * (alpha_i_new - alpha_i_old) - y_j * k_j_j * (alpha_j_new - alpha_j_old)
            if ((alpha_i_new > 0) and (alpha_i_new < self.C)):
                self.b = b_i
            elif ((alpha_j_new > 0) and (alpha_j_new < self.C)):
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2

            #print("i=%d, %f,    j=%d, %f" %(i,j,alpha_i_new,alpha_j_new))
            self.alphas[i] = alpha_i_new
            self.alphas[j] = alpha_j_new

            #self.update_cache(i)
            #self.update_cache(j)

            return 1
        else:
            return 0

    def examine_example_simple(self, i):
        x_i = self.feature_mat[i, :].transpose()
        y_i = float(self.label_mat[i])
        error_i = self.calc_error(i)
        if (((y_i * error_i < -self.tolerance) and (self.alphas[i] < self.C)) or ((y_i * error_i > self.tolerance) and (self.alphas[i] > 0))):
            j = self.select_rand_J(i)
            error_j = self.calc_error(j)
            x_j = self.feature_mat[j, :].transpose()
            y_j = float(self.label_mat[j])
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            if (y_i != y_j):
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_j_old + alpha_i_old - self.C)
                H = min(self.C, alpha_j_old + alpha_i_old)
            if L == H:
                #print("L==H")
                return 0
            k_i_i = float(x_i.transpose() * x_i)
            k_j_j = float(x_j.transpose() * x_j)
            k_i_j = float(x_j.transpose() * x_i)
            eta = k_i_i + k_j_j - 2 * k_i_j
            if (eta <= 0):
                #print("eta <= 0")
                return 0
            else:
                alpha_j_new = alpha_j_old + y_j * (error_i - error_j) / eta
                alpha_j_new = clamp(alpha_j_new, H, L)
            if (abs(alpha_j_new - alpha_j_old) < 0.0001):
                # print("alpha_j_new equal alpha_j_old")
                return 0
            alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
            b_i = self.b - error_i - y_i * k_i_i * (alpha_i_new - alpha_i_old) - y_j * k_i_j * (alpha_j_new - alpha_j_old)
            b_j = self.b - error_j - y_i * k_i_j * (alpha_i_new - alpha_i_old) - y_j * k_j_j * (alpha_j_new - alpha_j_old)
            if ((alpha_i_new > 0) and (alpha_i_new < self.C)):
                self.b = b_i
            elif ((alpha_j_new > 0) and (alpha_j_new < self.C)):
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2
            self.alphas[i] = alpha_i_new
            self.alphas[j] = alpha_j_new
            return 1
        else:
            return 0

    def svm(self,max_itor=1):
        num_alpha_pair_changed = 0
        itor = 0
        example_all = True
        while( (itor < max_itor ) and ( (num_alpha_pair_changed > 0 ) or example_all ) ):
            num_alpha_pair_changed = 0
            if example_all:
                for i in range(self.sample_count):
                    num_alpha_pair_changed += self.examine_example(i)
                print(" full set: itor:%d, pairs changed:%d" % (itor,num_alpha_pair_changed))
                itor+=1
            else:
                non_bounds = np.nonzero((self.alphas.A > 0 ) * (self.alphas.A < self.C))[0]
                for i in non_bounds:
                    num_alpha_pair_changed += self.examine_example(i)
                print(" non bounds set: itor:%d, pairs changed:%d"%(itor, num_alpha_pair_changed))
                itor += 1
            if example_all:
                example_all = False
            elif(num_alpha_pair_changed==0):
                example_all = True

        #print(self.alphas)
        w = np.multiply(self.alphas, self.label_mat)
        w = np.multiply(w,self.feature_mat)
        w = np.sum(w,axis=0)

        weights = [float(self.b)]
        weights.extend(np.asarray(w)[0].tolist())
        return  weights

    def simple_svm(self,max_itor=1):
        itor = 0
        while (itor < max_itor):
            alpha_pair_changed_count = 0
            for i in range(self.sample_count):
                alpha_pair_changed_count += self.examine_example_simple(i)

            if (alpha_pair_changed_count == 0):
                itor += 1
                print("iteration numeber: %d" % itor)
            else:
                print("alpha_pair_changed_count:%d" % alpha_pair_changed_count)
                itor = 0


        w = np.multiply(self.alphas, self.label_mat)
        w = np.multiply(w, self.feature_mat)
        w = np.sum(w, axis=0)
        weights = [float(self.b)]
        weights.extend(np.asarray(w)[0].tolist())
        return weights

if __name__ == "__main__":
    features, lables = load_dataset("data/ch06/testSet.txt")
    use_simple_svm = False
    platt_svm = PlattSVM(features, lables, 0.6, 0.001)
    if use_simple_svm:
        weights = platt_svm.simple_svm(100)
        plot(features, lables, weights)
    else:
        weights = platt_svm.svm(100)
        plot(features, lables, weights)

