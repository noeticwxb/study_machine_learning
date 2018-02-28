import numpy as np
import math
import random
import os
import zipfile

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


def kernal_trans(feature_mat:np.matrix, feature:np.matrix, kTup:list):
    m,n = np.shape(feature_mat)
    k = np.mat(np.zeros((m,1)))
    if kTup[0] == "linear":
        k = feature_mat * feature.T
    elif kTup[0] == "rbf":
        for j in range(m):
            diff = feature_mat[j] - feature
            k[j] = diff * diff.T
        #括号没打对，调试了好久。太低级了
        #k = np.exp(k / -1 * (kTup[1] ** 2))
        k = np.exp( k / (-1 * (kTup[1]**2)))
    else:
        raise NameError("huston,wo have a problem. unknown kernel")
    return  k


class PlattSVM:
    def __init__(self,data_set,class_label,C,tolerance,kTup):
        self.feature_mat = np.mat(data_set)
        self.label_mat = np.mat(class_label).transpose()
        self.C = C
        self.tolerance = tolerance
        self.sample_count = np.shape(self.feature_mat)[0]
        self.alphas = np.mat(np.zeros((self.sample_count,1)))
        self.b = 0
        self.cache = np.mat(np.zeros((self.sample_count,2)))
        self.K = np.mat(np.zeros((self.sample_count,self.sample_count)))
        for i in range(self.sample_count):
            self.K[:,i] = kernal_trans(self.feature_mat,self.feature_mat[i],kTup)


    def calc_error(self,i):
        y_i = float(self.label_mat[i])
        f_x_i = float(np.multiply(self.label_mat, self.alphas).transpose() * (self.K[:,i])) + self.b

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
            k_i_i = float(self.K[i,i])
            k_j_j = float(self.K[j,j])
            k_i_j = float(self.K[i,j])
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
            k_i_i = float(self.K[i,i])
            k_j_j = float(self.K[j,j])
            k_i_j = float(self.K[i,j])
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

    def get_linear_weigets(self):
        w = np.multiply(self.alphas, self.label_mat)
        w = np.multiply(w, self.feature_mat)
        w = np.sum(w, axis=0)
        weights = [float(self.b)]
        weights.extend(np.asarray(w)[0].tolist())
        return weights

def test_linear():
    features, lables = load_dataset("data/ch06/testSet.txt")
    use_simple_svm = False
    kTup = ["linear"]
    platt_svm = PlattSVM(features, lables, 0.6, 0.001,kTup)

    if use_simple_svm:
        platt_svm.simple_svm(100)
        weights = platt_svm.get_linear_weigets()
        plot(features, lables, weights)
    else:
        platt_svm.svm(100)
        weights = platt_svm.get_linear_weigets()
        plot(features, lables, weights)

def test_none_linear(k1=1.4):
    features, lables = load_dataset("data/ch06/testSetRBF.txt")
    kTup = ["rbf",k1]
    platt_svm = PlattSVM(features, lables, 200, 0.001, kTup)
    platt_svm.svm(100)
    support_alpha_indecies = np.nonzero(platt_svm.alphas > 0 )[0]
    support_featues_mat = platt_svm.feature_mat[support_alpha_indecies]
    support_label_mat = platt_svm.label_mat[support_alpha_indecies]
    support_alphas = platt_svm.alphas[support_alpha_indecies]
    print("support vector %d " % np.shape(support_featues_mat)[0])

    m,n = np.shape(platt_svm.feature_mat)
    error_count = 0
    for i in range(m):
        kernal_eval = kernal_trans(support_featues_mat,platt_svm.feature_mat[i],kTup)
        temp:np.matrix = np.multiply(support_alphas,support_label_mat)
        predict = temp.T * kernal_eval + platt_svm.b
        if( np.sign(predict) != np.sign(lables[i])):
            error_count+=1
    print("training error rate is %f" % (float(error_count)/m) )
    test_features,test_labels = load_dataset("data/ch06/testSetRBF2.txt")
    test_feature_mat = np.mat(test_features)
    test_labels_mat = np.mat(test_labels)
    error_count = 0
    m, n = np.shape(test_feature_mat)
    for i in range(m):
        kernal_eval = kernal_trans(support_featues_mat,test_feature_mat[i],kTup)
        temp: np.matrix = np.multiply(support_alphas, support_label_mat)
        predict = temp.T * kernal_eval + platt_svm.b
        if( np.sign(predict) != np.sign(test_labels[i])):
            error_count+=1
    print("test error rate is %f" % (float(error_count)/m) )

def img_to_vector(zip_file,file_name): #type: (zipfile.ZipFile,str)->np.ndarray
    vec = np.zeros((1,1024))
    with zip_file.open(file_name) as zip_item_file:
        line_count = 0
        for line in zip_item_file:
            for i in range(32):
                num = int(line[i])-48
                vec[0,line_count*32+i]= num
            line_count += 1
    return vec

def hand_writing_data_set(zip_file,train_name_list):
    train_data_count = len(train_name_list)
    data_label = [0]*train_data_count
    data_set = np.zeros((train_data_count, 1024))
    for i in range(train_data_count):
        item_name = train_name_list[i]# type:str
        data_set[i] = img_to_vector(zip_file,item_name)
        label_name = item_name.replace("trainingDigits/","").replace(".txt","")
        label_name = label_name.replace("testDigits/", "").replace(".txt", "")
        lable_num = label_name.split("_")[0]
        lable_num = int(lable_num)
        data_label[i] = lable_num
    return data_set,data_label

def digital_to_label(digit_list,right_digit)->list:
    label = []
    for i in digit_list:
        if i == right_digit:
            label.append(1)
        else:
            label.append(-1)
    return label


def hand_writing_class_test(k1=0.01):
    abs_cur_dir = os.path.abspath(os.curdir)
    zip_file_name = os.path.join(abs_cur_dir,"data/ch06/digits.zip")
    with zipfile.ZipFile(zip_file_name,'r') as zip_file:
        name_list = zip_file.namelist()

        train_name_list = [x for x in name_list if x.startswith("trainingDigits") and x.endswith(".txt")]
        train_feature, train_digit = hand_writing_data_set(zip_file, train_name_list)
        train_label = digital_to_label(train_digit,9)

        #kTup = ["rbf", k1]
        #kTup = ["rbf",0.1]
        kTup = ["rbf", 100]

        platt_svm = PlattSVM(train_feature, train_label, 200, 0.0001, kTup)
        platt_svm.svm(100)

        support_alpha_indecies = np.nonzero(platt_svm.alphas > 0)[0]
        support_featues_mat = platt_svm.feature_mat[support_alpha_indecies]
        support_label_mat = platt_svm.label_mat[support_alpha_indecies]
        support_alphas = platt_svm.alphas[support_alpha_indecies]
        print("support vector %d / %d" % ( np.shape(support_featues_mat)[0],platt_svm.sample_count) )

        m, n = np.shape(platt_svm.feature_mat)
        error_count = 0
        for i in range(m):
            kernal_eval = kernal_trans(support_featues_mat, platt_svm.feature_mat[i], kTup)
            temp: np.matrix = np.multiply(support_alphas, support_label_mat)
            predict = temp.T * kernal_eval + platt_svm.b
            if (np.sign(predict) != np.sign(train_label[i])):
                error_count += 1
        print("training error rate is %f" % (float(error_count) / m))

        test_name_list = [x for x in name_list if (x.startswith("testDigits") and x.endswith(".txt"))]
        test_features, test_digit = hand_writing_data_set(zip_file, test_name_list)
        test_labels = digital_to_label(test_digit,9)

        test_feature_mat = np.mat(test_features)
        test_labels_mat = np.mat(test_labels)
        error_count = 0
        m, n = np.shape(test_feature_mat)
        for i in range(m):
            kernal_eval = kernal_trans(support_featues_mat, test_feature_mat[i], kTup)
            temp: np.matrix = np.multiply(support_alphas, support_label_mat)
            predict = temp.T * kernal_eval + platt_svm.b
            if (np.sign(predict) != np.sign(test_labels[i])):

                #print("%f %f %f "% (np.sum(support_featues_mat), np.sum(test_feature_mat[i]), np.sum(kernal_eval[i])) )
                #print("error for %f %d" % (predict, test_digit[i] ) )

                error_count += 1
            else:
                #print("right for %d" % test_digit[i])
                pass

        print("test error rate is %f" % (float(error_count) / m))

if __name__ == "__main__":
    #test_linear();
    #test_none_linear()
    hand_writing_class_test()