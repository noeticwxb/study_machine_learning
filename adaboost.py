import numpy as np
import math as math
import io as io

class Stump:
    '''
        single layer decision-making tree
    '''
    def __init__(self,thresh_dim,thresh_op,thresh_val):
        self.set_param(thresh_dim,thresh_op,thresh_val)
        self.alpha = 1.0

    def set_param(self,thresh_dim,thresh_op,thresh_val):
        self.thresh_dim = thresh_dim
        self.thresh_val = thresh_val
        self.thresh_op = thresh_op

    def set_alpha(self,alpha):
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha

    def get_param(self):
        return self.thresh_dim,self.thresh_op,self.thresh_val

    def classify(self,samples)->np.ndarray:
        rets = np.ones((np.shape(samples)[0],1))
        if self.thresh_op == "lt":
            rets[ samples[:,self.thresh_dim] <= self.thresh_val ] = -1.0
        else:
            rets[ samples[:, self.thresh_dim] > self.thresh_val ] = -1.0
        return rets


class AdaBoost:
    def __init__(self,features,labels):
        self.samples:np.ndarray = features
        self.labels:np.ndarray = labels
        self.sample_weight:np.ndarray = np.ones(np.shape(self.labels)) / np.shape(self.labels)[0]
        self.stumps:list[Stump] = []

    def build_stump(self):
        sample_count,feature_count = np.shape(self.samples)
        best_param = [0,"lt",0]
        best_classify_ret = []
        min_error = float("inf")
        num_steps = 10
        stump = Stump(*best_param)
        for feature_index in range(feature_count):
            max_feature_val = self.samples[:, feature_index].max()
            min_feature_val = self.samples[:, feature_index].min()
            step_size = (max_feature_val-min_feature_val) / num_steps
            for loop_step in range(-1,num_steps):
                for op in ["lt","gt"]:
                    thresh_val = min_feature_val+float(loop_step)*step_size
                    stump.set_param(feature_index,op,thresh_val)
                    predict_ret = stump.classify(self.samples)
                    error_ret = np.ones((sample_count,1))
                    test= predict_ret==self.labels
                    error_ret[test] = 0
                    weighted_error = np.sum(error_ret*self.sample_weight)
                    #print("dim:%s thresh:%.2f op:%s weighted_error:%.3f" % (feature_index,thresh_val ,op,weighted_error ))
                    if weighted_error < min_error:
                        best_param = stump.get_param()
                        best_classify_ret = predict_ret.copy()
                        min_error = weighted_error


        alpha = float(0.5 * np.log((1.0 - min_error) / max(min_error, 1e-16)))
        stump.set_alpha(alpha)
        stump.set_param(*best_param)
        return stump,best_classify_ret,min_error

    def training(self,max_iter=40,tolerant = 0):
        self.stumps = []
        sample_count, feature_count = np.shape(self.samples)
        aggregate_classEst = np.zeros((sample_count,1))

        for iter in range(max_iter):
            best_stump,best_classify_ret,min_error = self.build_stump()
            alpha = best_stump.get_alpha()
            self.stumps.append(best_stump)

            expon = -1 * alpha * (best_classify_ret * self.labels)
            self.sample_weight = self.sample_weight * np.exp(expon)
            self.sample_weight = self.sample_weight / np.sum(self.sample_weight)

            aggregate_classEst += alpha * best_classify_ret
            error_class = (np.sign(aggregate_classEst) != self.labels)
            aggregate_error = error_class * np.ones((sample_count,1))
            error_rate = aggregate_error.sum() / sample_count
            if error_rate < tolerant:
                print("break alpha=%s %s"%(alpha,error_rate))
                break
            else:
                #print("alpha=%s %s %s" % (alpha,best_classify_ret, aggregate_classEst))
                print("alpha=%s %s" % (alpha, error_rate))
                pass

    def classify(self,data_set):
        samples = np.asarray(data_set)
        sample_count = np.shape(data_set)[0]
        agg_classEst = np.zeros((sample_count,1))
        for stump in self.stumps:
            classEst = stump.classify(samples)
            agg_classEst += (classEst * stump.get_alpha())
            #print("alpha=%s %s %s "%(stump.get_alpha(), classEst,agg_classEst))
        return np.sign(agg_classEst),agg_classEst


def load_simple_date()->(np.ndarray,np.ndarray):
    feature_array = np.asarray([[1.0,2.1],
                         [2.0,1.1],
                         [1.3,1.0],
                         [1.0,1.0],
                         [2.0,1.0]
                         ])
    label_array = np.asarray([1.0,1.0,-1.0,-1.0,1.0]).reshape(1,-1).transpose()
    return feature_array,label_array

def load_Dataset(filename)->(np.ndarray,np.ndarray):
    samples = []
    labels = []
    fr = io.open(filename)
    for line in fr.readlines():
        line_splits = line.strip().split('\t')
        num_split = len(line_splits)
        features = []
        for i in range(num_split-1):
            features.append(float(line_splits[i]))
        if float(line_splits[-1]) > 0:
            labels.append(1)
        else:
            labels.append(-1)

        samples.append(features)
    return np.asarray(samples),np.asarray(labels).reshape(1,-1).transpose()


def test_ada():
    traning_sample,traning_labels = load_Dataset("data/ch05/horseColicTraining.txt")
    test_samples,test_labels = load_Dataset("data/ch05/horseColicTest.txt")
    ada = AdaBoost(traning_sample, traning_labels)
    ada.training(500)
    ret_lables,ret_strength = ada.classify(test_samples)
    plot_ROC(ret_strength,test_labels)
    num_ret = np.shape(ret_lables)[0]
    error_sum:np.ndarray = np.ones((num_ret,1))
    error_sum = error_sum[ret_lables!=test_labels]
    print("error num %d rate %f" % (error_sum.sum(),float(error_sum.sum())/float(num_ret)))

def  plot_ROC(predict_strength:np.ndarray,lables:np.ndarray):
    import matplotlib.pyplot as plt

    positive_lable_count = np.sum(lables == 1.0)
    negative_label_count = np.shape(lables)[0] - positive_lable_count
    #print("postive= %d negtive= %d" %(positive_lable_count,negative_label_count)  )
    y_step = 1 / float(positive_lable_count)
    x_step = 1 / float(negative_label_count)

    sorted_index:np.ndarray = predict_strength.argsort(axis=0)
    flg: plt.Figure = plt.figure()
    flg.clf()
    ax:plt.Axes = plt.subplot(111)

    y_sum = 0
    cur = (1.0, 1.0)
    l = sorted_index.tolist()
    for index in sorted_index.tolist():
        if lables[index[0]] == 1.0:
            delta_x = 0;delta_y = y_step
        else:
            delta_x = x_step;delta_y = 0
            y_sum += cur[1]
        point_x = [cur[0],cur[0]-delta_x]
        point_y = [cur[1],cur[1]-delta_y]
        ax.plot(point_x,point_y,c='b')
        cur = (cur[0]-delta_x,cur[1]-delta_y)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True postive rate')
    plt.title('Roc Curve')
    ax.axis([-0.1,1.5,-0.1,1.5])
    plt.show()
    print("the area under the curve isL ", y_sum * x_step )

if __name__ == '__main__':
    if True:
        test_ada()
    else:
        feature_array, label_array = load_simple_date()
        ada = AdaBoost(feature_array,label_array)
        ada.training(1)
        ret,ret_strength = ada.classify(feature_array)
        plot_ROC(ret_strength,label_array)
    #ret = ada.classify([[5.0,5.0],[0.0,0.0]])
    #print(ret)
    #print(ada.sample_weight)
    #ada.build_stump()
    #k = np.asarray([1,2,3])
    #print(np.shape(k))
    #g = [ k < 2.0 ]
    #print(k[g])