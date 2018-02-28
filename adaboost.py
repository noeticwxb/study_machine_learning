import numpy as np

class Stump:
    '''
        single layer decision-making tree
    '''
    def __init__(self,thresh_dim,thresh_op,thresh_val):
        self.set_param(thresh_dim,thresh_op,thresh_val)

    def set_param(self,thresh_dim,thresh_op,thresh_val):
        self.thresh_dim:int = thresh_dim
        self.thresh_val:int = thresh_val
        self.thresh_op:str = thresh_op

    def get_param(self):
        return self.thresh_dim,self.thresh_val,self.thresh_op

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
        self.sample_weight:np.ndarray = np.ones(np.shape(self.labels)) / 5

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
                    print("dim:%s thresh:%.2f op:%s weighted_error:%.3f" % (feature_index,thresh_val ,op,weighted_error ))
                    if weighted_error < min_error:
                        best_param = stump.get_param()
                        best_classify_ret = predict_ret.copy()
                        min_error = weighted_error

        return stump,best_classify_ret,min_error



def load_simple_date()->(np.ndarray,np.ndarray):
    feature_array = np.asarray([[1.0,2.1],
                         [2.0,1.1],
                         [1.3,1.0],
                         [1.0,1.0],
                         [2.0,1.0]
                         ])
    label_array = np.asarray([1.0,1.0,-1.0,-1.0,1.0]).reshape(1,-1).transpose()
    return feature_array,label_array

def test_func(i,j):
    return i,j

if __name__ == '__main__':
    feature_array, label_array = load_simple_date()
    ada = AdaBoost(feature_array,label_array)
    ada.build_stump()
    #k = np.asarray([1,2,3])
    #print(np.shape(k))
    #k[ k < 2.0 ] = 2.0
    #print(k)
