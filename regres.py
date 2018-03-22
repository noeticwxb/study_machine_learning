import numpy as np
import io
import matplotlib.pyplot as plt

class Regres:
    def load_dataset(self,filename)->(np.matrix,np.matrix):
        fr = io.open(filename)
        features = []
        lables = []
        for line in fr.readlines():
            line_split = line.strip().split('\t')
            line_attr = []
            for i_attr in range(len(line_split)-1):
                line_attr.append(float(line_split[i_attr]))
            features.append(line_attr)
            lables.append(float(line_split[-1]))
        return np.asmatrix(features),np.asmatrix(lables).reshape(-1,1)

    def stand_regres(self,features:np.matrix,lables:np.matrix):
        xTx = features.T * features
        if np.linalg.det(xTx) == 0.0:
            print("the matrix is singular,can not be inverse")
            return
        ws = xTx.I * (features.T * lables)
        return  ws

    def local_weight_Linear_regres(self,test_point,features:np.matrix,lables:np.matrix,k=1.0):
        sample_count,feature_count = np.shape(features)
        local_weight = np.asmatrix(np.eye(sample_count))
        k_square = k ** 2
        for j in range(sample_count):
            diff_mat = test_point - features[j,:]
            local_weight[j,j] = np.exp((diff_mat*diff_mat.T)/(-2.0*k_square))
        xTx = features.T * (local_weight * features)
        if np.linalg.det(xTx) == 0.0:
            print("the matrix is singular,can not be inverse")
            return
        ws = xTx.I * (features.T *(local_weight* lables))
        return test_point*ws


def test_stand_regres():
    regres = Regres()
    features, lables = regres.load_dataset("data/ch08/ex0.txt")
    weights = regres.stand_regres(features, lables)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(features[:,1].flatten().getA()[0],lables[:,0].flatten().getA()[0])

    features_sort:np.matrix = features.copy()
    features_sort.sort(0)
    lables_predit = features_sort * weights
    ax.plot(features_sort[:,1].flatten().getA()[0],lables_predit[:,0].flatten().getA()[0])

    plt.show()


def test_lwl_regres(k=1.0):
    regres = Regres()
    features, lables = regres.load_dataset("data/ch08/ex0.txt")
    sample_count,feature_count = np.shape(features)
    labels_predit = np.zeros(sample_count)
    for i in range(sample_count):
        labels_predit[i] = regres.local_weight_Linear_regres(features[i],features,lables,k)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(features[:, 1].flatten().getA()[0], lables[:, 0].flatten().getA()[0])

    sordInd = features[:,1].argsort(0)
    features_sorted = features[sordInd][:,0,:]
    #print(np.shape(features_sorted))
    #print(features_sorted)
    ax.plot(features_sorted[:, 1].flatten().getA()[0], labels_predit[sordInd])

    fig.show()


if __name__ == '__main__':
    #test_stand_regres()
    #m = np.asmatrix([[4,3],[2,10],[5,6]])
    #print(np.shape(m))
    #m.sort(1)
    #print(m)
    test_lwl_regres(0.002)



