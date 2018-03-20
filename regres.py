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



if __name__ == '__main__':
    test_stand_regres()
    #m = np.asmatrix([[4,3],[2,10],[5,6]])
    #print(np.shape(m))
    #m.sort(1)
    #print(m)



