import numpy as np
import io

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


if __name__ == '__main__':
    regres = Regres()
    features,lables = regres.load_dataset("data/ch08/ex0.txt")
    weights = regres.stand_regres(features,lables)
    print(weights)



