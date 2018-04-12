import numpy as np

def load_dataset(filename)->np.ndarray:
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_split = line.strip().split('\t')
        float_array = map(float,line_split)
        data_mat.append(float_array)
    return np.asarray(data_mat)

def bin_split_dataset(dataset:np.ndarray,feature,value)->(np.ndarray,np.ndarray):
    sample_great  = dataset[:,feature] > value
    sample_lq = dataset[:,feature] <= value
    sample_greate_index = np.nonzero(sample_great)[0]
    sample_lq_index = np.nonzero(sample_lq)[0]
    dataset_greate = dataset[sample_greate_index,:]
    dataset_lq = dataset[sample_lq_index,:]
    return dataset_greate,dataset_lq

def create_tree(dataset:np.ndarray,leaf_type,errType,ops=(1,4)):
    pass




if __name__ == "__main__":
    test_array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    great = test_array[:,1]<=5
    great_index = np.nonzero(great)[0]
    a = test_array[great_index]


