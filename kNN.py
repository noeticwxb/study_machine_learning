import numpy as np;
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def create_dataset():
    group = np.array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group,lables


def classify0(in_x, dataset, labels, k):
    #compute distance
    dataset_size = dataset.shape[0]
    in_x_mat = np.tile(in_x,(dataset_size,1))
    #print in_x_mat
    diff_mat = in_x_mat - dataset
    #print diff_mat
    # (every element in diff_mat) ** 2
    sq_diff_mat = diff_mat ** 2
    #print sq_diff_mat
    sq_distance = sq_diff_mat.sum(axis=1)
    #print sq_distance
    sorted_sq_dis_indices = sq_distance.argsort()

    #select points with mini distance
    #print sorted_sq_dis_indices
    classify_count = {}
    for i in range(k):
        vote_label = labels[sorted_sq_dis_indices[i]]
        classify_count[vote_label] = classify_count.get(vote_label,0) + 1

    sort_key = operator.itemgetter(1)
    print classify_count
    sorted_classify_count = sorted(classify_count.iteritems(),key=sort_key,reverse=True)
    print sorted_classify_count
    return sorted_classify_count[0][0]


def file2matrix(filename):
    #read from file
    fr = open(filename)
    array_lines = fr.readlines()
    number_lines = len(array_lines)

    #read and convert
    return_mat = np.zeros((number_lines,3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    return return_mat,class_label_vector


def auto_norm(data_set):# type: (np.ndarray) -> None
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)


#group,labels = create_dataset()
#print classify0([0,0],group,labels,3)
abs_cur_dir = os.path.abspath(os.curdir)

dating_data_mat,dating_labels = file2matrix( abs_cur_dir + "/data/ch02/datingTestSet2.txt")
#print dating_data_mat
#print dating_data_mat[:,1]

fg = plt.figure()
ax = fg.add_subplot(111)
count = dating_data_mat.shape[0]

ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15*np.array(dating_labels),np.array(dating_labels))
plt.show()



