import numpy as np;
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import zipfile


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
    #print classify_count
    sorted_classify_count = sorted(classify_count.iteritems(),key=sort_key,reverse=True)
    #print sorted_classify_count
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


def auto_norm(data_set):  # type: (np.ndarray) -> np.ndarray
    min_vals = data_set.min(0) # type: np.ndarray
    max_vals = data_set.max(0) # type: np.ndarray
    ranges = max_vals - min_vals
    row_count = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals,(row_count,1))
    norm_data_set = norm_data_set / np.tile(ranges,(row_count,1))
    return  norm_data_set,ranges,min_vals


def dating_class_test():
    test_radio = 0.1
    abs_cur_dir = os.path.abspath(os.curdir)
    dating_data_mat, dating_labels = file2matrix(abs_cur_dir + "/data/ch02/datingTestSet2.txt")
    norm_data_set,_,_ = auto_norm(dating_data_mat)
    count_for_test = (int)(norm_data_set.shape[0]*test_radio)

    num_error=0
    for i in range(count_for_test):
        classify_ret = classify0(norm_data_set[i],norm_data_set[count_for_test:],dating_labels[count_for_test:],3)
        if classify_ret != dating_labels[i]:
            num_error+=1

    print "total error count %d, rate is %f" % (num_error, (float)(num_error)/(float)(count_for_test))


def img_to_vector(zip_file,file_name): #type: (zipfile.ZipFile,str)->np.ndarray
    vec = np.zeros((1,1024))
    with zip_file.open(file_name) as zip_item_file:
        line_count = 0
        for line in zip_item_file:
            for i in range(32):
                vec[0,line_count*32+i]=int(line[i])
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
        lable_num = label_name.split("_")[0]
        lable_num = int(lable_num)
        data_label[i] = lable_num
    return data_set,data_label


def hand_writing_class_test():
    abs_cur_dir = os.path.abspath(os.curdir)
    zip_file_name = os.path.join(abs_cur_dir,"data/ch02/digits.zip")
    with zipfile.ZipFile(zip_file_name,'r') as zip_file:
        name_list = zip_file.namelist()

        train_name_list = [x for x in name_list if x.startswith("trainingDigits") and x.endswith(".txt")]
        data_set, data_label = hand_writing_data_set(zip_file, train_name_list)

        num_error = 0
        test_name_list = [x for x in name_list if (x.startswith("testDigits") and x.endswith(".txt"))]
        for test_item_name in test_name_list:
            test_vec = img_to_vector(zip_file,test_item_name)
            test_label = test_item_name.replace("testDigits/","").replace(".txt","")
            test_label = int(test_label.split("_")[0])
            class_label = classify0(test_vec,data_set,data_label,3)
            if test_label!=class_label:
                print "%d error to %d" % (test_label,class_label )
                num_error+=1
            else:
                print "right get %d" % test_label

        print "total error count %d, rate is %f" % (num_error, (float)(num_error) / (float)(len(test_name_list)))

if __name__ == '__main__':
    hand_writing_class_test()
    #group,labels = create_dataset()
    #print classify0([0,0],group,labels,3)

    #fg = plt.figure()
    #ax = fg.add_subplot(111)
    #count = dating_data_mat.shape[0]
    #ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15*np.array(dating_labels),np.array(dating_labels))
    #plt.show()



