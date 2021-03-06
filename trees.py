# -*- coding:utf-8 -*-
import numpy as np
import operator
import json
import os
import tree_plotter

def create_test_dataset():
    dataset = [[1,1,'yes'],
                [1,1,'yes'],
                [1,0,'no'],
                [0,1,'no'],
                [0,1,'no']];
    feature_names = ['no surfacing','flippers']
    return dataset,feature_names


def calc_shannon_ent(dataset):
    label_count_map = {}
    for data_vec in dataset:
        label_count = label_count_map.get(data_vec[-1],0)
        label_count_map[data_vec[-1]] = label_count + 1

    num_entries = len(dataset)
    shannon_ent = 0
    for key in label_count_map:
        prop = float(label_count_map[key]) / num_entries
        shannon_ent -= prop * np.log2(prop)

    return shannon_ent


def split_dataset(dataset,feature_index,feature_value):
    ret_data_set = []

    for data_vec in dataset:
        if(data_vec[feature_index]==feature_value):
            ret_vec = data_vec[0:feature_index] #type: list
            ret_vec.extend(data_vec[feature_index+1:])
            ret_data_set.append(ret_vec)

    return ret_data_set


def choose_best_feature_to_split(dataset):
    # 这是决策树算法的关键，选择信息增益最大的
    shannon_ent = calc_shannon_ent(dataset)
    num_feature = len(dataset[0]) - 1
    best_info_gain = -1
    best_feature_index = -1

    for feature_index in range(0,num_feature):
        feature_values = [row_vec[feature_index] for row_vec in dataset]
        feature_values_set = set(feature_values)
        shannon_ent_split = 0
        for feature_value in feature_values_set:
            dataset_split = split_dataset(dataset,feature_index,feature_value)
            dataset_split_prop = float(len(dataset_split)) / float(len(dataset))
            shannon_ent_split += ( dataset_split_prop * calc_shannon_ent(dataset_split) )
        info_gain = shannon_ent - shannon_ent_split
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = feature_index

    return best_feature_index


def majority_cnt(class_list):
    class_count = {}
    for classify in class_list:
        cnt = class_count.get(classify[-1],0)
        class_count[classify[-1]] = cnt + 1
    #根据dic的value进行排序,生成list
    sorted_class_count = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset,feature_names):
    class_list = [row_vec[-1] for row_vec in dataset]
    #数据集合里面的所有分类都是一样
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    #数据集已经没有特征，按照最大权重进行投票
    if len(dataset[0])==1:
        return majority_cnt()

    best_split_feature_index = choose_best_feature_to_split(dataset)
    best_split_feature_name = feature_names[best_split_feature_index]
    tree = {best_split_feature_name:{}}
    feature_values_set = set( [row_vec[best_split_feature_index] for row_vec in dataset] )
    for feature_value in feature_values_set:
        sub_dataset = split_dataset(dataset,best_split_feature_index,feature_value)
        sub_feature_names = feature_names[0:best_split_feature_index] #type:list
        sub_feature_names.extend( feature_names[best_split_feature_index+1:])
        tree[best_split_feature_name][feature_value] = create_tree(sub_dataset,sub_feature_names)

    return  tree


def classify(my_tree,feature_lables,to_classify_vec): #type:(dict,list,list)->str

    #first_str = my_tree.keys()[0]
    first_str = next(iter(my_tree))
    search_dict = my_tree[first_str] #type:dict
    feature_index = feature_lables.index(first_str)
    classify_label = ''
    for key,node in search_dict.items():
        if to_classify_vec[feature_index] == key:
            if type(node).__name__=='dict':
                classify_label = classify(node,feature_lables,to_classify_vec)
            else:
                classify_label = node
    return classify_label


def load_tree(filename):
    with open(filename,'r') as file_obj:
        load_dict = json.load(file_obj,ensure_ascii=True)
        print(load_dict)
        return load_dict


def write_tree(my_tree,filename):
    with open(filename,'w') as file_obj:
        json.dump(my_tree,file_obj)


def read_data_set():
    abs_cur_dir = os.path.abspath(os.curdir)
    file_name = os.path.join(abs_cur_dir,"data/ch03/lenses.txt")
    data_set = []
    with open(file_name,'r') as file_obj:
        for line in file_obj:
            data_vec = line.strip().split('\t')
            data_set.append(data_vec)
    return data_set


if __name__ == '__main__':
    #dataset,feature_names = create_test_dataset()
    #print calc_shannon_ent(dataset)
    #print choose_best_feature_to_split(dataset)
    #tree = create_tree(dataset,feature_names)
    #write_tree(tree,'d:\\tree.json')
    #load_tree('d:\\tree.json')
    #print tree
    data_set = read_data_set()
    data_lables = ['age','prescript','astigmatic','tearrate']
    my_tree = create_tree(data_set,data_lables)
    #tree_plotter.create_plot(my_tree)
    error_count = 0
    for data_vec in data_set:
        ret = classify(my_tree,data_lables,data_vec)
        ret_right = data_vec[-1]
        if(ret != ret_right):
            error_count += 1
    error_percentage = float(error_count) / float(len(data_set));
    print(error_percentage)