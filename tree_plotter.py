# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

DecisionNode = {'boxstyle':'sawtooth','fc':'0.8'}
LeafNode = {'boxstyle':'round4','fc':'0.8'}
Arrow_Args = {'arrowstyle':'<-'}


def plot_node(node_text,center_pt,parent_pt,node_type):
    create_plot.ax1.annotate(node_text, xy=parent_pt ,xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va='center', ha='center', bbox=node_type, arrowprops=Arrow_Args)


def create_plot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111,frameon=False)
    plot_node('Decision Node', (0.5,0.1), (0.1,0.5), DecisionNode)
    plot_node('Leaf Node', (0.8,0.1), (0.3,0.8), LeafNode)
    plt.show()


def get_num_leafs(my_tree): #type:(dict)->int
    num_leaf = 0
    root_name = my_tree.keys()[0]
    childs = my_tree[root_name] #type:dict
    for child_name in childs.keys():
        if type(childs[child_name]).__name__ == 'dict':
            num_leaf += get_num_leafs(childs[child_name])
        else:
            num_leaf += 1
    return num_leaf


def get_num_depth(my_tree): #type:(dict)->int
    max_depth = 0;
    root_name = my_tree.keys()[0]
    childs = my_tree[root_name] #type:dict
    for child_name in childs.keys():
        if type(childs[child_name]).__name__ == 'dict':
            this_depth = get_num_leafs(childs[child_name])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_trees = [
                {
                    'no surfacing':{
                        0:'no',
                        1:{
                            'flippers':{
                                0:'no',
                                1:'yes'
                            }
                        }
                    }
                },
                {
                    'no surfacing': {
                        0: 'no',
                        1: {
                            'flippers': {
                                0: {
                                    'head':{
                                        0: 'no',
                                        1: 'yes'
                                    }
                                },
                                1: 'no'
                            }
                        }
                    }
                }
                ]
    return list_trees[i]


if __name__ == '__main__':
    #create_plot()
    print get_num_leafs(retrieve_tree(1))
    print get_num_depth(retrieve_tree(1))