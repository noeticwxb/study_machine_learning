# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

DecisionNode = {'boxstyle':'sawtooth','fc':'0.8'}
LeafNode = {'boxstyle':'round4','fc':'0.8'}
Arrow_Args = {'arrowstyle':'<-'}


def plot_node(node_text,center_pt,parent_pt,node_type):
    create_plot.ax1.annotate(node_text, xy=parent_pt ,xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction', va='center', ha='center', bbox=node_type, arrowprops=Arrow_Args)


def plot_mid_text(tex,center_pt,parent_pt):
    mid_x = (parent_pt[0]-center_pt[0]) / 2.0 + center_pt[0]
    mid_y = (parent_pt[1]-center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(mid_x,mid_y,tex)


def create_plot(my_tree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111,frameon=False)
    #plot_node('Decision Node', (0.5,0.1), (0.1,0.5), DecisionNode)
    #plot_node('Leaf Node', (0.8,0.1), (0.3,0.8), LeafNode)
    X_OFF = 1.0 / (get_num_leafs(my_tree)-1 )
    Y_OFF = 1.0 / get_num_depth(my_tree)
    plot_tree(my_tree, 0, 1, 1, (0.5, 1), X_OFF, Y_OFF)
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
            this_depth = get_num_depth(childs[child_name])+1
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


def plot_tree(my_tree, x_pos_begin, x_pos_end, y_pos, parent_pt,X_OFF,Y_OFF):#type:(dict,float,float,float,tuple,float,float)->void
    node_str = my_tree.keys()[0]
    node_pos = ((x_pos_begin+x_pos_end)/2.0,y_pos)
    plot_node(node_str,node_pos,parent_pt,DecisionNode)
    child_dict = my_tree[node_str] #type:dict
    x_child_pos = x_pos_begin
    y_child_pos = y_pos - Y_OFF
    for key,child_node in child_dict.iteritems():
        if( type(child_node).__name__=='dict'):
            x_child_range = X_OFF * (get_num_leafs(child_node)-1)
            x_child_end = x_child_pos + x_child_range
            plot_tree(child_node,
                      x_child_pos,
                      x_child_end,
                      y_child_pos,
                      node_pos,
                      X_OFF,Y_OFF)
            plot_mid_text(key, ((x_child_pos+x_child_end)/2.0,y_child_pos ),node_pos)
            x_child_pos = x_child_pos + x_child_range + X_OFF
        else:
            plot_node(child_node, (x_child_pos,y_child_pos), node_pos, LeafNode)
            plot_mid_text(key, (x_child_pos,y_child_pos), node_pos)
            x_child_pos = x_child_pos +  X_OFF

if __name__ == '__main__':
    create_plot(retrieve_tree(1))
    #print get_num_leafs(retrieve_tree(1))
    #print get_num_depth(retrieve_tree(1))