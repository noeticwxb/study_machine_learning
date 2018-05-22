import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def load_dataset(filename)->np.ndarray:
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_split = line.strip().split('\t')
        float_array_map = map(float,line_split)
        data_mat.append(list(float_array_map))
    return np.asarray(data_mat)


def bin_split_dataset(dataset:np.ndarray,feature,value)->(np.ndarray,np.ndarray):
    sample_great  = dataset[:,feature] > value
    sample_lq = dataset[:,feature] <= value
    sample_greate_index = np.nonzero(sample_great)[0]
    sample_lq_index = np.nonzero(sample_lq)[0]
    dataset_greate = dataset[sample_greate_index,:]
    dataset_lq = dataset[sample_lq_index,:]
    return dataset_greate,dataset_lq


def reg_leaf(dataset):
    return np.mean(dataset[:,-1])


def reg_error(dataset):
    sample_count = np.shape(dataset)[0]
    return np.var(dataset[:,-1]) * sample_count


def create_tree(dataset:np.ndarray,leaf_func = reg_leaf,error_func = reg_error ,ops=(1,4)):
    feat_index,feat_val = choose_best_split(dataset,leaf_func,error_func,ops)
    if feat_index == None:
        return feat_val
    ret_tree = {}
    ret_tree['spInd'] = feat_index
    ret_tree['spVal'] = feat_val
    left_set,right_set = bin_split_dataset(dataset,feat_index,feat_val)
    ret_tree['left'] = create_tree(left_set,leaf_func,error_func,ops)
    ret_tree['right'] = create_tree(right_set,leaf_func,error_func,ops)
    return ret_tree


def choose_best_split(dataset,leaf_func=reg_leaf,error_func=reg_error,ops=(1,4)):
    #允许的误差下降值
    Error_Descent_Limit = ops[0]
    #切分的最少样本数
    Split_Sample_Limit = ops[1]

    # label只有一个，说明没有误差了
    not_repeate_label = set(dataset[:,-1].tolist())
    if len(not_repeate_label) == 1:
        return None,leaf_func(dataset)

    # 依次使用所有属性的所有值进行切分，找到切分后误差最小的。
    sample_count,column_count = np.shape(dataset)
    feature_count = column_count-1
    best_error = np.inf
    best_index = 0
    best_value = 0
    for feature_index in range(feature_count):
        for split_value in set(dataset[:,feature_index]):
            dataset0,dataset1 = bin_split_dataset(dataset,feature_index,split_value)
            if np.shape(dataset0)[0] < Split_Sample_Limit or np.shape(dataset1)[0] < Split_Sample_Limit:
                continue
            new_error = error_func(dataset0) + error_func(dataset1)
            if new_error < best_error:
                best_error = new_error
                best_index = feature_index
                best_value = split_value

    #最佳的切分方式的误差下降还是太少，所以直接生成叶子节点
    error_not_split = error_func(dataset)
    if (error_not_split - best_error) < Error_Descent_Limit:
        return None,leaf_func(dataset)

    #切分出的数据集很小也退出
    dataset0, dataset1 = bin_split_dataset(dataset, best_index, best_value)
    if np.shape(dataset0)[0] < Split_Sample_Limit or np.shape(dataset1)[0] < Split_Sample_Limit:
        return None, leaf_func(dataset)

    return best_index,best_value


def is_tree(obj):
    return type(obj).__name__ == "dict"


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right']+tree['left']) / 2.0

# 算法输出的结果和书上不一样，剪枝的效果好像不太好。 我把自带的代码编译跑过也是一样的，说明我没有写错，但是为毛剪枝效果不太好呢？
def prune(tree,test_data):
    # 对树进行塌陷
    m,n = np.shape(test_data)
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)

    if is_tree(tree['left']) or is_tree(tree['right']):
        l_set,r_set = bin_split_dataset(test_data,tree['spInd'],tree['spVal'])

    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'],l_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'],r_set)

    # 剪枝之后还是至少有一颗子树还是树，说明不再需要剪枝了，直接返回
    # 否则，判断下当前节点是否可以剪枝
    if is_tree(tree['left']) or is_tree(tree['right']):
        return tree
    else:
        # 我觉得这句话是多余的。l_set, r_set应该没有变化
        l_set, r_set = bin_split_dataset(test_data, tree['spInd'], tree['spVal'])
        error_before_merge = sum( np.power(l_set[:,-1] - tree['left'],2)) + sum( np.power(r_set[:,-1]-tree['right'],2) )
        merged_mean = ( (tree['right'] + tree['left']) / 2.0 )
        error_after_merge = sum( np.power(test_data[:,-1] - merged_mean ,2))
        if error_after_merge <= error_before_merge:
            print("merge")
            return merged_mean
        else:
            return tree


def liner_model_solve(dataset):
    m,n = np.shape(dataset)
    x = np.mat( np.ones((m,n)) )
    y = np.mat( np.ones((m,1)) )
    x[:,1:n] = dataset[:,0:n-1]
    y = np.mat(dataset[:,-1]).T
    xTx = x.T * x
    if np.linalg.det(xTx) == 0.0:
        raise ArithmeticError("matrix can not do inverse")
    t = (x.T * y)
    w = xTx.I * t
    return w,x,y


def liner_model_leaf(dataset):
    w,x,y = liner_model_solve(dataset)
    return w


def liner_model_error(dataset):
    w,x,y = liner_model_solve(dataset)
    y_hat = x * w
    return sum( np.power(y-y_hat,2) )


def reg_tree_eval(model,data):
    return float(model)


def liner_tree_eval(model,data):
    m,n = np.shape(data)
    x = np.mat(np.ones((1,n+1)))
    x[:,1:n+1] = data
    return float(x*model)


def tree_forecast(tree,in_data,model_eval = reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree,in_data)
    if in_data[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'],in_data,model_eval)
        else:
            return model_eval(tree['left'],in_data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'],in_data,model_eval)
        else:
            return model_eval(tree['right'],in_data)


def create_forcast(tree,test_data,model_eval=reg_tree_eval):
    m = len(test_data)
    y_hat = np.mat(np.zeros((m,1)))
    for i in range(m):
        y_hat[i,0]= tree_forecast(tree, np.mat(test_data[i]),model_eval)
    return y_hat


def test():
    training_set = load_dataset("data/ch09/bikeSpeedVsIq_train.txt")
    test_set = load_dataset("data/ch09/bikeSpeedVsIq_test.txt")
    my_reg_tree = create_tree(training_set,reg_leaf, reg_error, ops=(0,20))
    y_hat = create_forcast(my_reg_tree,test_set[:,0],reg_tree_eval)
    print( np.corrcoef(y_hat,test_set[:,1],rowvar=0)[0,1] )

    my_liner_tree = create_tree(training_set,liner_model_leaf,liner_model_error,ops=(1,20))
    y_hat = create_forcast(my_liner_tree,test_set[:,0],liner_tree_eval)
    print( np.corrcoef(y_hat,test_set[:,1],rowvar=0)[0,1] )


class ReDrawData:
    def __init__(self):
        self.raw_dat:np.matrix = None
        self.test_data: np.matrix = None
        self.f:Figure = None
        self.canvas:FigureCanvasTkAgg = None
        self.check_btn_var:tk.IntVar = None

def re_draw(tol_s,tol_n):
    data:ReDrawData = re_draw.data
    data.f.clf()
    data.a = data.f.add_subplot(111)
    if data.check_btn_var.get():
        if tol_n < 2:
            tol_n = 2
        my_liner_tree = create_tree(data.raw_dat, liner_model_leaf, liner_model_error, ops=(tol_s, tol_n))
        y_hat = create_forcast(my_liner_tree,data.test_data,liner_tree_eval)
    else:
        my_reg_tree = create_tree(data.raw_dat, reg_leaf, reg_error, ops=(tol_s, tol_n))
        y_hat = create_forcast(my_reg_tree, data.test_data, reg_tree_eval)

    data.a.scatter(data.raw_dat[:,0],data.raw_dat[:,1],5)
    data.a.plot(data.test_data,y_hat,linewidth=2.0)
    data.canvas.show()


def get_input(tol_s_entry:tk.Entry, tol_n_entry:tk.Entry):
    try:
        tol_s = float(tol_s_entry.get())
    except:
        tol_s = 0
        print("input Float for tol s ")
        tol_s_entry.delete(0,tk.END)
        tol_s_entry.insert(0,"1.0")

    try:
        tol_n= int(tol_s_entry.get())
    except:
        tol_n = 0
        print("input integer for tol n")
        tol_s_entry.delete(0,tk.END)
        tol_s_entry.insert(0,"10")

    return tol_s,tol_n




def draw_new_tree():
    tol_s,tol_n = get_input(draw_new_tree.tol_s_entry,draw_new_tree.tol_n_entry)
    re_draw(tol_s,tol_n)

def test_gui():
    # global root
    root = tk.Tk()
    # c = Canvas(root,width = 300, height = 200,bg = 'white')
    tk.Label(root,text="plot place holder").grid(row=0,columnspan = 3)

    tk.Label(root, text="tolN").grid(row=1,column=0)
    tol_N_entry = tk.Entry(root)
    tol_N_entry.grid(row=1,column=1)
    tol_N_entry.insert(0,"10")

    tk.Label(root, text="tolS").grid(row=2,column=0)
    tol_S_entry = tk.Entry(root)
    tol_S_entry.grid(row=2,column=1)
    tol_S_entry.insert(0,"1.0")

    draw_new_tree.tol_s_entry = tol_S_entry
    draw_new_tree.tol_n_entry = tol_N_entry

    tk.Button(root,text='Redraw',command=draw_new_tree).grid(row=1,column=2,rowspan=3)

    check_btn_var = tk.IntVar()
    check_btn = tk.Checkbutton(root,text='Model Tree',variable = check_btn_var)
    check_btn.grid(row=3,column=0,columnspan=2)

    re_draw_data = ReDrawData()

    re_draw_data.raw_dat = load_dataset("data/ch09/sine.txt")
    re_draw_data.test_data = np.arange(min(re_draw_data.raw_dat[:,0]),max( re_draw_data.raw_dat[:,0]),0.01 )

    re_draw_data.f = Figure(figsize=(5,4),dpi=100)
    re_draw_data.canvas = FigureCanvasTkAgg(re_draw_data.f,master=root)
    re_draw_data.canvas.show()
    re_draw_data.canvas.get_tk_widget().grid(row=0,columnspan=3)

    re_draw_data.check_btn_var = check_btn_var
    re_draw.data = re_draw_data
    re_draw(1.0,10)
    root.mainloop()


if __name__ == "__main__":
    test_gui()

    # training_set = load_dataset("data/ch09/ex2.txt")
    # test_set = load_dataset("data/ch09/ex2test.txt")
    # tree = create_tree(training_set,ops=(0,1))
    # print(tree)
    # tree = prune(tree,test_set)
    # print(tree)
    #tree = create_tree(data_set)
    #print(tree)

    # training_set = load_dataset("data/ch09/exp2.txt")
    # tree = create_tree(training_set,liner_model_leaf,liner_model_error,(1,10))
    # print(tree)


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(training_set[:,0],training_set[:,1])
    # fig.show()




