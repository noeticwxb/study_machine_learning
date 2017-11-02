# -*- coding:utf-8 -*-
import numpy as np
import io
import re

def load_dataset():#type:()->(list,list)
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['may', 'be', 'not', 'taken', 'them', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting','stupid','worthless', 'garbage'],
                    ['mr', 'licks','ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                    ]
    class_vec = [0,1,0,1,0,1]
    return posting_list,class_vec


def create_vocab_list(dataset):
    vocab_set = set()
    for doc in dataset:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)


def set_of_word_to_vec(vocab_list,word_list): #type:(list,list)->list
    ret_vec = [0]*len(vocab_list)
    for word in word_list:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)]=1
        else:
            print "the word %s is not in vocabulary" % word
    return ret_vec


def train_NB0(train_matrix,train_category):
    p_abusive = np.sum(train_category) / float(len(train_category))

    num_train_doc = len(train_matrix)
    num_words_one_doc = len(train_matrix[0])
    p0_accumulate = np.zeros(num_words_one_doc)
    p1_accumulate = np.zeros(num_words_one_doc)
    for i in range(0,num_train_doc):
        if train_category[i] == 0:
            p0_accumulate += train_matrix[i]
        else:
            p1_accumulate += train_matrix[i]

    p0_vect = p0_accumulate / float(sum(p0_accumulate))
    p1_vect = p1_accumulate / float(sum(p1_accumulate))

    return p0_vect,p1_vect,p_abusive

def text_parse(text):
    tokens = re.split(r"\W*",text)
    return  [token.lower() for token in tokens if len(token) > 2 ]

def spam_test():
    for i in range(1,26):
        email_text = io.open("data/ch04/email/spam/%d.txt" % i).read()





def test():
    data_set,classify_labels = load_dataset()
    vocab_list = create_vocab_list(data_set)
    train_matrix = []
    for doc in data_set:
        train_vec = set_of_word_to_vec(vocab_list,doc)
        train_matrix.append(train_vec)
    p1_vec,p2_vec,p_abusive = train_NB0(train_matrix,classify_labels)
    print vocab_list
    print p1_vec
    print p2_vec

if __name__ == '__main__':
    print text_parse("355 eeee i love her")