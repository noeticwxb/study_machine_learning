# -*- coding:utf-8 -*-
import numpy as np
import io
import re
import random
import math
import feedparser
import operator

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
        #else:
        #    print "the word %s is not in vocabulary" % word
    return ret_vec

def bag_of_word_to_vec_nm(vocab_list,word_list):
    return_vec = [0] * len(vocab_list)
    for word in word_list:
        if word in vocab_list:
            return_vec[ vocab_list.index(word) ] += 1
    return return_vec

def train_NB0(train_matrix,train_category):
    p_abusive = np.sum(train_category) / float(len(train_category))

    num_train_doc = len(train_matrix)
    num_words_one_doc = len(train_matrix[0])
    p0_accumulate = np.ones(num_words_one_doc)
    p1_accumulate = np.ones(num_words_one_doc)
    p0_sum = 2
    p1_sum = 2
    for i in range(0,num_train_doc):
        if train_category[i] == 0:
            p0_accumulate += train_matrix[i]
            p0_sum += sum(train_matrix[i])
        else:
            p1_accumulate += train_matrix[i]
            p1_sum += sum(train_matrix[i])

    #print sum(p0_accumulate)
    p0_vect = np.log(p0_accumulate / p0_sum)
    #print sum(p1_accumulate)
    p1_vect = np.log(p1_accumulate / p1_sum)

    return p0_vect,p1_vect,p_abusive

def classify_NB0(vec_to_classify,p0_vect,p1_vect,p_abusive):
    p0 = sum(vec_to_classify*p0_vect) + np.log(p_abusive)
    p1 = sum(vec_to_classify*p1_vect) + np.log(p_abusive)
    if(p1 > p0):
        return 1
    else:
        return 0

def text_parse(text):
    tokens = re.split(r"\W*",text)
    return  [token.lower() for token in tokens if len(token) > 2 ]

def spam_test():
    mail_list = []
    class_list = []
    #full_word_list = []
    for i in range(1,26):
        # spam
        file_name = "data/ch04/email/spam/%d.txt" % i
        #print file_name
        email_words = io.open(file_name,encoding='UTF-8').read()
        mail_list.append(email_words)
        class_list.append(1)
        #full_word_list.extend(email_words)
        #
        file_name = "data/ch04/email/ham/%d.txt" % i
        #print file_name
        email_words = io.open(file_name,encoding='UTF-8').read()
        mail_list.append(email_words)
        class_list.append(0)
        #full_word_list.extend(email_words)

    train_set = range(50)
    test_set = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(train_set)))
        test_set.append(train_set[randIndex])
        del(train_set[randIndex])

    vocab_list = create_vocab_list(mail_list)

    train_mat=[]
    train_class =[]
    for doc_index in train_set:
        train_vec = set_of_word_to_vec(vocab_list,mail_list[doc_index])
        train_mat.append( train_vec)
        train_class.append( class_list[doc_index] )

    p0_vect, p1_vect, p_abusive = train_NB0(train_mat,train_class)

    error_count = 0
    for doc_index in test_set:
        test_vec = set_of_word_to_vec(vocab_list,mail_list[doc_index])
        classfy__ret = classify_NB0(test_vec,p0_vect,p1_vect,p_abusive)
        if classfy__ret != class_list[doc_index]:
            error_count += 1

    print "erro rate is %f " % (float(error_count)/len(test_set))

# 读取两个RSS源，使用贝叶斯方法进行分类。看分类是否正确
def rss_test_start():
    print "read rss and parse"
    rss_url_1 = "https://newyork.craigslist.org/search/stp?format=rss"
    rss_url_2 = "https://sfbay.craigslist.org/search/stp?format=rss"
    ny_feed = feedparser.parse(rss_url_1)
    sf_feed = feedparser.parse(rss_url_2)
    print "begin classfy"
    rss_test(ny_feed,sf_feed)

def cal_most_frequency(vocab_list,full_words):
    freq_dict = {}
    for vocab in vocab_list:
        freq_dict[vocab] = full_words.count(vocab)
    sorted_freq_dict = sorted(freq_dict.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sorted_freq_dict[:30]

def rss_test(feed0, feed1):
    ENTRIES = 'entries'
    SUMMARY = 'summary'

    # read rss and construct dataset
    doc_list = []
    class_list = []
    full_words = []
    min_len = min(len(feed0[ENTRIES]),len(feed1[ENTRIES]))

    for idoc in range(min_len):
        doc = text_parse(feed0[ENTRIES][idoc][SUMMARY])
        doc_list.append(doc)
        class_list.append(1)
        full_words.extend(doc)
        doc = text_parse(feed1[ENTRIES][idoc][SUMMARY])
        doc_list.append(doc)
        class_list.append(0)
        full_words.extend(doc)

    # construct vocab_list, and remove frequence top 30
    vocab_list = create_vocab_list(doc_list)
    freq_dict = cal_most_frequency(vocab_list,full_words)
    for wordPair in freq_dict:
        if wordPair[0] in vocab_list:
            vocab_list.remove(wordPair[0])

    print_vocab_list_in_order2(vocab_list,full_words)

    # construct train_set and test_set
    train_set = range( len(doc_list) )
    test_set = []
    #test_set_size = len(doc_list) / 10
    test_set_size = 5
    for i in range( test_set_size ):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append( train_set[rand_index])
        del(train_set[rand_index] )

    train_matrix = []
    train_category = []
    for iDoc in train_set:
        vec_to_train = bag_of_word_to_vec_nm(vocab_list,doc_list[iDoc])
        train_matrix.append(vec_to_train)
        train_category.append(class_list[iDoc])

    p0_vect, p1_vect, p_abusive = train_NB0(train_matrix,train_category)

    #print_vocab_list_in_order(vocab_list,p0_vect)
    #print_vocab_list_in_order(vocab_list, p1_vect)

    error_count = 0
    for iDoc in test_set:
        vec_to_classify =bag_of_word_to_vec_nm(vocab_list,doc_list[iDoc])
        ret = classify_NB0(vec_to_classify, p0_vect, p1_vect, p_abusive)
        if ret != class_list[iDoc]:
            error_count += 1

    print "Error count is %d rate is %f" % (error_count, float(error_count) / len(test_set) )
    return vocab_list,p0_vect,p1_vect

def test():
    data_set,classify_labels = load_dataset()
    vocab_list = create_vocab_list(data_set)
    train_matrix = []
    for doc in data_set:
        train_vec = set_of_word_to_vec(vocab_list,doc)
        train_matrix.append(train_vec)
    p1_vec,p2_vec,p_abusive = train_NB0(train_matrix,classify_labels)
    #print vocab_list
    #print p1_vec
    #print p2_vec

def print_vocab_list_in_order2(vocab_list, full_words):
    dict = {}
    for i in range( len(vocab_list) ):
        dict[vocab_list[i]] = full_words.count(vocab_list[i])
    dict_sorted = sorted(dict.iteritems(),key=operator.itemgetter(1),reverse=False)
    print "sorted"
    print dict_sorted[:10]

def print_vocab_list_in_order(vocab_list,p_vect):
    dict = {}
    for i in range( len(vocab_list) ):
        dict[vocab_list[i]] = p_vect[i]
    dict_sorted = sorted(dict.iteritems(),key=operator.itemgetter(1),reverse=True)
    #print dict_sorted[:20]
    print "sorted"
    print dict_sorted[:10]

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=rss_test(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]

if __name__ == '__main__':
    ny = feedparser.parse("https://newyork.craigslist.org/search/stp?format=rss")
    sf = feedparser.parse("https://sfbay.craigslist.org/search/stp?format=rss")
    rss_test(ny, sf)

