import os
import gzip
import re
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import learn

def makeCSV(xml_path):

    # xml_path = os.path.join('download', 'drugbank_preview.xml.gz')
    with gzip.open(xml_path) as xml_file:
        source = BeautifulSoup(xml_file, "lxml")
    # print(source)
#     rawXML = source.findAll('drug type')
#     rawXML = source.findAll('drugbank', attrs={'xmlns':'http://www.drugbank.ca'})
#     drug_inCategory = source.findAll('drug', attrs={'type':'biotech'})  
    drug_inCategory = source.findAll('drug')
    print(drug_inCategory)
    
    texts = []
    target = []
    nameList = []
    
    n = 0
    for onedrug in drug_inCategory:
        n += 1
        print('작업수행중이다옹!!! ->', n)
        onedrug_Name = onedrug.find('name')
        onedrug_indic = onedrug.find('indication')
        onedrug_ATC = onedrug.find('atc-code')
        
        try:
            for i in onedrug_ATC:
                i = str(i)
                aa = re.findall('code="\w?"', i)
                if len(aa) > 0 :
                    target.append(aa[0][6])
        except Exception as e:
            print(e)
            continue       
    
        for name, indic in zip(onedrug_Name, onedrug_indic):
            nameList.append(name)
            texts.append(indic)
    
    print(texts)
    print(target)
    print('lentext : ',len(texts))
    print('lentarget : ',len(target))

    tmp = []
    for te, ta in zip(texts, target):
        te = str(te)
        ta = str(ta)
        tmp.append([te, ta])
    
    df_tmp = pd.DataFrame(tmp)
    df_tmp.to_csv('DrugBankData_final.csv', encoding='utf-8')


# csv 를 읽는 함수?
def readCSV(csv_path):
    drugCSV = pd.read_csv(csv_path, encoding='utf-8', names=['index', 'indication', 'atc'])
    
    drugNP = np.array(drugCSV)
    drugNP = drugNP[1:len(drugNP),1:3]
    print(drugNP)
    
    texts_data = []
    target_data = []
    
    for one in drugNP:    
        texts_data.append(one[0])
        target_data.append(one[1])
    
    return texts_data, target_data        
        
def main():

#     xml_path = os.path.join('download', 'DB00999.xml.gz')        
#     makeCSV(xml_path)

    csv_path = 'DrugBankData_final.csv'
    texts_data, target_data = readCSV(csv_path)
    
    ### PreProcessing
       
    target = [1 if x=='N' else 0 for x in target_data]
#   
    texts = [x.lower() for x in texts_data]# 모두 소문자
#       
    # 문장 부호(구두점) 제거
    # print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
       
    # 숫자는 제거
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
       
    texts = [' '.join(x.split()) for x in texts] # 여분의 공백(whitespace) 문자 제거
    
    for num in range(5):
        print(texts[num-1])
    print('target_forNLP', target)
#      
    # 문장의 최대 크기를 정해야 한다.
    # 데이터 셋에 포함된 문서 길이의 히스토 그램을 확인하고 값을 정하도록 한다.
      
    # text_lengths : 각 행당 단어들의 갯수를 list 구조로 가지고 있다.
    text_lengths = [len(x.split()) for x in texts]
    print( text_lengths )

    
        # learn.preprocessing 라이브러리에 사전 임베딩 방식을 지정하는 내장된 처리 도구가 있다.
        
    sentence_size = 25
    min_word_freq = 3
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        sentence_size, min_frequency=min_word_freq)
     
    vocab_processor.transform(texts)
     
    embedding_size = len([x for x in vocab_processor.transform(texts)])
     
    # embedding_size : 각 행들을 변형 시키므로 결국 행 갯수가 된다.
    print('embedding_size', embedding_size)
     
    # 학습 셋(80%)과 테스트 셋(20%)으로 분할한다.
    train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
     
    texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
    texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
     
    target_train = [x for ix, x in enumerate(target) if ix in train_indices]
    target_test = [x for ix, x in enumerate(target) if ix in test_indices]
     
#     print('texts_train', texts_train)
#     print('texts_test', texts_test)
#      
#     print('target_train', target_train)
#     print('target_test', target_test)
     
    # embedding_size=3이므로 대각선 요소가 모두 1인 단위 행렬을 만든다.
    identity_mat = tf.diag(tf.ones(shape=[embedding_size]))
#     print('identity_mat', identity_mat)
     
    # Create variables for logistic regression
    A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
     
    # Initialize placeholders
    x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
    y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)
#     print('x_data.shape', x_data.shape)
     
    # Text-Vocab Embedding
    # 임베딩 조회 함수를 이용하여 문장의 단어 인덱스를 단위 행렬의 
    # 원 핫 인코딩 벡터로 할당한다.
    x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
#     print('x_embed', x_embed)
     
    x_col_sums = tf.reduce_sum(x_embed, 0)
#     print('x_col_sums', x_col_sums)
     
    # Declare model operations
    x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
#     print('x_col_sums_2D', x_col_sums_2D)
     
    model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)
     
    # Declare loss function (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
     
    # Prediction operation
    prediction = tf.sigmoid(model_output)
     
    # Declare optimizer
#     my_opt = tf.train.GradientDescentOptimizer(0.01)
    my_opt = tf.train.AdamOptimizer(1e-3)
    train_step = my_opt.minimize(loss)
     
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    ###################################################################################################
    # Start Logistic Regression
    print('Starting Training Over {} Sentences.'.format(len(texts_train)))
    loss_vec = []
    train_acc_all = []
    train_acc_avg = []
    
    epoch = 10
    for num in range(epoch+1): 
        for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
            # ix는 색인
            # t는 원소가 sentence_size(25)인 배열
            # print(t.shape)# shape(5,)
    #         print('--------------------')
    #         print(t)
            # print(type(t)) # <class 'numpy.ndarray'>
                 
            y_data = [[target_train[ix]]]
    #         print('+++++++++++++++++++')
    #         print(y_data)
            sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
        #     temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
            temp_loss, _x, _y = sess.run([loss, x_data, y_target], feed_dict={x_data: t, y_target: y_data})
            loss_vec.append(temp_loss)
             
    #         print('x : ',_x)
    #         print('y : ',_y)
              
        #     print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))     
            if (ix+1)%200==0:
                print('[ epoch ] '+'Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))
                  
            # Keep trailing average of past 50 observations accuracy
            # Get prediction of single observation
            [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
            # Get True/False if prediction is accurate
            train_acc_temp = target_train[ix]==np.round(temp_pred)
            train_acc_all.append(train_acc_temp)
            if len(train_acc_all) >= 50:
                train_acc_avg.append(np.mean(train_acc_all[-50:]))
     
     
    # Get test set accuracy
    print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
    test_acc_all = []
    for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
        y_data = [[target_test[ix]]]
         
        if (ix+1)%50==0:
            print('Test Observation #' + str(ix+1))    
         
        # Keep trailing average of past 50 observations accuracy
        # Get prediction of single observation
        [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
        # Get True/False if prediction is accurate
        test_acc_temp = target_test[ix]==np.round(temp_pred)
        test_acc_all.append(test_acc_temp)
     
    print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
     
    print('끝')
     
    # Plot training accuracy over time
    plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
    plt.title('Avg Training Acc Over Past 50 Generations')
    plt.xlabel('Generation')
    plt.ylabel('Training Accuracy')
    plt.show()

     
if __name__ == '__main__':
    main()
    print('다했다옹!!!')

 

# 
# # 각 행마다 50개 미만의 단어로 구성된 행만 취하겠다. 
# text_lengths = [x for x in text_lengths if x < 50] 
# print('text_lengths', text_lengths) 
# 
# # plt.hist(text_lengths, bins=25)
# # plt.title('Histogram of # of Words in Texts')
# # plt.show()
# 
# # learn.preprocessing 라이브러리에 사전 임베딩 방식을 지정하는 내장된 처리 도구가 있다.
# vocab_processor = learn.preprocessing.VocabularyProcessor(
#     sentence_size, min_frequency=min_word_freq)
# 
# vocab_processor.transform(texts)
# 
# embedding_size = len([x for x in vocab_processor.transform(texts)])
# 
# # embedding_size : 각 행들을 변형 시키므로 결국 행 갯수가 된다.
# print('embedding_size', embedding_size)
# 
# # 학습 셋(80%)과 테스트 셋(20%)으로 분할한다.
# train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
# test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
# 
# texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
# texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
# 
# target_train = [x for ix, x in enumerate(target) if ix in train_indices]
# target_test = [x for ix, x in enumerate(target) if ix in test_indices]
# 
# print('texts_train', texts_train)
# print('texts_test', texts_test)
# 
# print('target_train', target_train)
# print('target_test', target_test)
# 
# # embedding_size=3이므로 대각선 요소가 모두 1인 단위 행렬을 만든다.
# identity_mat = tf.diag(tf.ones(shape=[embedding_size]))
# print('identity_mat', identity_mat)
# 
# # Create variables for logistic regression
# A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
# b = tf.Variable(tf.random_normal(shape=[1,1]))
# 
# # Initialize placeholders
# x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
# y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)
# print('x_data.shape', x_data.shape)
# 
# # Text-Vocab Embedding
# # 임베딩 조회 함수를 이용하여 문장의 단어 인덱스를 단위 행렬의 
# # 원 핫 인코딩 벡터로 할당한다.
# x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
# print('x_embed', x_embed)
# 
# x_col_sums = tf.reduce_sum(x_embed, 0)
# print('x_col_sums', x_col_sums)
# 
# # Declare model operations
# x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
# print('x_col_sums_2D', x_col_sums_2D)
# 
# model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)
# 
# # Declare loss function (Cross Entropy loss)
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
# 
# # Prediction operation
# prediction = tf.sigmoid(model_output)
# 
# # Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.001)
# train_step = my_opt.minimize(loss)
# 
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# ###################################################################################################
# # Start Logistic Regression
# print('Starting Training Over {} Sentences.'.format(len(texts_train)))
# loss_vec = []
# train_acc_all = []
# train_acc_avg = []
# 
# for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
#     # ix는 색인
#     # t는 원소가 sentence_size(25)인 배열
#     # print(t.shape)# shape(5,)
#     print('--------------------')
#     print(t)
#     # print(type(t)) # <class 'numpy.ndarray'>
#         
#     y_data = [[target_train[ix]]]
#     print('+++++++++++++++++++')
#     print(y_data)
#     sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
# #     temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
#     temp_loss, _x, _y = sess.run([loss, x_data, y_target], feed_dict={x_data: t, y_target: y_data})
#     loss_vec.append(temp_loss)
#     
#     print('x : ',_x)
#     print('y : ',_y)
#      
# #     print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))     
#     if (ix+1)%10==0:
#         print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))
#          
#     # Keep trailing average of past 50 observations accuracy
#     # Get prediction of single observation
#     [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
#     # Get True/False if prediction is accurate
#     train_acc_temp = target_train[ix]==np.round(temp_pred)
#     train_acc_all.append(train_acc_temp)
#     if len(train_acc_all) >= 50:
#         train_acc_avg.append(np.mean(train_acc_all[-50:]))
# 
# 
# # Get test set accuracy
# print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
# test_acc_all = []
# for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
#     y_data = [[target_test[ix]]]
#     
#     if (ix+1)%50==0:
#         print('Test Observation #' + str(ix+1))    
#     
#     # Keep trailing average of past 50 observations accuracy
#     # Get prediction of single observation
#     [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
#     # Get True/False if prediction is accurate
#     test_acc_temp = target_test[ix]==np.round(temp_pred)
#     test_acc_all.append(test_acc_temp)
# 
# print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
# 
# print('끝')
# 
# # Plot training accuracy over time
# plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
# plt.title('Avg Training Acc Over Past 50 Generations')
# plt.xlabel('Generation')
# plt.ylabel('Training Accuracy')
# plt.show()
# 


