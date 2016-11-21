import os 
import sys 
import gensim 
import nltk 
import numpy as np 
import pandas as pd
import re
from pyemd import emd 
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cross_validation import train_test_split 
import operator
from scipy.stats.stats import pearsonr
import json
import math
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer


from gensim.models import Word2Vec
model = Word2Vec.load("only_coravin_more")


qst_ans = pd.read_csv("/Users/joshua/BotCentral/word2vec_training/relationship.csv", sep = ",", header=None)
qst_data = pd.DataFrame(qst_ans)
main = qst_data.loc[:,1]
quest = qst_data.loc[:,0] 
reslt = qst_data.loc[:,2]
quest1=quest.copy()
n=quest.shape[0] 

qst_lst = ["" for x in range(n)]
sim_lst = ["" for x in range(n)]

while True: 
    sent = raw_input('Enter Sentence (Type End to stop): ')
    sent01 = sent
    if sent == 'End': 
        print('Found end and quitting...') 
        break
    n=quest.shape[0]
    quest = quest1.copy()

    print(n)
    for i in range(n):
        sentence = quest[i]

        sentence1 =  re.sub("[^a-zA-Z]"," ", sent)
        sentence2 =  re.sub("[^a-zA-Z]"," ", sentence)

        sentence1  = sentence1.lower().split()
        sentence2 = sentence2.lower().split() 

        # Remove non-letters
        #sentence1 = re.sub("[^a-zA-Z]"," ", sentence1)
        #sentence1 =  re.sub(r'[^a-zA-Z0-9 ]',r'',sentence1)
        #sentence2 = re.sub("[^a-zA-Z]"," ", sentence2)

        #print(sentence)
        from nltk.corpus import stopwords 
        stopwords = nltk.corpus.stopwords.words('english') 
        sentence1 = [w for w in sentence1 if w not in stopwords] 
        sentence2 = [w for w in sentence2 if w not in stopwords] 

        distance = model.wmdistance(sentence1, sentence2) 
        #distance = model.1 - spatial.distance.cosine(sentence1, sentence2) 
        #print(sentence2) 
        if distance == float('inf'):
            distance = 0

        #print(sentence, distance)

        qst_lst[i] = sentence
        sim_lst[i] = distance
  
    qst_lst1 = pd.DataFrame(qst_lst)
    sim_wmd = pd.DataFrame(sim_lst)

    quest[0] = sent
    quest1[0] = sent
    from nltk.corpus import stopwords 
    s=set(stopwords.words('english'))
    for k in range(n):
        sent = quest.loc[k]
        sent1 = re.sub(r'[^a-zA-Z0-9 ]',r'',sent)
        low_str=' ' + sent1.lower() + ' '
        sent2 = " ".join(filter(lambda w: not w in s,low_str.split()))
        #wnl = WordNetLemmatizer()
        #quest[k] = " ".join([wnl.lemmatize(i) for i in sent2.split()])
        port = PorterStemmer()
        quest[k] =  " ".join([port.stem(i) for i in sent2.split()])
    
    #s=set(stopwords.words('english'))
    strng = " ".join(line.strip() for line in quest)
    nestr = re.sub(r'[^a-zA-Z0-9 ]',r'',strng)
    low_str=' ' + nestr.lower() + ' '

    word_all= filter(lambda w: not w in s,low_str.split())
    uni_wrds=pd.DataFrame(dict.fromkeys(word_all).keys())

    n=quest.shape[0]
    m=uni_wrds.shape[0]

    data = np.zeros((n,m))
    for j in range(m):
        for i in range(n):
            wd = ' ' + uni_wrds.loc[j,0] + ' '
            doc = ' ' + re.sub(r'[^a-zA-Z0-9 ]',r'',quest.loc[i]).lower() + ' '
            cnt = doc.count(wd)
            data[i,j] = cnt 
    tf={}
    tf = 1+np.log10(data)
    tf[tf==-np.inf]= 0
    tf[tf==np.inf]= 0

    v_len=np.sqrt(np.sum(np.square(tf), axis=1))

    # Normalizing the data, row wise documents and column wise unique words
    df=np.sum(data, axis=0)
    total_df=sum(df)

    # Calculating Inverse Document Frequency
    idf=np.log10(total_df/df)
    norm=(tf.T/v_len.T).T
    n=len(data)
    m=data.shape[1]

    doc_sim=np.zeros((n,n))
    doc_sim=np.around(np.dot(norm,norm.T), decimals=2)

    sim = doc_sim[0,:].T
    
    #qst_lst = pd.DataFrame(quest1)
    sim_tf = 1 - pd.DataFrame(sim)    

    sim_tot = sim_wmd+sim_tf 
    sim_tot.loc[0] = 5
    similarities = pd.DataFrame(np.concatenate((qst_lst1,sim_wmd, sim_tf, sim_tot), axis=1), columns=['QUESTIONS','WMD','TF','TOT'])

    index = sim_tot.idxmin()

    similarities = similarities.sort(['TOT'], ascending=[False])

    pd.options.display.max_colwidth = 50

    print(similarities)

    #index, value = min(enumerate(sim_lst1), key=operator.itemgetter(0)) 

    print(" ")
    print('you : ', sent01)

    print('Similar question is  : ', main[index])
    print(" ")
    
    pd.options.display.max_colwidth = 500

    print('Bot : ', reslt.iloc[index])
    print(" ")
    


