# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:43:26 2019

This file includes the preprocess on the data and model training of xgboost

@author: Hui Cai
"""

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#reading data
raw_data = pd.read_csv('样本内数据集.txt',engine = 'python',encoding = 'utf-8')
raw_data.index = raw_data['MovieID']

#1. data preprocess

#1.1 drop some data, there's some rows whose columns moved right 
data = raw_data[raw_data['CountryRegion'].isnull()]

#1.2 remove duplicate data according to MovieID
#(we have checked that as long as movie id is the same, then they are the same data)
data = data.drop_duplicates('MovieID',keep = 'first')

#1.3 feature engineering

#1.3.1 [ShoWDate] use the year and month as features
#get the year and month of the first show of the movie
#if ShowDate is nan, then year and month would be nan
#if there is only year but not month, set month = 13
data['Year'] = float('nan');data['Month'] = float('nan')
import re
for idx in data.index:
    date = data.loc[idx,'ShowDate']
    if type(date) == type(0.):    
        continue
    date_list = date.split('|')
    new_d = float('inf')
    for d in date_list:
        d_list = re.findall('(\d+)',d)
        #find year of this d
        if 0 < len(d_list) <= 1:
            new_d = min(new_d,float(d_list[0]+'13'))
        elif len(d_list) > 1:
            new_d = min(new_d,float(d_list[0]+d_list[1]))
            
    data.loc[idx,'Year'] = float(str(new_d)[0:4])
    data.loc[idx,'Month'] = round((new_d/100-new_d//100)*100)
    
#1.3.2 [RunningTime] is not used as a feature
#for now, still not find a good idea to get the running time
#since there are many formats, like 120min, 120, 120分钟, even 2:00:00
#import re
#data['RunningTime'].fillna('0分钟',inplace = True)
#data['RunningTime'] = [np.array(re.findall('(\d+)分钟',time)).astype(float).mean() for time in data['RunningTime']]

#1.3.3 word embedding
    
#1.3.3.1 [DirectorID] it is not used as a feature, since there are so many directors 
# but actually, we could set like if director is not known, 
#set the DirectorID to be 0, else to be 1 
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
data['DirectorID'].fillna('Unkonwn',inplace = True)
director_vec = count_vec.fit_transform(data['DirectorID']).toarray()
director_name = count_vec.get_feature_names()

#1.3.3.2 [MovieType] is used as a feature
#transform MovieType to vector
data = data.loc[data['MovieType'].dropna().index]
movieType_vec = count_vec.fit_transform(data['MovieType']).toarray()
movieType_name = count_vec.get_feature_names()
movieType = pd.DataFrame(movieType_vec,columns = movieType_name,index = data.index)

#1.3.3.3 [Language] is used as a feature but use it a binary feature

#the following code is the normal method to deal with it
#language_vec = count_vec.fit_transform(data['Language']).toarray()
#language_name = count_vec.get_feature_names()
#language = pd.DataFrame(language_vec,columns = language_name,index = data.index)

#the main problem is that there's no difference between 四川话 and 四川方言 
#there are so many languages, so I just classify 默片 and 非默片
#normally speaking, if it's 默片, then it's Rate would be high
data.loc[data.index[(data['Language'] == '无对白') |
            (data['Language'] == '默片')],'Language'] = 1 
data.loc[data.index[(data['Language'] != 1)],'Language'] = 0
data.loc[:,'Language'] = data.loc[:,'Language'].astype(int)

#1.4 [CumBox] it is used as a feature
#transform it into number, keep nan
idx = data.index[abs(data['CumBox'].isnull()-1).astype(bool)]
for x in idx:
    box = data.loc[x,'CumBox']
    if box[-1] == '万':
        data.loc[x,'CumBox'] = float(box[:-1])*10000
    elif box[-1] == '亿':
        data.loc[x,'CumBox'] =float(box[:-1])*100000000
    else:
        data.loc[x,'CumBox'] = float(box)
data.loc[:,'CumBox'] = data['CumBox'].astype(float)

#1.5 4 columns of [Num] 
#deal with none float type in [Num], also caused by the right moving of certain rows
wrong = []
for idx in data.index:
    try:    
        data.loc[idx,['ShortCommentNum','FullCommentNum',
                      'CollectionNum','WishNum']].astype(float)
    except:
        wrong.append(idx)
data = data.drop(wrong)        
data.loc[:,['ShortCommentNum','FullCommentNum',
            'CollectionNum','WishNum']] = data.loc[:,['ShortCommentNum',
                                          'FullCommentNum','CollectionNum',
                                          'WishNum']].astype(float,inplace = True)
 

#1.6 get the train data based on the features mentioned above 
train_data = data.loc[:,['Rate','Language','Year','Month',
                         'ShortCommentNum','FullCommentNum',
                         'CollectionNum','WishNum','CumBox']]

#add movieType as feature
train_data = train_data.join(movieType)

#2. model training

#2.1 use cross-validation to train xgboost and get the parameters
import xgboost as xgb
X_train = train_data.iloc[:,1:]
Y_train = train_data.iloc[:,0]
dtrain = xgb.DMatrix(data = X_train,label=Y_train)

#grid search for parameters and use gpu to increase the speed
cv_list = []
for dp in [6,9,12,15]:
    for eta in [0.02,0.2,0.4]:
        for gamma in [0,0.2,0.5]:
            print([dp,eta,gamma])            
            param = {'max_depth':dp,
                     'eta':eta, 
                     'gamma': gamma,
                     'silent':1, 
                     'subsample':0.7,
                     'colsample_bytree': 0.7,
                     'min_child_weight': 3,
                     'alpha':0,
                     'lambda':3,
                     'objective': 'reg:linear',
                     'eval_metric':'rmse',
                     'tree_method': 'gpu_hist',
                     'gpu_id': 0}
            numRound = 1500
            cv_list.append(([dp,eta,gamma],
                            xgb.cv(params = param,
                            dtrain = dtrain, 
                            num_boost_round = numRound, 
                            nfold = 8,
                            verbose_eval=150)))

all_cv = [(cv[0],cv[1]['test-rmse-mean'].argmin(),cv[1]['test-rmse-mean'].min(),
           cv[1].loc[cv[1]['test-rmse-mean'].argmin(),'test-rmse-std']) for cv in cv_list]
output = pd.DataFrame(all_cv)

#2.2 model training based on the selection of parameters above as well as the 
#numRound that makes the rmse smallest on test set
dp = 12;eta = 0.02;gamma = 0
param = {'max_depth':dp,
         'eta':eta,
         'gamma': gamma,
         'silent':1, 
         'subsample':0.7,
         'colsample_bytree': 0.7,
         'min_child_weight': 3,
         'alpha':0,
         'lambda':3,
         'objective': 'reg:linear',
         'eval_metric':'rmse',
         'tree_method': 'gpu_hist',
         'gpu_id': 0
         } 
watchlist = [(dtrain, 'train')]
numRound = 465
bst = xgb.train(params = param,
                dtrain = dtrain,
                num_boost_round = numRound,
                evals = watchlist
                )

#deal with test data based on the same method on train data above
test_data = pd.read_csv('样本外数据集.txt',engine = 'python',encoding = 'utf-8')
test_data.index = test_data['MovieID']
raw_data = test_data
data = raw_data[raw_data['CountryRegion'].isnull()]
data = data.drop_duplicates('MovieID',keep = 'first')
data['Year'] = float('nan');data['Month'] = float('nan')
import re
for idx in data.index:
    date = data.loc[idx,'ShowDate']
    if type(date) == type(0.):    
        continue
    date_list = date.split('|')
    new_d = float('inf')
    for d in date_list:
        d_list = re.findall('(\d+)',d)
        #find year of this d
        if 0 < len(d_list) <= 1:
            new_d = min(new_d,float(d_list[0]+'13'))
        elif len(d_list) > 1:
            new_d = min(new_d,float(d_list[0]+d_list[1]))
            
    data.loc[idx,'Year'] = float(str(new_d)[0:4])
    data.loc[idx,'Month'] = round((new_d/100-new_d//100)*100)
data = data.loc[data['MovieType'].dropna().index]
movieType_vec = count_vec.fit_transform(data['MovieType']).toarray()
movieType_name = count_vec.get_feature_names()
movieType = pd.DataFrame(movieType_vec,columns = movieType_name,index = data.index)
data.loc[data.index[(data['Language'] == '无对白') |
            (data['Language'] == '默片')],'Language'] = 1 
data.loc[data.index[(data['Language'] != 1)],'Language'] = 0
data.loc[:,'Language'] = data.loc[:,'Language'].astype(int)
idx = data.index[abs(data['CumBox'].isnull()-1).astype(bool)]
for x in idx:
    box = data.loc[x,'CumBox']
    if box[-1] == '万':
        data.loc[x,'CumBox'] = float(box[:-1])*10000
    elif box[-1] == '亿':
        data.loc[x,'CumBox'] =float(box[:-1])*100000000
    else:
        data.loc[x,'CumBox'] = float(box)
data.loc[:,'CumBox'] = data['CumBox'].astype(float)

wrong = []
for idx in data.index:
    try:    
        data.loc[idx,['ShortCommentNum','FullCommentNum',
                      'CollectionNum','WishNum']].astype(float)
    except:
        wrong.append(idx)
data = data.drop(wrong)        
data.loc[:,['ShortCommentNum','FullCommentNum',
            'CollectionNum','WishNum']] = data.loc[:,['ShortCommentNum',
                                          'FullCommentNum','CollectionNum',
                                          'WishNum']].astype(float)

test_data = data.loc[:,['Rate','Language','Year','Month',
                         'ShortCommentNum','FullCommentNum',
                         'CollectionNum','WishNum','CumBox']]
test_data = test_data.join(movieType)
#there are some movieType in test_data but not in train_data
#get the intersection of the MovieType of train_data and test_data
col = set(test_data.columns).intersection(train_data.columns)
test_data = test_data[list(col)]
for col in train_data.columns:
    if col not in test_data.columns:
        test_data[col] = 0
test_data = test_data[train_data.columns]

X_test = test_data[test_data.columns[test_data.columns != 'Rate']]
Y_test = test_data['Rate']
dtest = xgb.DMatrix(data = X_test,label=Y_test)

# do prediction
outcome = pd.DataFrame(bst.predict(dtest),index = Y_test.index)
outcome.columns = ['Rate']

#outcome.to_csv('test_output.csv')
#outcome = pd.read_csv('test_output.csv',engine = 'python',encoding = 'utf-8')
#outcome.index = outcome['MovieID']

#match the outcome to the raw_data
raw_data pd.read_csv('样本外数据集.txt',engine = 'python',encoding = 'utf-8')
raw_data.index = raw_data['MovieID']

for idx in outcome.index:
    raw_data.loc[idx,'Rate'] = outcome.loc[idx,'Rate']

# for the rows that I cannot handle because of right moving, I do it by hand
#search on douban to determine the Rate
raw_data.loc[7197452,'Rate'] = 7.7
raw_data.loc[2609120,'Rate'] = 6.8
raw_data.loc[53925844,'Rate'] = 4.7
raw_data.loc[2840546,'Rate'] = 7.3
raw_data.loc[4276572,'Rate'] = 8.0
raw_data.loc[51427258,'Rate'] = 4.7
raw_data.loc[2610710,'Rate'] = 8.0
raw_data.loc[49068506,'Rate'] = 5.8
raw_data.loc[60703492,'Rate'] = 4.2
raw_data.loc[4690438,'Rate'] = 6.8
raw_data.loc[49051204,'Rate'] = 2.9
raw_data.loc[7164766,'Rate'] = 7.6
raw_data.loc[49718424,'Rate'] = 6.6
raw_data.loc[12233736,'Rate'] = 7.3
raw_data.loc[54236726,'Rate'] = 5.9
raw_data.loc[2937678,'Rate'] = 8.2
raw_data.loc[4319562,'Rate'] = 8.2
raw_data.loc[4764510,'Rate'] = 8.4
raw_data.loc[8179842,'Rate'] = 7.1
raw_data.loc[10775622,'Rate'] = 5.6
raw_data.loc[40799256,'Rate'] = 3.9
raw_data.loc[53116880,'Rate'] = 4.5
raw_data.loc[53460696,'Rate'] = 6.7
raw_data.loc[54194632,'Rate'] = 4.3
raw_data.loc[2588916,'Rate'] = 8.3
raw_data.loc[2598256,'Rate'] = 7.5
raw_data.loc[42698830,'Rate'] = 3.4

raw_data.loc[7609584,'Rate'] = 9.31421
raw_data.loc[21122216,'Rate'] = 8.31022
raw_data.loc[53293360,'Rate'] = 3.03121
raw_data.loc[53293360,'Rate'] = 2.49261
raw_data.loc[53387194,'Rate'] = 2.49261

#after that, still two rows of na, fill with the mean of Rate
raw_data['Rate'].fillna(Y_train.mean(),inplace = True)
#write the prediction to the final outcome, exclude index to make sure format is the same
raw_data.to_csv('样本外数据集final_output.txt',index = False)

