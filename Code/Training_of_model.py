#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge,RidgeCV#线性回归模型的岭回归
from datetime import datetime
# from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier     #随机森林用于分类
from sklearn.ensemble import RandomForestRegressor as RFR      #随机森林用于回归
from sklearn.model_selection import train_test_split           #划分训练集与测试集
from sklearn import metrics    
from sklearn.metrics import r2_score              #用于模型拟合优度评估
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
pd.options.display.max_rows = None

 


# In[194]:


data = pd.read_excel('polymer-6features.xlsx')
data.head()


# In[4]:


def Create_Table(model):
    
    Table_train = data.iloc[data.index.isin(X_train_index),1:-2]
    Table_train.insert(loc=0, column='NAME', value=data.loc[data.index.isin(X_train_index),"NAME"])
    Table_train['PCE'] = data.loc[data.index.isin(X_train_index),'PCE']
    Table_train['Predict'] = model.predict(stand_scaler.transform(Table_train.iloc[:,1:-1]))

    Table_test = data.iloc[data.index.isin(X_test_index),1:-2]
    Table_test.insert(loc=0, column='NAME', value=data.loc[data.index.isin(X_test_index),"NAME"])
    Table_test['PCE'] = data.loc[data.index.isin(X_test_index),'PCE']
    Table_test['Predict'] = model.predict(stand_scaler.transform(Table_test.iloc[:,1:-1]))

    part_row = pd.DataFrame([[np.NaN]*len(Table_train.columns)],columns = Table_train.columns)

    Table_train = Table_train.append(part_row)
    Table_train = Table_train.append(Table_test)

    
    return Table_train


# In[197]:


Model_series=Series()
Table_series = Series()
Log_df = pd.DataFrame(columns=['Train_r2','Test_r2','Train_RMSE','Train_MAE','Test_RMSE','Test_MAE'])


# In[18]:


Model_series1=Model_series.copy()
Table_series1=Table_series.copy()
Log_df1=Log_df.copy()


# In[130]:


Model_series2=Model_series.copy()
Table_series2=Table_series.copy()
Log_df2=Log_df.copy()


# In[169]:


Model_series3=Model_series.copy()
Table_series3=Table_series.copy()
Log_df3=Log_df.copy()


# In[196]:


Model_series4=Model_series.copy()
Table_series4=Table_series.copy()
Log_df4=Log_df.copy()


# In[175]:


data.shape


# In[195]:


# 划分训练集与测试集并保持固定
random_data = shuffle(data)
X = random_data.iloc[:,1:-2] #特征
y = random_data.loc[:,['PCE']] #预测值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=16/116, random_state=66)  
X_train_index = X_train.index
y_train_index = y_train.index
X_test_index = X_test.index
y_test_index  = y_test.index


# In[202]:


i=1
#lines = Feature_name9
random_data = shuffle(data)
while(1):
    print("开始第"+str(i)+"次运行********************************")

    
    # 数据标准化处理
    stand_scaler = StandardScaler()
    X_train = stand_scaler.fit_transform(X_train)
    X_test = stand_scaler.transform(X_test) 
#     X_predict = stand_scaler.transform(X_predict) 

    # 使用GBRT模型的代码
    GDBT = GradientBoostingRegressor(random_state=50)
    GDBT.fit(X_train,y_train)
    predictions= GDBT.predict(X_test)
    r2_train = round(r2_score(y_train,GDBT.predict(X_train)),3)
    r2_test = round(r2_score(y_test,predictions),3)
    param_grid ={
        "n_estimators": range(80,300,10),
         "min_samples_leaf": [1,5,10,15,27,30,35],
        "learning_rate": [0.001,0.005,0.01,0.015, 0.02,0.03,0.1,0.3,0.5,0.7],
        "max_depth":[2, 3, 4, 5,8,10],
        "min_samples_split": [2, 4,6,8.10],
        'max_features':['auto','sqrt','log2'],
    }
    
    optimizestarttime = datetime.now()
    print("开始对GBRT优化")
    rnd_search = RandomizedSearchCV(GDBT, param_distributions=param_grid,
                            n_iter=500, cv=5, scoring='r2',n_jobs=-1)
    rnd_search.fit(X_train, y_train)
    finishtime = datetime.now()
    print("完成次对GBRT优化"+'Running time: %s Seconds'%(finishtime-optimizestarttime))
    
    
    #使用RF模型的代码如需使用就解除注释
    '''
    #构造随机森林模型
    #rf=RFR(n_estimators = 800,oob_score = True,n_jobs = -1,random_state =42,max_features='auto',min_samples_leaf = 4)
    rf=RFR(n_jobs = -1)
    rf.fit(X_train,y_train)               #模型拟合
    predictions= rf.predict(X_test)                 #预测
    r2_train = round(r2_score(y_train,rf.predict(X_train)),3)
    r2_test = round(r2_score(y_test,predictions),3)
    #r2_predic = round(r2_score(y_predict,GDBT.predict(X_predict)),3)

    print("训练集：",round(r2_train,3))  
    print("测试集：",round(r2_test,3))  
    #print("验证集：",round(r2_predic,3))  

    param_grid = {
        "n_estimators": range(10,300,5),
        "max_depth": [2, 4, 6, 8, 10, 12, 14],
        "min_samples_split": [2,4,6,8,10,12],
        'max_features':['auto','sqrt'],
        "min_samples_leaf":range(5,50,2),
    } 
    optimizestarttime = datetime.now()
    print("开始第"+str(i)+"次对随机森林优化")

    rnd_search = RandomizedSearchCV(rf, param_distributions=param_grid,
                            n_iter=500, cv=5, scoring='r2',n_jobs=-1)
    rnd_search.fit(X_train, y_train)

    finishtime = datetime.now()
    print("完成第"+str(i)+"次对随机森林优化"+'Running time: %s Seconds'%(finishtime-optimizestarttime))
    '''
    
    
    #SVR使用代码
    '''
    #SVR训练

    parameters =[ 
    {
    'C':  np.logspace(-3, 3, 100000000),
    'gamma': np.logspace(-3, 3,100000000),
    'kernel': ['rbf','linear','Poly'],
    'cache_size':[10000]
    }
    ]
    optimizestarttime = datetime.now()
    print("开始第"+str(i)+"次对SVR优化")

    svr = RandomizedSearchCV(SVR(),parameters, cv=5)

    svr.fit(X_train, y_train)
    finishtime = datetime.now()
    print("完成第"+str(i)+"次对SVR优化"+'Running time: %s Seconds'%(finishtime-optimizestarttime))
    '''
    #XGBoost代码如需使用解除以下注释
    '''
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=9, learning_rate=0.09, n_estimators=100, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)
    train_pre = model.predict(X_train)
    test_pre = model.predict(X_test)
    #predict_pre = model.predict(X_predict)


    train_score = np.round(r2_score(y_train,train_pre),3)
    test_score = np.round(r2_score(y_test,test_pre),3)
    #predict_score = np.round(r2_score(y_predict,predict_pre),3)

    print("train_score:",train_score)
    print("test_score:",test_score)
    #print("predict_score:",predict_score)
    print("开始优化:")
    Optimization_start = datetime.now()

    param_grid = {
        'n_estimators':[100,200,400,1000],
        'learning_rate':[0.001,0.01, 0.02, 0.05, 0.1, 0.15,0.1],
        'alpha':[0, 0.01,0.03,0.05,0.07,0.09,0.1, 1],
        'lambda':[0, 0.1, 0.5, 1],
        'max_depth':[3, 5, 6, 7, 9, 12, 15, 17, 25],
        'min_child_weight':[1, 3, 5, 7],
        'gamma':[0, 0.05,0.07,0.09,0.1, 0.3, 0.5, 0.7, 0.9, 1],
        'subsample':[0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1]
    }      #要调优的参数

    optimizestarttime = datetime.now()
    print("开始第"+str(i)+"次对XGBoost优化")

    rnd_search = RandomizedSearchCV(model, param_distributions=param_grid,
                            n_iter=500, cv=5, scoring='r2',n_jobs=-1)
    rnd_search.fit(X_train, y_train)

    finishtime = datetime.now()
    print("完成第"+str(i)+"次对XGBoost优化"+'Running time: %s Seconds'%(finishtime-optimizestarttime))
    '''
    
    
    best_model = rnd_search.best_estimator_
    best_model.fit(X_train,y_train)
    predictions= best_model.predict(X_test)                 #预测
    bestmodel_r2_train = r2_score(y_train,best_model.predict(X_train))
    bestmodel_r2_test = r2_score(y_test,predictions)
#     bestmodel_r2_predic = r2_score(y_predict,best_model.predict(X_predict))

    standard_value = round((bestmodel_r2_test + bestmodel_r2_train)/2,3)
    
    print("训练集：",round(r2_train,3))  
    print("测试集：",round(r2_test,3))  
#     print("验证集：",round(r2_predic,3))  

    print("调优后训练集：",round(bestmodel_r2_train,3))  
    print("调优后测试集：",round(bestmodel_r2_test,3))    
#     print("调优后验证集：",round(bestmodel_r2_predic,3))
    
    Train_RMSE = np.round(np.sqrt(mean_squared_error(y_train,best_model.predict(X_train))),3)
    Train_MAE = np.round(mean_absolute_error(y_train,best_model.predict(X_train)),3)
    Test_RMSE = np.round(np.sqrt(mean_squared_error(y_test,predictions)),3)
    Test_MAE = np.round(mean_absolute_error(y_test,predictions),3)
#     Predict_RMSE = np.round(np.sqrt(mean_squared_error(y_predict,best_model.predict(X_predict))),3)
#     Predict_MAE = np.round(mean_absolute_error(y_predict,best_model.predict(X_predict)),3) 
    
#     if(bestmodel_r2_test>0.81):
#         break
# 显示网格图
    if(bestmodel_r2_train>0.6):
        print(best_model)
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(y_train,best_model.predict(X_train))
        plt.plot(range(1,20),range(1,20))
        plt.grid()
        plt.subplot(122)
        plt.scatter(y_test,best_model.predict(X_test))
        plt.plot(range(1,20),range(1,20))
        plt.grid()
        plt.show()
    if(bestmodel_r2_test>0.5):
        print(best_model)
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(y_train,best_model.predict(X_train))
        plt.plot(range(1,20),range(1,20))
        plt.grid()
        plt.subplot(122)
        plt.scatter(y_test,best_model.predict(X_test))
        plt.plot(range(1,20),range(1,20))
        plt.grid()
        plt.show()
        
#         模型保存至候选列表
    if(bestmodel_r2_test>0.5 and bestmodel_r2_train > 0.5):
        Table = Create_Table(best_model)
        if len(Log_df)<10:
            Model_series[standard_value] = best_model
            Table_series[standard_value] = Table
            Log_df.loc[standard_value] = [bestmodel_r2_train,bestmodel_r2_test,Train_RMSE,Train_MAE,Test_RMSE,Test_MAE]
            Log_df = Log_df.sort_index(ascending=False)
        elif (standard_value > min(Model_series.index)):
            sorted_value = sorted(Model_series.index)
            for j in range(len(Model_series.index)):
                if standard_value == sorted_value[j]:
                    if Log_df[Log_df.index == sorted_value[j]]['Test_r2']  <  bestmodel_r2_test:
                        break
                if standard_value > sorted_value[j]:
                    break
            del Model_series[sorted_value[j]]
            del Table_series[sorted_value[j]]
            Log_df.drop(sorted_value[j],inplace=True)
            Model_series[standard_value] = best_model
            Table_series[standard_value] = Table
            Log_df.loc[standard_value] = [bestmodel_r2_train,bestmodel_r2_test,Train_RMSE,Train_MAE,Test_RMSE,Test_MAE]
            Log_df = Log_df.sort_index(ascending=False)
    print("已保存"+str(len(Log_df))+"个优秀模型*****")    
    if(len(Log_df)>0):
        print(Log_df)     
    i+=1
    


# In[182]:


Model_series.index


# In[142]:


print(best_model)


# In[95]:


Model_series[0.626]


# In[26]:


# 保存合适的模型
joblib.dump(Model_series1[0.613],'model0613.pkl')


# In[27]:


# 将数据输出为Excel表格
Table_series1[0.613].to_excel("Table0613.xlsx")


# In[22]:


#散点图矩阵
import seaborn as sns
fig = plt.figure(figsize=(10,10))
sns.pairplot(data=data, vars=data.iloc[:,1:-2], diag_kind="kde", markers="o")
plt.show()

