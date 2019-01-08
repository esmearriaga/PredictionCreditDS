#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:04:43 2018

@author: EsmeArriaga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
#from collections import Counter
#%% Descarga de datos
dir_file = 'cs-training.csv' 
creditos = pd.read_csv(dir_file, 
                        header = 0,
                        sep = ',',
                        index_col = None)

#%% Eliminar los na
creditos=creditos.dropna()

#%% Descripción básica
#Reporte basico
quick_report_trab1 = pd.DataFrame(creditos.describe().transpose())

#Revolving utilization
ulines=pd.DataFrame(creditos.groupby(['SeriousDlqin2yrs'])['RevolvingUtilizationOfUnsecuredLines'].mean())
ulines=ulines.reset_index()
ax = sns.barplot(x='SeriousDlqin2yrs', y='RevolvingUtilizationOfUnsecuredLines', data=ulines)
plt.title('Number of revolving utilization of unsecured lines')
plt.xlabel('Serious delinquence in 2 years')
plt.ylabel('Revolving utilization of unsecured lines')
plt.show()

#Age
age=pd.DataFrame(creditos.groupby(['age'])['SeriousDlqin2yrs'].mean())
age=age.reset_index()
age=age.iloc[:-10]
age=age.sort_values(['age'], ascending=[True])
plt.figure(figsize=(10,4))
ax = sns.barplot(x='age', y='SeriousDlqin2yrs', data=age)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.title('Mean of serious delinquence in 2 years by age')
plt.xlabel('Age')
plt.ylabel('Serious delinquence in 2 years')
plt.tight_layout()
plt.show()

#Number of times between 30 and 60
n3060=pd.DataFrame(creditos.groupby(['NumberOfTime30-59DaysPastDueNotWorse'])['SeriousDlqin2yrs'].sum())
n3060=n3060.reset_index()
n3060=n3060.iloc[:-1]
n3060=n3060.sort_values(['SeriousDlqin2yrs'], ascending=[False])
ax = sns.barplot(x='SeriousDlqin2yrs', y='NumberOfTime30-59DaysPastDueNotWorse', data=n3060)
plt.title('Number of times between 30 and 60 days past but not worse')
plt.xlabel('Number of times between 30 and 60 days past')
plt.ylabel('Serious delinquence in 2 years')
plt.show()

#Debt ratio
dratio=pd.DataFrame(creditos.groupby(['SeriousDlqin2yrs'])['DebtRatio'].mean())
dratio=dratio.reset_index()
dratio=dratio.sort_values(['DebtRatio'], ascending=[True])
ax = sns.barplot(x='SeriousDlqin2yrs', y='DebtRatio', data=dratio)
plt.title('Mean of debt ratio')
plt.xlabel('Serious delinquence in 2 years')
plt.ylabel('Debt ratio')
plt.show()

#Monthly income
minco=pd.DataFrame(creditos.groupby(['SeriousDlqin2yrs'])['MonthlyIncome'].mean())
minco=minco.reset_index()
minco=minco.sort_values(['MonthlyIncome'], ascending=[True])
ax = sns.barplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=minco)
plt.title('Mean of monthly income')
plt.xlabel('Serious delinquence in 2 years')
plt.ylabel('Monthly income')
plt.show()

#Number of open credits
noline=pd.DataFrame(creditos.groupby(['SeriousDlqin2yrs'])['NumberOfOpenCreditLinesAndLoans'].mean())
noline=noline.reset_index()
noline=noline.sort_values(['NumberOfOpenCreditLinesAndLoans'], ascending=[True])
ax = sns.barplot(x='SeriousDlqin2yrs', y='NumberOfOpenCreditLinesAndLoans', data=noline)
plt.title('Mean of number of open credit lines and loans')
plt.xlabel('Serious delinquence in 2 years')
plt.ylabel('Number of open lines and loans')
plt.show()

#Number of 90 days late
no90=pd.DataFrame(creditos.groupby(['NumberOfTimes90DaysLate'])['SeriousDlqin2yrs'].mean())
no90=no90.reset_index()
no90=no90.iloc[:-2]
pdel=no90.SeriousDlqin2yrs == 0
no90=no90.drop((no90.SeriousDlqin2yrs[pdel]).index)
no90=no90.sort_values(['SeriousDlqin2yrs'], ascending=[True])
ax = sns.barplot(x='NumberOfTimes90DaysLate', y='SeriousDlqin2yrs', data=no90)
plt.title('Mean of number of times more than 90 days late')
plt.xlabel('Number of times more than 90 days late')
plt.ylabel('Serious delinquence in 2 years')
plt.show()

#Number real estate
nreale=pd.DataFrame(creditos.groupby(['NumberRealEstateLoansOrLines'])['SeriousDlqin2yrs'].mean())
nreale=nreale.reset_index()
nreale=nreale.iloc[:-3]
pdel=nreale.SeriousDlqin2yrs == 0
nreale=nreale.drop((nreale.SeriousDlqin2yrs[pdel]).index)
nreale=nreale.sort_values(['NumberRealEstateLoansOrLines'], ascending=[True])
ax = sns.barplot(x='NumberRealEstateLoansOrLines', y='SeriousDlqin2yrs', data=nreale)
plt.title('Mean of number of real estate, loans or lines')
plt.xlabel('Number of real estate, loans or lines')
plt.ylabel('Serious delinquence in 2 years')
plt.show()

#Number of times between 60 and 90
n6090=pd.DataFrame(creditos.groupby(['NumberOfTime60-89DaysPastDueNotWorse'])['SeriousDlqin2yrs'].mean())
n6090=n6090.reset_index()
pdel=n6090.SeriousDlqin2yrs == 0
n6090=n6090.drop((n6090.SeriousDlqin2yrs[pdel]).index)
n6090=n6090.sort_values(['SeriousDlqin2yrs'], ascending=[True])
ax = sns.barplot(x='NumberOfTime60-89DaysPastDueNotWorse', y='SeriousDlqin2yrs', data=n6090)
plt.title('Mean of number of times between 60 and 89 days past but not worse')
plt.xlabel('Number of times between 60 and 89 days past')
plt.ylabel('Serious delinquence in 2 years')
plt.show()

#Number of dependants
depend=pd.DataFrame(creditos.groupby(['NumberOfDependents'])['SeriousDlqin2yrs'].mean())
depend=depend.reset_index()
depend=depend.iloc[:-4]
depend=depend.sort_values(['NumberOfDependents'], ascending=[True])
ax = sns.barplot(x='NumberOfDependents', y='SeriousDlqin2yrs', data=depend)
plt.title('Mean of number of dependants')
plt.xlabel('Number of dependants')
plt.ylabel('Serious delinquence in 2 years')
plt.show()

depend=pd.DataFrame(creditos.groupby(['age'])['NumberOfDependents'].mean())
depend=depend.reset_index()
depend=depend.iloc[:-5]
depend=depend.iloc[-77:]
depend=depend.sort_values(['NumberOfDependents'], ascending=[True])
ax = sns.barplot(x='age', y='NumberOfDependents', data=depend)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.title('Number of dependants by age')
plt.xlabel('Age')
plt.ylabel('Number of dependants')
plt.tight_layout()
plt.show()

#%%Modelo predictivo
#Separando train y test
X = creditos.iloc[:,2:12].values
y = creditos.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#%%Logistic regression, no tiene tunning
lreg = LogisticRegression(random_state = 0)
lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)
proba_lreg=pd.DataFrame({'prob':lreg.predict_proba(X_test)[:,1]})
#Matriz de confusion
cm_lreg = confusion_matrix(y_test, y_pred)
#Cross validation
scores_lreg=cross_val_score(lreg, X_train, y_train, cv=15)
inf_scor_lreg=pd.DataFrame([scores_lreg.mean(),scores_lreg.std()],index=['Mean','Standard dev'])
#plot coeficientes
coef_lreg=np.transpose(lreg.coef_)
x_eje=[1,2,3,4,5,6,7,8,9,10]
plt.scatter(x_eje,coef_lreg)
plt.title('Nivel de importancia de coeficientes')
plt.xlabel('Coeficientes')
plt.ylabel('Nivel de importancia')
plt.show()
#plot de depndences
coef_lreg=pd.DataFrame(coef_lreg)
coef_lreg=coef_lreg.sort_values([0], ascending=[False])
coef_lreg=coef_lreg[0:3]
coef_lreg=coef_lreg.index
## coeficiente 1 estoy haciendo probabilidad de malos
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_lreg[0]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_lreg)
intervalo=[0,20,40,60,98]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 1 Number of times between 30 and 60 days past but not worse')
## coeficiente 2
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_lreg[1]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_lreg)
intervalo=[0,0.5,1,2,54]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 2 Number of real estate, loans or lines')
## coeficiente 3
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_lreg[2]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_lreg)
intervalo=[0,20,40,60,98]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 3 Number of times more than 90 days late')
#%%Random Forest
roc_auc_df_rf = pd.DataFrame(columns=('n_features', 'n_trees', 'auc_fit'))
i=0
for n_features in np.arange(2,11,1):
 for n_tree in np.arange(50,151,50):
   i=i+1 
   fref = RandomForestClassifier(random_state=0, n_estimators = n_tree, max_features = n_features)
   #aqui tengo que hacer el early stoppinggg
   fref.fit(X_train, y_train)
   y_pred_rm_forest = fref.predict(X_test)
   # get roc/auc info
   Y_score = fref.predict_proba(X_test)[:,1]
   fpr = dict()
   tpr = dict()
   fpr, tpr, _ = roc_curve(y_test, Y_score)
   auc_fit = auc(fpr, tpr)
   
   roc_auc_df_rf.loc[i]=[n_features,n_tree,auc_fit]
   

a=np.transpose(np.reshape(roc_auc_df_rf.iloc[:,2],(9,3)))  
#El mejor es 2 variables con 150 arboles linea 1
plt.style.use('seaborn-pastel') 
eje_x=roc_auc_df_rf.n_trees[0:3]
plt.plot(eje_x,a[:,0],eje_x,a[:,1],eje_x,a[:,2],eje_x,a[:,3],eje_x,a[:,4],eje_x,a[:,5],eje_x,a[:,6],eje_x,a[:,7],eje_x,a[:,8])
plt.legend(["Variable x 2", "Variable x 3","Variable x 4", "Variable x 5","Variable x 6", "Variable x 7","Variable x 8", "Variable x 9","Variable x 10"])
plt.title('AUCROC vs Num. de árboles')
plt.xlabel('Número de árboles')
plt.ylabel('AUCROC')
plt.show()

#Random forest ideal
fref = RandomForestClassifier(random_state=0, n_estimators = 150, max_features = 2)
fref.fit(X_train, y_train)
y_pred_rm_forest = fref.predict(X_test)
proba_fref=pd.DataFrame({'prob':fref.predict_proba(X_test)[:,1]})
#Matriz de confusion
cm_fref = confusion_matrix(y_test, y_pred_rm_forest)
#Cross validation
scores_fref=cross_val_score(fref, X_train, y_train, cv=15)
inf_scor_fref=pd.DataFrame([scores_fref.mean(),scores_fref.std()],index=['Mean','Standard dev'])
#plot coeficientes
coef_fref=np.transpose(fref.feature_importances_)
x_eje=[1,2,3,4,5,6,7,8,9,10]
plt.scatter(x_eje,coef_fref)
plt.title('Nivel de importancia de coeficientes')
plt.xlabel('Coeficientes')
plt.ylabel('Nivel de importancia')
plt.show()
#plot de depndences
coef_fref=pd.DataFrame(coef_fref)
coef_fref=coef_fref.sort_values([0], ascending=[False])
coef_fref=coef_fref[0:3]
coef_fref=coef_fref.index
## coeficiente 1 estoy haciendo probabilidad de malos
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_fref[0]])
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_fref)
intervalo=[0,0.03508,0.1772,0.5794,1]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
plt.plot(intervalo[-4:],y_r_p.real,intervalo[-4:],y_r_p_p)
plt.show()
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 1 Revolving utilization of unsecured lines')
## coeficiente 2
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_fref[1]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_fref)
intervalo=[0,0.1433,0.2960,0.4825,1]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 2 Debt ratio')

## coeficiente 3
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_fref[2]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_fref)
intervalo=[0,3400,5400,8249,3008750]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 3 Monthly income')



#%%XGBoost
#Subsample
roc_auc_df_subs = pd.DataFrame(columns=('subsample', 'auc_fit'))
i=0
X_trainVal, X_testVal, y_trainVal, y_testVal = train_test_split(X_train, y_train, test_size = 1/3, random_state = 0)

for subsample in np.arange(0.5,1,0.05):
   i=i+1
   xgb = XGBClassifier(subsample=subsample,early_stopping_rounds=100)
   xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc",
           eval_set=[(X_trainVal, y_trainVal), (X_testVal, y_testVal)], verbose=100)
   y_pred_rm_xgb = xgb.predict(X_test)
   # get roc/auc info
   Y_score = xgb.predict_proba(X_test)[:,1]
   fpr = dict()
   tpr = dict()
   fpr, tpr, _ = roc_curve(y_test, Y_score)
   auc_fit = auc(fpr, tpr)
   
   roc_auc_df_subs.loc[i]=[subsample,auc_fit]
   
#El mejor es un subsample de 1
roc_auc_df_subs=roc_auc_df_subs.drop_duplicates()
plt.style.use('seaborn-pastel') 
plt.plot(roc_auc_df_subs.subsample,roc_auc_df_subs.auc_fit)
plt.title('AUCROC vs Subsample')
plt.xlabel('Número de subsample')
plt.ylabel('AUCROC')
plt.show()

#Learning rate
roc_auc_df_lr = pd.DataFrame(columns=('learning_rate', 'auc_fit'))
i=0
for learning_rate in np.arange(0.01,0.26,0.04):
   i=i+1
   xgb = XGBClassifier(learning_rate=learning_rate,early_stopping_rounds=100)
   xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc",
           eval_set=[(X_trainVal, y_trainVal), (X_testVal, y_testVal)], verbose=100)
   y_pred_rm_xgb = xgb.predict(X_test)
   # get roc/auc info
   Y_score = xgb.predict_proba(X_test)[:,1]
   fpr = dict()
   tpr = dict()
   fpr, tpr, _ = roc_curve(y_test, Y_score)
   auc_fit = auc(fpr, tpr)
   
   roc_auc_df_lr.loc[i]=[learning_rate,auc_fit]

#El mejor es un learning rate de 0.13
plt.style.use('seaborn-pastel') 
plt.plot(roc_auc_df_lr.learning_rate,roc_auc_df_lr.auc_fit)
plt.title('AUCROC vs Num. de learning rate')
plt.xlabel('Número de learning rate')
plt.ylabel('AUCROC')
plt.show()

#Max depth
roc_auc_df_md = pd.DataFrame(columns=('learning_rate', 'auc_fit'))
trees=pd.DataFrame()
i=0
for max_depth in np.arange(3,11,1):
   i=i+1
   xgb = XGBClassifier(n_estimators=1000,max_depth=max_depth)
   xgb.fit(X_train, y_train, early_stopping_rounds=150, eval_metric="auc",
           eval_set=[(X_trainVal, y_trainVal), (X_testVal, y_testVal)], verbose=100)
   y_pred_rm_xgb = xgb.predict(X_test)
   # get roc/auc info
   Y_score = xgb.predict_proba(X_test)[:,1]
   fpr = dict()
   tpr = dict()
   fpr, tpr, _ = roc_curve(y_test, Y_score)
   auc_fit = auc(fpr, tpr)
   trees[i]=[xgb.best_iteration]
   roc_auc_df_md.loc[i]=[max_depth,auc_fit]

#El mejor es un depth rate de 4
po=pd.DataFrame(np.arange(3,11,1)) 
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.style.use('seaborn-pastel') 
ax1.plot(roc_auc_df_md.learning_rate,roc_auc_df_md.auc_fit)
ax2 = ax1.twinx()
ax2.plot(po,np.transpose(trees), color='r')
plt.title('AUCROC vs Num. de maxdepth, Num. de iteraciones')
plt.xlabel('Número de maxdepth')
ax1.set_ylabel('AUCROC')
ax2.set_ylabel('Iteraciones')
plt.show()

#xgboost ideal
xgb = XGBClassifier(max_depth=4,learning_rate=0.13,subsample=1)
xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc",
        eval_set=[(X_trainVal, y_trainVal), (X_testVal, y_testVal)], verbose=100)
y_pred_rm_xgb = xgb.predict(X_test)
proba_xgb=pd.DataFrame({'prob':xgb.predict_proba(X_test)[:,1]})
#Matriz de confusion
cm_xgb = confusion_matrix(y_test, y_pred_rm_xgb)
#Cross validation
scores_xgb=cross_val_score(xgb, X_train, y_train, cv=15)
inf_scor_xgb=pd.DataFrame([scores_xgb.mean(),scores_xgb.std()],index=['Mean','Standard dev'])
#plot coeficientes
coef_xgb=np.transpose(xgb.feature_importances_)
x_eje=[1,2,3,4,5,6,7,8,9,10]
plt.scatter(x_eje,coef_xgb)
plt.title('Nivel de importancia de coeficientes')
plt.xlabel('Coeficientes')
plt.ylabel('Nivel de importancia')
plt.show()
#plot de depndences
coef_xgb=pd.DataFrame(coef_xgb)
coef_xgb=coef_xgb.sort_values([0], ascending=[False])
coef_xgb=coef_xgb[0:3]
coef_xgb=coef_xgb.index
## coeficiente 1 estoy haciendo probabilidad de malos
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_xgb[0]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_xgb)
intervalo=[0,0.03508,0.1772,0.5794,1]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 1 Revolving utilization of unsecured lines')
## coeficiente 2
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_xgb[0]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_xgb)
intervalo=[0,0.1433,0.2960,0.4825,1]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 2 Debt ratio')

## coeficiente 3
temp1=pd.DataFrame(pd.DataFrame(X_test).iloc[:,coef_xgb[2]])
temp1.columns=[0]
y_coef=pd.DataFrame({'y':y_test})
top1=temp1.join(y_coef).join(proba_xgb)
intervalo=[0,3400,5400,8249,3008750]
y_r_p=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).sum()
y_r_p_p=top1['prob'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).mean()
y_r_p_cuent=top1['y'].groupby(pd.cut(temp1[0], [intervalo[0],intervalo[1], intervalo[2],intervalo[3],intervalo[4]])).count()
y_r_p=pd.DataFrame(y_r_p/y_r_p_cuent)
y_r_p=y_r_p.join(y_r_p_p)
y_r_p.columns=['real','prom']
y_r_p=y_r_p.reset_index()
y_r_p.columns=['index','real','prom']
fig, ax1 = plt.subplots(figsize=(6.5, 6.5))
tidy = (
    y_r_p.set_index('index')
      .stack()  # un-pivots the data 
      .reset_index()  # moves all data out of the index
      .rename(columns={'level_1': 'Variable', 0: 'Value'})
)
fig1=sns.barplot(x='index', y='Value', hue='Variable', data=tidy, ax=ax1)
fig1.set(xlabel='Bins', ylabel='% probabilidad de impago')
plt.title('Coeficiente 3 Monthly income')

#grafica final de 3 aucroc
# overall accuracy
acc_lreg = lreg.score(X_test,y_test)
# get roc/auc info
Y_score_lreg = lreg.predict_proba(X_test)[:,1]
fpr_lreg = dict()
tpr_lreg = dict()
fpr_lreg, tpr_lreg, _ = roc_curve(y_test, Y_score_lreg)

acc_fref = fref.score(X_test,y_test)
# get roc/auc info
Y_score_fref = fref.predict_proba(X_test)[:,1]
fpr_fref = dict()
tpr_fref = dict()
fpr_fref, tpr_fref, _ = roc_curve(y_test, Y_score_fref)

acc_xgb = xgb.score(X_test,y_test)
# get roc/auc info
Y_score_xgb = xgb.predict_proba(X_test)[:,1]
fpr_xgb = dict()
tpr_xgb = dict()
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, Y_score_xgb)

roc_auc = dict()
roc_auc = auc(fpr_lreg, tpr_lreg)
roc_auc=np.append(roc_auc,auc(fpr_fref,tpr_fref))
roc_auc=np.append(roc_auc,auc(fpr_xgb,tpr_xgb))

plt.style.use('seaborn-pastel')
# make the plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.title('AUCROC 3 models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr_lreg, tpr_lreg,fpr_fref,tpr_fref,fpr_xgb,tpr_xgb, label='AUC = {0}'.format(roc_auc))        
#plt.legend(loc="lower right", shadow=True, fancybox =True) 
plt.show()






