# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:15:09 2020

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns

#importing the data
df=pd.read_csv("raw.txt",delimiter="\t")

#seeing the properties
rows,cols=df.shape
df.describe()
df.head()

#checking for nulls
df.isnull().sum()

#preprocessing
#feature engineering
df["Strike_Rate"]=(df.Runs/df.Balls)*100
df["Balls_without_boundaries"]=(df["Balls"]-(df["Fours"]+df["Sixes"]))
df["Runs_ran"]=df["Runs"]-(df["Fours"]*4+df["Sixes"]*6)
df["Boundary_runs"]=df["Fours"]*4+df["Sixes"]*6


#removing spaces 
def strip_space_at_loc(col_num,index_of_space_only_if_zero):
    for i in range(rows):
        if(df.iloc[i,col_num][index_of_space_only_if_zero]==" "):
            #print(df.iloc[i,0])
            df.iloc[i,col_num]=df.iloc[i,col_num][index_of_space_only_if_zero+1:]
            print(df.iloc[i,0])

strip_space_at_loc(0,0)

def create_list_of_uniq(data,col_name):
    #print(col_name)
    #print(type(col_name))
    col_name=[i for i in data.loc[:,col_name].unique()]
    return col_name

opp=create_list_of_uniq(df,"Opp")
players=create_list_of_uniq(df,"Player")
result=create_list_of_uniq(df,"Result")

#individual player scores
runs=[]
for i in players:
    runs.append(df.loc[df["Player"]==i,"Runs_ran"].sum())
#runs=pd.DataFrame(runs,columns=["runs"])
#print(type(runs[1][0]))
#individual boundary scores
b_runs=[]
for i in players:
    b_runs.append(df.loc[df["Player"]==i,"Boundary_runs"].sum())
#b_runs=pd.DataFrame(b_runs,columns=["b_runs"])

#creating new data frame
df2=pd.DataFrame(data=[players,runs,b_runs])
df2=df2.transpose()
df2=df2.rename(columns={0:"Players",1:"Runs",2:"B_runs"})


def power_hitters(player_name):
   run=df2.loc[df2["Players"]==player_name,"Runs"].max()
   b_run=df2.loc[df2["Players"]==player_name,"B_runs"].max()
   df2.loc[df2["Players"]==player_name,"Total_runs"]=df.loc[df["Player"]==player_name,"Runs"].sum()
   if run==0 & b_run==0:
       print("{} did not play".format(player_name))
   elif run>b_run:
       print("{} is not power hitter".format(player_name))
   else:
       print("{} is a power hitter".format(player_name))
       

#seeing all the power hitters
for i in players:
    power_hitters(i)

df2=df2.sort_values("Total_runs",ascending=False)

from sklearn.preprocessing import LabelEncoder
df3=df2.copy()
na_enc=LabelEncoder()
df3.Players=na_enc.fit_transform(df3.Players)
print(na_enc.classes_)

#plots
plt.bar(x=df3.Players,height=df3.Runs,color="blue",alpha=1)
plt.bar(x=df3.Players,height=df3.B_runs,color="red",alpha=0.6)
plt.label("Normal runs and boundary runs")
plt.xlabel("Players")
plt.ylabel("Runs")
plt.show()
#orange shows the players gets most of his runs from boundaries than he takes runs

#best players on the team according to runs
for i in na_enc.inverse_transform([3,4,5,6]):
    print("{} is one of the best players".format(i))

#to run predictive analysis
df_pred=df.copy()
df_pred=df_pred.rename(columns={"Balls_without_boundaries":"bwb","Boundary_runs":"b_runs"})
df_pred=df_pred.drop(["b_runs","Runs_ran","bwb","Strike_Rate","Sixes","Fours"],axis=1)

#encoding results
res_enc=LabelEncoder()
df_pred.Result=res_enc.fit_transform(df_pred.Result)
nam_enc=LabelEncoder()
df_pred.Player=nam_enc.fit_transform(df_pred.Player)
opp_enc=LabelEncoder()
df_pred.Opp=opp_enc.fit_transform(df_pred.Opp)

#finding the correlation
sns.heatmap(df_pred.corr(),annot=True)

#encoding players and opp teams
from sklearn.preprocessing import OneHotEncoder
nam_hot_enc=OneHotEncoder(categorical_features=[0])
df_pred=nam_hot_enc.fit_transform(df_pred).toarray()
opp_hot_enc=OneHotEncoder(categorical_features=[19])
df_pred=opp_hot_enc.fit_transform(df_pred).toarray()

#lets consider it as classification
#lets apply random forest calssifier
df_pred=pd.DataFrame(df_pred)
rows_pred,cols_pred=df_pred.shape
x=df_pred.drop([25],axis=1)
y=df_pred[25]
y=pd.DataFrame(y)

#splitting test and train sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

#fitting to random forest classifier algorithm
from sklearn.ensemble import RandomForestClassifier
rf_cls=RandomForestClassifier(criterion="entropy",random_state=101)
rf_cls.fit(x_train,y_train)

#predicting the values for x_test set
y_pred=rf_cls.predict(x_test)

#creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

#need more data since there is less correlation bet ind and dep var
#should do more featue engineering









