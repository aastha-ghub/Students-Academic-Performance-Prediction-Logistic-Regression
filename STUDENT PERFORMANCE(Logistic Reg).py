#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Students Academic Performance Prediction Case Study


# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

sns.set()


# In[29]:


data = pd.read_csv("/Users/Admin/Downloads/xAPI-Edu-Data.csv")


# # 1.Visualize just the categorical features individually to see what options are included and how each option fares when it comes to count(how many times it appears) and see what can be deduce from that?

# In[30]:


fig, axarr = plt.subplots(2,2,figsize = (10,10))
sns.countplot(x='Class', data= data, ax=axarr[0,0],order=['L','M','H'])
sns.countplot(x='gender', data= data, ax=axarr[0,1],order=['M','F'])
sns.countplot(x='StageID', data= data, ax=axarr[1,0])
sns.countplot(x='Semester', data= data, ax=axarr[1,1])


# In[31]:


fig,(axis1,axis2) = plt.subplots(2,1,figsize=(10,10))
sns.countplot(x='Topic', data= data, ax=axis1)
sns.countplot(x='NationalITy', data= data, ax=axis2)


# ANS: most of these countries are in the middle east(islamic states), perhaps this explains the gender disparity

# 

# # 2. Look at some categorical features in relation to each other, to see what insights could be possibly read?

# In[32]:


fig, axarr = plt.subplots(2,2,figsize = (10,10))
sns.countplot(x='gender',hue = 'Class', data= data, ax=axarr[0,0],order=['M','F'],hue_order=['L','M','H'])
sns.countplot(x='gender',hue = 'Relation', data= data, ax=axarr[0,1],order=['M','F'])
sns.countplot(x='gender',hue = 'StudentAbsenceDays', data= data, ax=axarr[1,0],order=['M','F'])
sns.countplot(x='gender',hue = 'ParentAnsweringSurvey', data= data, ax=axarr[1,1],order=['M','F'])


# In[33]:


fig, (axis1,axis2) = plt.subplots(2,1,figsize = (10,10))
sns.countplot(x='Topic', hue='gender',data= data, ax=axis1)
sns.countplot(x='NationalITy',hue='gender', data= data, ax=axis2)


# Ans :
# 
# .Girls seem to have performed better than boys
# .In the case of girls, mothers seem to be more interested in their education than fathers
# .Girls had much better attendance than boys
# .No apparent gender bias when it comes to subject/topic choices, we cannot conclude that girls performed better because they perhaps took less technical subjects
# .Gender disparity holds even at a country level. May just be as a result of the sampling

# # 3.Visualize categorical variables with numerical variables and give conclusions?
# 

# In[34]:


fig, axarr = plt.subplots(2,2,figsize = (10,10))
sns.barplot(x='Class',y='VisITedResources', data= data,order=['L','M','H'],ax=axarr[0,0])
sns.barplot(x='Class',y='AnnouncementsView', data= data,order=['L','M','H'],ax=axarr[0,1])
sns.barplot(x='Class',y='raisedhands', data= data,order=['L','M','H'],ax=axarr[1,0])
sns.barplot(x='Class',y='Discussion', data= data,order=['L','M','H'],ax=axarr[1,1])


# In[35]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize = (10,5))
sns.pointplot(x='Semester',y='VisITedResources',hue='gender' ,data= data,ax=axis1)
sns.pointplot(x='Semester',y='AnnouncementsView',hue='gender', data= data,ax=axis2)


# Ans :
# 
# As expected, those that participated more (higher counts in Discussion, raisedhands, AnnouncementViews, RaisedHands), performed better
# In the case of both visiting resources and viewing announcements, students were more vigilant in the second semester, perhaps that last minute need to boost your final grade

# In[36]:


ave_raisedhands = sum(data['raisedhands'])/len(data['raisedhands'])
ave_VisITedResources = sum(data['VisITedResources'])/len(data['VisITedResources'])
ave_AnnouncementsView = sum(data['AnnouncementsView'])/len(data['AnnouncementsView'])
unsuccess = data.loc[(data['raisedhands'] >= ave_raisedhands) & (data['VisITedResources'] >= ave_VisITedResources) & (data['AnnouncementsView'] >= ave_AnnouncementsView) &(data['Class']=='L')]
                     


# In[37]:


unsuccess


# # 4.From the above result, what are the factors that leads to get low grades of the students?

# In[38]:


data['numeric_class'] = [1 if data.loc[i,'Class'] == 'L' else 2 if data.loc[i,'Class'] =='M' else 3 for i in range(len(data))]


# In[39]:


grade_male_ave = sum(data[data.gender == 'M'].numeric_class)/float(len(data[data.gender == 'M']))
grade_female_ave = sum(data[data.gender == 'F'].numeric_class)/float(len(data[data.gender == 'F']))


# . Gender comparison cannot completely explain low grades

# In[42]:


#now lets look at nationality
nation =data.NationalITy.unique()
nation_grades_ave=  [sum(data[data.NationalITy == i].numeric_class)/float(len(data[data.NationalITy == i])) for i in nation]
ax=sns.barplot(x=nation,y=nation_grades_ave)
jordan_ave = sum(data[data.NationalITy == 'Jordan'].numeric_class)/float(len(data[data.NationalITy == 'Jordan']))
print('jordan average:' +str(jordan_ave))
plt.xticks(rotation=90)


# . As it can be seen in barplot jordan is seventh country with avg 2.09 so jordan has positive impact on these two students actually

# In[47]:


#lets look at relation with family members
relation =  data.Relation.unique()
relation_grades_ave=  [sum(data[data.Relation == i].numeric_class)/float(len(data[data.Relation == i])) for i in relation]
ax=sns.barplot(x=relation,y=relation_grades_ave)
plt.title('relation with father or mother affects success of students')


# .relation with mum has +ve effect on students

# In[45]:


#lets look at how many times the student particiate in discussion groups
discussion = data.Discussion
discussion_ave = sum(discussion)/len(discussion)
ax = sns.violinplot(y=discussion,split=True,inner='quart')
ax = sns.swarmplot(y=discussion,color='black')
ax = sns.swarmplot(y=unsuccess.Discussion,color='red')
plt.title('Discussion group particiption')


# In[49]:


absence_day = data.StudentAbsenceDays.unique()
absence_day_eve = [sum(data[data.StudentAbsenceDays == i].numeric_class)/float(len(data[data.StudentAbsenceDays == i])) for i in absence_day]
ax = sns.barplot(x=absence_day,y=absence_day_eve)
plt.title('Absence effect on success')


# Ans :
# 
# These two students are under the average of discussion (43). Therefore, not participating in discussion groups can be important reason to get low grades
# Their absence days are above seven which resulted in low grades

# # 5.Build classification model and present it's classification report?
# 

# In[50]:


data.head()


# In[51]:


data1 = data.drop('Class',axis = 1)
data_with_dummies = pd.get_dummies(data1,drop_first = True)


# In[52]:


data_with_dummies.head()


# In[53]:


Features = data_with_dummies.drop(['numeric_class'],axis = 1)
Target = data_with_dummies['numeric_class']


# In[54]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(Features)


# In[55]:


x = scaler.fit_transform(Features)


# In[56]:


x_train,x_test,y_train,y_test = train_test_split(x,Target,test_size=0.3,random_state=45)


# In[57]:


Logit_Model = LogisticRegression()
Logit_Model.fit(x_train,y_train) 


# In[58]:


Prediction = Logit_Model.predict(x_test)
Score = accuracy_score(y_test,Prediction)
Report = classification_report(y_test,Prediction)


# In[59]:


Prediction


# In[60]:


Score


# In[63]:


print(Report)


# In[ ]:




