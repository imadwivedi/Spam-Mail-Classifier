import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


df=pd.read_csv('smsspam.txt',sep='\t',names=['Status','Message'])
#print df.head()
#print len(df) #5572

#print len(df[df.Status=='spam']) #747

df.loc[df['Status']=='ham','Status']=1 #df['Status'] = df.Status.map({'ham':0, 'spam':1})
df.loc[df['Status']=='spam','Status']=0

#print df.head()

df_x=df['Message']
df_y=df['Status']

x_train,x_test,y_train,y_test= train_test_split(df_x,df_y, test_size=0.2,random_state=4) #Training and testing data

y_train=y_train.astype('int') #Converting y_train to be in Integer earlier it was in String
y_test=y_test.astype('int')


#cv=TfidfVectorizer(min_df=1,stop_words='english') #Can use CountVectorizer() as well
cv=CountVectorizer()

cv.fit(x_train)
XtrainCv=cv.transform(x_train) #Converting data to values to text
XtestCv=cv.transform(x_test)

#Building models Naive Bayes
mnb=MultinomialNB()
mnb.fit(XtrainCv, y_train)
pred=mnb.predict(XtestCv)
print 'Accuracy of Naive Bayes Classifier is: '+str(accuracy_score(y_test,pred))

#Decision Tree
clf=DecisionTreeClassifier()
clf.fit(XtrainCv, y_train)
pred1=clf.predict(XtestCv)
print 'Accuracy of decision tree classifier is: '+str(accuracy_score(y_test,pred1))

#SVM Classifier
clf1=svm.SVC()
clf1.fit(XtrainCv, y_train)
pred2=clf.predict(XtestCv)
print 'Accuracy of SVM classifier is: '+str(accuracy_score(y_test,pred2))

print x_test.iloc[1]+ " Label for this is"
label_final=pred[1]
if(label_final==1):
    l="Ham"
else:
    l="Spam"
print l

print x_test.iloc[3]+ " Label for this is"
label_final=pred[3]
if(label_final==1):
    l="Ham"
else:
    l="Spam"
print l
print "Confusion Matrix is of Naive Bayes is "
print confusion_matrix(y_test,pred)
