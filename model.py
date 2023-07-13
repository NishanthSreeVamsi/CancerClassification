import pandas as pd
import pandas as pd
import numpy as np
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("Data/data_Original.csv")
label = pd.read_csv("Data/labels.csv")
newdf = pd.merge(label,df)

df_cat_data = newdf
df_cat_data['Class'] = df_cat_data['Class'].map({'PRAD': 1, 'LUAD': 2, 'BRCA': 3, 'KIRC': 4, 'COAD': 5}) 
df_cat_data = df_cat_data.drop(['Unnamed: 0'],axis=1)


df_lda = newdf.drop(['Unnamed: 0'], axis=1)
df_lda = df_lda.drop(['Class'], axis=1)
x_lda = df_lda
y_lda = newdf['Class']

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
x_r2 = lda.fit(x_lda,y_lda).transform(x_lda)
x_r3 = pd.DataFrame(data=x_r2)
x_r3['y']=y_lda

ml_x = x_lda
ml_y = y_lda


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ml_x,ml_y,test_size=0.30,random_state=30)
from sklearn import tree

dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(x_train,y_train)
pickle.dump(dt_clf , open('model.pkl' , 'wb'))