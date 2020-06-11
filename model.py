import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('train.csv')
#removing id column
data=data.drop('id',axis=1)
#removing stop words and punctuations
string.punctuation
nltk.download('stopwords')
stopword=nltk.corpus.stopwords.words('english')
def remove_punct(text):
  text_nopunct=''.join([char for char in text if char not in string.punctuation])
  return [word for word in text_nopunct.split() if word.lower() not in stopword]
x=data['text'].apply(remove_punct)
ps=nltk.PorterStemmer()
def stemming(text):
  text=[ps.stem(word)for word in text]
  return ' '.join(text)
x=x.apply(stemming)
#vectrorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)
print(vectorizer.get_feature_names())
y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=0)

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))
vec_file = 'vectorizer.pickle'
# pickle.dump(vectorizer, open(vec_file, 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
vect=pickle.load(open('vectorizer.pickle','rb'))