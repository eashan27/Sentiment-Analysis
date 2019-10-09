import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset= pd.read_csv('book1.csv',error_bad_lines=False,quoting=2,encoding='ISO-8859-1')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
from gensim.models import Word2Vec
corpus= []
tagged_corpus= []
#print(dataset['Liked'])
for i in range(0,1002):
     review =re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
     review=review.lower()
     review=review.split()
     #print(review)
     wordnet=WordNetLemmatizer()
     t1=nltk.pos_tag(review)
     s1=''
     for j in range(0,len(t1)):
         s1=s1+t1[j][0]+'/'+t1[j][1]+' '
     tagged_corpus.append(s1.split())
     review =[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
     review = ' '.join(review)
     corpus.append(review.split(" "))
#print(tagged_corpus)
model=Word2Vec(tagged_corpus,size=10, window=5, min_count=1, workers=4)
print(model.wv['slow/JJ'])
print(model.wv['this/DT'])
X=model[model.wv.vocab]
from sklearn.cluster import KMeans
wcss=[]

kmeans=KMeans(n_clusters=20,init='k-means++',max_iter=300,n_init=10,random_state=0)
y=kmeans.fit_predict(X)
wcss.append(kmeans.inertia_)
words=list(model.wv.vocab)
#for i,word in enumerate(words):
 #   print(word+":"+str(y[i]))
'''for i in range(0,2402):
    if y[i]==17:
        print(words[i])

'''

integer_encoded = [None] * 1002
for k in range(0,1002):
    values1=array(corpus[k])
    label_encoder = LabelEncoder()
    integer_encoded[k] = label_encoder.fit_transform(values1)
    print(integer_encoded[k])

from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Embedding, LSTM,SpatialDropout1D
from keras.utils.np_utils import to_categorical
embedding_layer = Embedding(input_dim=X.shape[0], output_dim=X.shape[1], weights=[X])

model1 = Sequential()
model1.add(embedding_layer)
model1.add(SpatialDropout1D(0.4))
model1.add(LSTM(X.shape[1],dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(X.shape[0]))   
model1.add(Activation('softmax'))
model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

from sklearn.model_selection import train_test_split
Z = pd.DataFrame(dataset['Review']).values
m=pd.DataFrame(Z)
Y = pd.DataFrame(dataset['Liked']).values

X_train, X_test, Y_train, Y_test = train_test_split(corpus,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

print(len(X_train[0]))
batch_size = 32
#model1.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)