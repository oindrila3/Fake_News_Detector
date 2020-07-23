from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 

application = app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	df= pd.read_csv("./Data/train.csv", encoding= 'unicode_escape')
	## Droping the "id" column as it does not provide any useful information

	df.drop(['id'],axis=1,inplace=True)

	## Filling the NA values with the space
	df=df.fillna(' ')

	### Creating a new column
	df['total']=df['title']+' '+df['author']+df['text']

	## Dropping the additional columns
	df = df[['total','label']]

	df=df.head(1000)
	
	stop = stopwords.words('english')
	lemmatizer = WordNetLemmatizer()


	def clean(doc):
	    tokenized=nltk.word_tokenize(doc) # tokenize
	    lowercase=[i.lower() for i in tokenized] # convert to lower case
	    stop_free=[i for i in lowercase if i not in stop] # get rid of stop words
	    punc_free=[i for i in stop_free if not i in string.punctuation] # get rid of the punctuations
	    normalized=[lemmatizer.lemmatize(i,'v') for i in punc_free] # lemmatization of each words
	    return punc_free


	df['total']=[" ".join(clean(i)) for i in df['total'].values]
	print(df.total)

	vectorizer = TfidfVectorizer(stop_words='english',
	                            analyzer='word',
	                            ngram_range=(1, 2),
	                            max_features=30
	                            )
	feature_vec = vectorizer.fit_transform(df.total)
	tfidf_matrix = feature_vec.toarray()

	x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, list(df['label']),test_size=0.30, random_state=0)
	logisticRegr = LogisticRegression()
	logisticRegr.fit(x_train, y_train)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		cleaned_data=[" ".join(clean(i)) for i in data]
		vect = vectorizer.transform(cleaned_data)
		predicted_matrix = vect.toarray()
		my_prediction = logisticRegr.predict(predicted_matrix)
	return render_template('results.html', prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)