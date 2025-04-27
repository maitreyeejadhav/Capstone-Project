from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB



# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Preprocessing function
def preprocess_text(text):
    port_stem = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Load the datasets
fake_data_file = 'fake_news_sample_2.csv'
new_data_file = 'real_result_2.csv'

true_data = pd.read_csv(new_data_file)   # real news from web scraping
fake_data = pd.read_csv(fake_data_file)  # manually labeled fake news

# Combine datasets
all_data = pd.concat([fake_data, true_data])

# Shuffle rows with a fixed random state for reproducibility
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and description, and preprocess text
all_data['text'] = all_data['title'] + ' ' + all_data['description']
all_data['text'] = all_data['text'].apply(preprocess_text)

# Extract features and labels
X = all_data['text']
y = all_data['label']

# Vectorize text
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)


# vectorizer = TfidfVectorizer()   ## original code
X_transformed = vectorizer.fit_transform(X)

# Split data with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.5, random_state=42)

# Train SVM model
model = SVC(kernel='linear', C=10, class_weight='balanced')
# model = SVC(kernel='linear')  ## original code
model.fit(X_train, y_train)

# # # Train LR model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # # Train NB model
# model = MultinomialNB()
# model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Print results
print(f'Model retrained with accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')





# # Preprocessing function
# def preprocess_text(text):
#     port_stem = PorterStemmer()
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = text.lower().split()
#     text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
#     return ' '.join(text)

# fake_data_file = 'fake_news_sample.csv'   # fakes news data could be saved manually fo now 
# new_data_file = 'real_result.csv'
# true_data = pd.read_csv(new_data_file)   # this new file comes from the web scrapper, this only contains real news from news website 
# fake_data = pd.read_csv(fake_data_file)

# all_data=pd.concat([fake_data, true_data])     # combining fake and real news dataset 
# random_permutation = np.random.permutation(len(all_data))  #shuffles the rows for mixing sammples
# all_data= all_data.iloc[random_permutation]

# # Assuming dataset has 'title', 'description' and 'label' columns
# # label for fake nwes = 1
# # label for real news = 0
# all_data['text']=all_data['title']+' '+all_data['description']
# all_data['text'] = all_data['text'].apply(preprocess_text)
# X = all_data['text']
# y = all_data['label']

# vectorizer = TfidfVectorizer()
# X_transformed = vectorizer.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# model = SVC(kernel='linear')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='binary')  # or 'macro', 'micro', 'weighted' for multiclass
# recall = recall_score(y_test, y_pred, average='binary')
# f1 = f1_score(y_test, y_pred, average='binary')
# print(f'Model retrained with accuracy: {accuracy:.4f}')
# # Print results
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")

