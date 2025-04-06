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
from sklearn.metrics import accuracy_score

# Ensure stopwords are downloaded
nltk.download('stopwords')
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    port_stem = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# @app.route('/retrain', methods=['GET'])
# def retrain_model():
# 	link = request.args.get('link')
	
# 	# scrape this link in backend
# 	bs4.scrape somehow
	
# 	# save a .csv file from scraped data
# 	link_csv_file_path = '../data/link<date>.csv'
# 	retrain(link_csv_file_path)
	
# 	# restart server or model or something somehow. 
	
# 	return true if successfully done, else false


# Retrain SVM model
def retrain_svm_model(new_data_file):
    fake_data_file = '../data/fake.csv'   # fakes news data could be saved manually fo now 
    
    true_data = pd.read_csv(new_data_file)   # this new file comes from the web scrapper, this only contains real news from news website 
    fake_data = pd.read_csv(fake_data_file)
    
    all_data=pd.concat([fake_data, true_data])     # combining fake and real news dataset 
    random_permutation = np.random.permutation(len(all_data))  #shuffles the rows for mixing sammples
    all_data= all_data.iloc[random_permutation]

    # Assuming dataset has 'title', 'description' and 'label' columns
    # label for fake nwes = 1
    # label for real news = 0
    all_data['text']=all_data['title']+' '+all_data['description']
    all_data['text'] = all_data['text'].apply(preprocess_text)
    X = all_data['text']
    y = all_data['label']
    
    vectorizer = TfidfVectorizer()
    X_transformed = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model retrained with accuracy: {accuracy:.4f}')
    
    # Save the trained model and vectorizer
    joblib.dump(model, '../model/svm_model.pkl')
    joblib.dump(vectorizer, '../model/vectorizer.pkl')
    print('Model and vectorizer saved.')

from flask import Flask, request, jsonify

app = Flask(__name__)


# Predict using retrained model
@app.route('/predict' , methods=['GET'])
def predict_with_svm():
    model = '../model/svm_model_joblib.pkl'
    model = joblib.load(model)
    vectorizer = joblib.load('../model/vectorizer.pkl')
    text = request.args.get('text')
    processed_text = preprocess_text(text)
    input_vector = vectorizer.transform([processed_text])
    
    prediction = model.predict(input_vector)[0]
    if not text:
        return jsonify({"error": "No text provided"}), 400
    else:
        return jsonify({"prediction": str(prediction)})

if __name__ == '__main__':
    print("Starting Python Flask Server")
    app.run(debug=True)










# from flask import Flask, request, jsonify
# import pickle, joblib
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)

# # Load pre-trained model and vectorizer
# with open("Capstone project backend/model/svm_model_joblib", "rb") as model_file:
#     svm_model = joblib.load('svm_model_joblib')
# with open("vectorizer.pkl", "rb") as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json.get("text", "")
#     vectorized_text = vectorizer.transform([data]).toarray()
#     prediction = svm_model.predict(vectorized_text)[0]
#     return jsonify({"prediction": "Fake News" if prediction == 1 else "Real News"})

# if __name__ == '__main__':
#     print("Starting Python Flask Server")
#     app.run(debug=True)


