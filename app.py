import pickle
import string
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Ensure NLTK tokenization rules are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Recreate your preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    test = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message = ""
    
    if request.method == 'POST':
        message = request.form['message']
        
        # 1. Clean the text using the identical pipeline
        transformed_sms = transform_text(message)
        
        # 2. Convert to numbers
        vector_input = vectorizer.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Map the numerical result to text
        prediction = "Spam" if result == 1 else "Not Spam (Ham)"
            
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)