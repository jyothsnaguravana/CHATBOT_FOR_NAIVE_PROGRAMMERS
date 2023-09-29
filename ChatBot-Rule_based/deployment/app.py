from flask import Flask, render_template, request
import numpy as np
import pickle
import json
import tensorflow as tf
import random
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

d = {}

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbotmodel.h5')
lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method=="POST":
        input_string = request.form.get("question","")
        res=""
        if input_string == "bye" or input_string == "Goodbye":
            ints = predict_class(input_string)
            res = get_response(ints, intents)
            if(res==""):
                res="I am not trained for that query"
            else:
                res=res+'\n'+"reload me for new chat"
        else:
            try:
                ints = predict_class(input_string)
                res = get_response(ints, intents)
            except:
                if(res==""): 
                    res="I am not trained for that query"
        d[input_string] = res 
        print(d)
    # return render_template('index.html',Bot=' {}'.format(res), Human=' {}'.format(input_string))
    return render_template('index.html',d = d)

if __name__=="__main__":
    app.run(debug = True)