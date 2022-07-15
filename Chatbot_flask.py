#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask_restful import Api, Resource
from flask_cors import CORS


# In[2]:


import random
import json
import pickle
import numpy as np


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer


# In[4]:


from tensorflow.keras.models import load_model


# In[5]:


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents/intents.json').read())


# In[6]:


words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')


# In[7]:


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# In[8]:


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word ==w:
                bag[i] = 1
    return np.array(bag)


# In[9]:


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list


# In[10]:


def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# In[11]:


def chat_bot_response(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res


# In[ ]:


app = Flask(__name__)
api = Api(app)


#setting up my chatbotwebsite as primary origin so i can get access from this rest api
cors = CORS(app, resources={r"/*": {"https://hiredavidkimball.com/chatbot"}})

class Chatbot(Resource):
    def get(self, message):
        return chat_bot_response(message)
    
api.add_resource(Chatbot, "/chatbot/<message>")

if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:





# In[ ]:





# In[ ]:




