import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from streamlit_chat import message

try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('wordnet')


# Load model and data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('newintents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Define functions for the chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
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
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit UI
st.set_page_config(page_title="CodeSpazeBot", layout="wide")

# Initialize session state for pages
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Home Page
if st.session_state.page == "home":
    with st.sidebar:
        st.image("logo.png", width=250)
        st.title("CodeSpazeBot 🤖")
        st.write("Your friendly chatbot, CodeSpaze Assistant, is here to guide you through the exciting world of CodeSpaze. Ask me anything about our internships, programs, campus life, or anything else related to CodeSpaze. I'll do my best to provide clear and informative answers. Whether you're a prospective student, current student, or alumni, CodeSpaze Assistant is your go-to resource for learning and exploring everything CodeSpaze has to offer. 🌟")

    st.title("Welcome to CodeSpaze! 🎉")
    st.write("CodeSpazeBot is ready to answer your questions about CodeSpaze. Ask away!")

    if st.button("Use CodeSpazeBot"):
        st.session_state.page = "chatbot"

# Chatbot Page
elif st.session_state.page == "chatbot":
    st.title("Chatbot Interface 🤖")
    response_container = st.container()
    textcontainer = st.container()

    # Initialize session state for responses and requests
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    with textcontainer:
        query = st.text_input("You: ")
        if query:
            # Add user query to requests
            st.session_state.requests.append(query)

            # Generate response
            ints = predict_class(query)
            res = get_response(ints, intents)

            st.session_state.responses.append(res)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
