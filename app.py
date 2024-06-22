import streamlit as st
import pickle

# Define the ChatbotModel class again (it should match the one used during training)
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ChatbotModel:
    def __init__(self, clf, vectorizer, intents):
        self.clf = clf
        self.vectorizer = vectorizer = vectorizer
        self.intents = intents

    def chatbot(self, input_text):
        input_text = self.vectorizer.transform([input_text])
        tag = self.clf.predict(input_text)[0]
        for intent in self.intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response

# Load the model from the pickle file
pickle_in = open('chat_bot.pkl', 'rb')
model = pickle.load(pickle_in)

# Initialize a global counter to keep track of the conversation
counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = model(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
