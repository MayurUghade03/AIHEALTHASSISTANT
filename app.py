import streamlit as st
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from googletrans import Translator
import nltk
import time
import torch
import openai
import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Google Translator
translator = Translator()

# Set OpenAI API key (ensure you replace this with your actual OpenAI API key securely)
openai.api_key = os.getenv("put_your_api_key")  # Securely set API key via environment variable

# SQLite database setup for storing information
def create_database():
    conn = sqlite3.connect('healthcare_chatbot.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                      (id INTEGER PRIMARY KEY, user_input TEXT, assistant_response TEXT)''')
    conn.commit()
    conn.close()

def store_conversation(user_input, assistant_response):
    conn = sqlite3.connect('healthcare_chatbot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_input, assistant_response) VALUES (?, ?)",
                   (user_input, assistant_response))
    conn.commit()
    conn.close()

# Load the question answering model (T5)
qa_model = T5ForConditionalGeneration.from_pretrained("t5-base")
qa_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Healthcare chatbot logic with more detailed logic and AI/ML processing
def healthcare_chatbot(user_input, lang='en'):
    user_input = user_input.lower()

    # Enhanced healthcare-specific responses
    if "symptom" in user_input:
        return "I recommend consulting a healthcare professional for accurate advice on symptoms. They can guide you through any necessary tests or treatments."
    elif "appointment" in user_input:
        return "Please let me know the type of doctor you're looking to schedule an appointment with, and I can help you find one in your area."
    elif "medication" in user_input:
        return "Please follow your doctor's prescription for medication. If you have any concerns about the medication, please reach out to your doctor immediately."
    elif "doctor" in user_input:
        return "If you're seeking advice from a doctor, it's best to book an appointment with a healthcare provider who can offer personalized care."
    elif "emergency" in user_input:
        return "For medical emergencies, please call emergency services immediately or visit your nearest hospital. Time is critical."
    elif "health" in user_input:
        return ("Maintaining a healthy lifestyle is crucial. Regular exercise, a balanced diet, staying hydrated, "
                "and getting enough sleep are key elements. Additionally, stress management and mental health are equally important.")
    elif "covid" in user_input:
        return ("For information about COVID-19 symptoms, prevention, and treatment, please refer to the official guidelines from your local health authorities or the World Health Organization.")
    else:
        # For general or unknown queries, use T5 for Question Answering (QA)
        return get_answer_from_qa_model(user_input)

def get_answer_from_qa_model(question):
    # Prepare the question for the T5 model
    input_text = f"question: {question}  context: healthcare information."
    input_ids = qa_tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate an answer using the model
    output = qa_model.generate(input_ids)
    answer = qa_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Text similarity function using ML/AI to improve context recognition
def get_most_similar_response(user_input):
    # Load pre-existing conversations and calculate similarity with user input
    conn = sqlite3.connect('healthcare_chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations")
    conversations = cursor.fetchall()
    conn.close()
    
    if not conversations:
        return None

    corpus = [conv[1] for conv in conversations]  # Get all user inputs
    vectorizer = TfidfVectorizer().fit_transform(corpus + [user_input])  # Adding user input to the corpus
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    
    most_similar_idx = cosine_similarities.argmax()
    return conversations[most_similar_idx][2] if cosine_similarities[0][most_similar_idx] > 0.5 else None

def translate_to_english(user_input, lang):
    if lang != 'en':
        translated = translator.translate(user_input, src=lang, dest='en')
        return translated.text
    return user_input

def translate_response_to_user_language(response, lang):
    if lang != 'en':
        translated = translator.translate(response, src='en', dest=lang)
        return translated.text
    return response

def main():
    st.title("Healthcare Assistant Chatbot")
    language_option = st.selectbox("Select Language", ['English', 'Hindi', 'Marathi'])

    # Set the appropriate language code based on the selected option
    if language_option == 'Hindi':
        lang_code = 'hi'
    elif language_option == 'Marathi':
        lang_code = 'mr'
    else:
        lang_code = 'en'

    # Create a placeholder to display conversation
    chat_history = st.empty()

    # Initialize session state for storing conversation history
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    user_input = st.text_area(f"How can I assist you today? ({language_option})", height=150)
    
    if st.button("Submit"):
        if user_input:
            # Get similar response from previous conversations using AI/ML
            similar_response = get_most_similar_response(user_input)
            if similar_response:
                assistant_response = similar_response
            else:
                # Proceed with chatbot if no similar response found
                with st.spinner("Processing your query, please wait ...."):
                    # Translate input to English
                    translated_input = translate_to_english(user_input, lang_code)
                    assistant_response = healthcare_chatbot(translated_input, lang_code)
                
                # Store the conversation in the database
                store_conversation(user_input, assistant_response)
                
                # Translate response back to user's language
                final_response = translate_response_to_user_language(assistant_response, lang_code)
                st.session_state['conversation'].append(f"🧑🏻‍⚕️ Healthcare Assistant ({language_option}): {final_response}")

            # Display conversation
            st.session_state['conversation'].append(f"🧑🏻 User ({language_option}): {user_input}")
            chat_history.write('\n\n'.join(st.session_state['conversation']))
            
        else:
            st.warning("Please enter a message to get a response.")

    # Option to clear conversation
    if st.button("Clear Conversation"):
        st.session_state['conversation'] = []
        chat_history.empty()

# Execute the main function
if __name__ == "__main__":
    create_database()  # Ensure the database is created before running
    main()
