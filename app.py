import streamlit as st # import the Streamlit library
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
from PIL import Image
import numpy as np
import pandas as pd
from time import sleep
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFAlbertForSequenceClassification
from transformers import pipeline
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re
from PyPDF2 import PdfReader
import textract
import os

import warnings
warnings.filterwarnings("ignore")

import huggingface_hub
huggingface_hub.login(token="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")

image = Image.open('logo-lg-red.png')
st.image(image, caption=None, width=150, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.title("✅MindWatch[Experiment]: A smart AI tool to detect mental disorders and suggest recommendations.")
st.markdown("\n")
st.markdown("_The application may experience a slight delay during the initial start-up as it requires loading the models. Your patience is greatly appreciated_")
st.markdown("\n")

image_1 = Image.open('new.png')
st.image(image_1, caption=None, width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("\n")

st.markdown("\n")
st.markdown("\n")
st.markdown("⚠️ Caution: \
This webtool is designed for texts resembling social media posts, clinical notes, or patient records. Please follow appropriate guidelines to ensure reliable results. Incoherent or inappropriate input may lead to misleading or nonsensical responses. For best results, use clear, concise, and relevant input text.")

# copyright message
st.markdown("---")
st.markdown("**_© University of Illinois at Chicago (UIC) Department of Psychiatry @2023_**")
st.markdown("---")
st.markdown("\n")

import requests
from PyPDF2 import PdfReader


text = ""


pdf_reader_1 = PdfReader('WHO.pdf')

for page in pdf_reader_1.pages:
    text += page.extract_text()


pdf_reader_2 = PdfReader('American_Pyschic_Association.pdf')

for page in pdf_reader_2.pages:
    text += page.extract_text()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1024,
    chunk_overlap  = 200,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text]) 

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")

broad_model_id = 'vineet1409/test_mindwatch_broad'
broad_model = TFAlbertForSequenceClassification.from_pretrained(broad_model_id)

mood_model_id = 'vineet1409/test_mood_mental_disorder'
mood_model = TFAlbertForSequenceClassification.from_pretrained(mood_model_id)

anxiety_model_id = 'vineet1409/test_anxiety_trauma_mental_disorder'
anxiety_model = TFAlbertForSequenceClassification.from_pretrained(anxiety_model_id)

neuro_model_id = 'vineet1409/test_neuro_mental_disorder'
neuro_model = TFAlbertForSequenceClassification.from_pretrained(neuro_model_id)



def get_predictions(text):
    encoded_input = tokenizer.encode_plus(
        text,
        padding='max_length',
        truncation=True,
        return_tensors="tf",
        max_length=128
    )

    output_broad = broad_model(encoded_input['input_ids'])
    prob_broad = tf.nn.softmax(output_broad.logits, axis=1)
    predicted_labels_broad = np.argmax(prob_broad, axis=1)

    class_names_broad = ["Anxiety and Trauma-Related Disorders (F40-F48)", "Mood Mental Disorders (F30-F39)",
                         "Neurodevelopmental Disorders (F80-F89)", "Psychotic Disorder- schizophrenia (F20)",
                         "normal mental state"]
    predicted_classes_broad = class_names_broad[predicted_labels_broad[0]]
    confidence_scores_broad = np.max(prob_broad, axis=1)

    # Handling specific disorder predictions:
    if predicted_classes_broad == "Anxiety and Trauma-Related Disorders (F40-F48)":
        output_specific = anxiety_model(encoded_input['input_ids'])
        class_names_specific = ["anxiety", "ocd", "post-traumatic-stress-disorder"]
    elif predicted_classes_broad == "Mood Mental Disorders (F30-F39)":
        output_specific = mood_model(encoded_input['input_ids'])
        class_names_specific = ["bipolar", "depression", "suicide-watch"]
    elif predicted_classes_broad == "Neurodevelopmental Disorders (F80-F89)":
        output_specific = neuro_model(encoded_input['input_ids'])
        class_names_specific = ["adhd", "aspergers", "autism"]
    else:
        return predicted_classes_broad, confidence_scores_broad[0], 'N/A', 0.0
    
    prob_specific = tf.nn.softmax(output_specific.logits, axis=1)
    predicted_labels_specific = np.argmax(prob_specific, axis=1)
    predicted_classes_specific = class_names_specific[predicted_labels_specific[0]]
    confidence_scores_specific = np.max(prob_specific, axis=1)

    return predicted_classes_broad, confidence_scores_broad[0], predicted_classes_specific, confidence_scores_specific[0]


# Streamlit interface
st.title("**_Mental Disorder Prediction_**")

# open-ai-key
st.markdown('**Enter the OpenAI-key to continue, refer: https://platform.openai.com/account/api-keys \
            to generate a key if you dont have one..!!**')
openai_key = st.text_input('OpenAI-key', type="password")
openai_key = str(openai_key)

if len(openai_key)!=0:
    OpenAI.api_key  = openai_key
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
    user_input = st.text_area("Enter text:")

    #create vector database
    db = FAISS.from_documents(chunks, embeddings)
    broad_class, broad_confidence, specific_class, specific_confidence = get_predictions(user_input)
    user_input = 'text: '+ str(user_input) + 'Diagnosis class: '+ specific_class
    docs = db.similarity_search(user_input)

    text = str(docs[0]) 
    # Remove unnecessary characters and newlines
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"-+", "", text)

    # Clean up the text
    cleaned_text = re.sub(r"\s+", " ", text)

    # Split the text into prescriptions
    prescriptions = re.findall(r"[A-Z][^.]+(?=\.)", cleaned_text)

    # Format the prescriptions as bullet points
    formatted_prescriptions = "\n- " + "\n- ".join(prescriptions)
    #gpt-3.5-turbo
    llm = OpenAI(model_name = 'gpt-3.5-turbo', temperature=0.1) # text-davinci-003
    # Chain 1: Generating a rephrased version of the user's question
    
    template = """
    Act as a medical mental health specialist and format the below prescription or diagnosis report for mental health issues. Properly draft the possible \
    solutions or diagnosis in bullets. Be creative in answering the same. \

    {formatted_prescriptions}

    Output format:
    # Possible Mental Disorder detected: (with ICD10 code)
    # General Symptoms: (in bullets)
    # Prescriptions or diagnosis:
    # Drugs and/or helpine numbers: (if required) \
    Suicide and Crisis Hotline: 988, Crisis Text Line: Text Hello to 741741, YouthLine: Text teen2teen to 839863, or call 1-877-968-8491, \

    \n\n"""

    prompt_template = PromptTemplate(input_variables=["formatted_prescriptions"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    overall_chain = SimpleSequentialChain(
            chains=[question_chain]
        )

    # Running all the chains on the user's question and displaying the final answer
    final_prescriptions = overall_chain.run(formatted_prescriptions)

    if st.button("Predict"):
        #broad_class, broad_confidence, specific_class, specific_confidence = get_predictions(user_input)
        result = f"1. Predictions from ALBERT- Custom LLM : \n \
            | Predicted Broad Class: {broad_class} with confidence: {broad_confidence*100:.2f}% | \
            Predicted Specific Class: {specific_class} with confidence: {specific_confidence*100:.2f}% | \
            \n\n 2. Results from RAG Pipeline : \n\n \
            {final_prescriptions}"
        
        st.success(result)
