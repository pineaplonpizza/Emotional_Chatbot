# Emotional_Chatbot : Building Emotionally Intelligent Conversational AI

This project focuses on creating an emotionally aware chatbot that identifies user emotions and provides empathetic, contextually appropriate responses. Leveraging data from the GoEmotions dataset, the project fine-tunes a LLaMA-2-7b-Chat model to create a powerful, emotion-sensitive conversational agent.

Model and Dataset
  Model: LLaMA-2 Emotional Chatbot (https://huggingface.co/VaisakhKrishna/Llama-2-Emotional-ChatBot)
	
  Dataset: Emotional Sentiment Analysis (https://huggingface.co/datasets/VaisakhKrishna/Emotional_Sentiment_Analysis)

  
Overview
Steps Involved:

Dataset Creation:

Took 7,000 samples from the GoEmotions dataset, which provides text and the associated user emotion (28 different emotions).

Performed text preprocessing for consistency and cleanliness.

Generated responses tailored to the text and emotion using the Gemma2-9b-it model via Groq, ensuring relevance and empathy.


Model Fine-Tuning:

Used QLoRA (Quantized LoRA) to fine-tune the LLaMA-2-7b-Chat model on the prepared dataset.

Focused on improving the chatbot’s ability to understand emotions and provide matching responses.


Streamlit Application:

Built a Streamlit-based chatbot app that allows users to interact with the model.
The chatbot predicts the user's emotion and generates a corresponding empathetic response.


Final Uploads:

The fine-tuned model and processed dataset were uploaded to Hugging Face for public access:

Model: Emotional Chatbot Model (https://huggingface.co/VaisakhKrishna/Llama-2-Emotional-ChatBot)

Dataset: Emotional Sentiment Dataset (https://huggingface.co/datasets/VaisakhKrishna/Emotional_Sentiment_Analysis)

Files Included

dataset.ipynb: Script for dataset preparation, including text preprocessing and generating responses using Gemma2.

Model.ipynb: Code for fine-tuning the LLaMA-2-7b-Chat model using QLoRA.

app.py: The Streamlit app for interacting with the chatbot.

dataset.csv: Contains the following columns:

  text: User input.
	
  emotion: Emotion detected in the input.

  response: Tailored response based on emotion.
	
  formatted: Formatted text combining all the above, suitable for fine-tuning.
	
formatted.json: Preprocessed data containing only the formatted text column for direct use in fine-tuning.

requirement.txt: Contains all the required python libraries


Purpose

The goal of this project is to develop a chatbot that not only understands the content of user inputs but also their emotional undertone, enabling it to:

Provide empathetic and context-aware responses.
 
Assist in mental health applications, customer service, or personal assistant roles.


Limitations

Emotion Detection Accuracy: While the model can detect common emotions, nuanced or complex emotional states might not be accurately identified.

Response Generation: Generated responses may sometimes lack depth or appear overly generic.

Not a Substitute for Professional Support: The chatbot is not intended to provide medical, mental health, or crisis support.


Future Scope

Expanding the dataset to cover more nuanced emotions and scenarios.

Improving the response generation system for deeper contextual understanding.

Adding multilingual capabilities for broader accessibility.


To Run The Program:

Prerequisites

Install Python (version 3.8 or higher).

Install Dependencies: Make sure you have the required Python libraries installed. You can use the provided requirements.txt file 

  pip install -r requirements.txt
	
GPU Support: If you are using a GPU, ensure you have CUDA installed and configured.

Steps to Run

1. Clone the Repository
   
2. Download the project files to your local machine:

3. Run the Dataset Creation Script (Optional)

If you want to recreate the dataset:
  
	jupyter notebook dataset.ipynb

4. Run the Model Fine-Tuning Script (Optional)

To fine-tune the model using the prepared dataset:
  
	jupyter notebook model.ipynb

5. Start the Streamlit App

Run the Streamlit app to interact with the chatbot:
  
	streamlit run app.py

6. Open in Browser
Once the app starts, it will display a local URL (e.g., http://localhost:8501). Open this URL in your browser.
