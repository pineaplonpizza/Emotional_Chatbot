import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

def load_model():
    model_name = "VaisakhKrishna/Llama-2-Emotional-Chatbot"

    # # Configure bitsandbytes for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    # # Move the model to GPU explicitly (optional)
    model = model.to('cuda')


def clean_response(response):
    return response.split("[/INST]")[-1].strip()

#Streamlit App
st.title("Emotion Aware Chatbot")

#user input
user_input = st.text_input("Enter your text:", "")

if user_input:
    # Generate model response
    with st.spinner("Generating response..."):
        inputs = tokenizer(f"[INST] {user_input} [/INST]", return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_response = clean_response(raw_response)
    
    # Display the cleaned response
    st.subheader("Chatbot Response:")
    st.write(cleaned_response)