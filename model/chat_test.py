import gradio as gr
import os
import google.generativeai as genai

def gemini_chat(message, history):
    response = chat.send_message(message)
    return response.text

API_KEY = os.getenv('AIzaSyBwU6NMo3wjRAewk-fgJxS_Mp763cqmv8Q')
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat()

iface = gr.ChatInterface(
    fn=gemini_chat, 
    title='Gemini Chat',
    chatbot=gr.Chatbot(height=1000),
    textbox=gr.Textbox(
        placeholder="Message Gemini",
        scale=7
    ),
    retry_btn=None,
    undo_btn=None,
    clear_btn=None
)

if __name__ == '__main__':
    iface.launch(share=True)