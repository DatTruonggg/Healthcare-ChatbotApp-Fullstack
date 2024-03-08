import os
import google.generativeai as genai

API_KEY = os.getenv('AIzaSyAMs8UA8ujE4hT-Xed80KyC2rs9vmktoA8')

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat()
end_chat = False

try:
    while True:
        print()
        user_message = input('User: ')
        response = chat.send_message(user_message)
        print()
        print(f"Gemini: {response.text}")
        if end_chat: break
except KeyboardInterrupt:
    print('Ended chat with Keyboard Interrupt')
    end_chat = True