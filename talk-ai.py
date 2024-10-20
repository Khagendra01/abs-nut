import openai
import speech_recognition as sr
import pyttsx3

# Initialize OpenAI client
client = openai.OpenAI(api_key='sk-cHmTiunjcASker0H5Rch-LHYthetWv06_wt_Lg2HoKT3BlbkFJ7rI_1M5reSQfxxV0A5c28JUOGo4iMXE7tFUx8kuB4A')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

def listen_for_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return None

def speak_response(text):
    print("AI response:", text)
    engine.say(text)
    engine.runAndWait()

def chat_with_gpt4o():
    while True:
        speak_response("I am listening,")
        user_input = listen_for_speech()

        if user_input:
            if user_input.lower() == "exit":
                print("Exiting the program...")
                break
            response = client.chat.completions.create(
                model="gpt-4o",  # Replace with the actual model name when available
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150
            )
            ai_response = response.choices[0].message.content
            speak_response(ai_response)

if __name__ == "__main__":
    speak_response("Start speaking to chat with GPT-4o. Say 'exit' to end the conversation.")
    chat_with_gpt4o()