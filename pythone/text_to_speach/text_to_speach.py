"""Entry point"""
import datetime

#import speech_recognition as sr
import pyttsx3
"""""
import wikipedia
import webbrowser
import os
import pyjokes
"""""

#voices = engine.getProperty('voices') # getting details of current voice
#engine.setProperty('voice', voices[0].id)

def speak(text:str):
    """_summary_
    main
    """
    print(f"Assistant: {text}")
    
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    

def greeting():
    """_summary_
    greeting
    """
    hour = int(datetime.datetime.now().hour)
    if hour < 12:
        speak("Good Morning!")
    elif hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your voice assistant. How can I help you today?")

def run_assistant():
    """_summary_
    """
    greeting()
    while True:
        query = take_command()
        default(query)       
def default(query:str):
    """_summary_

    Args:
        query (_type_): _description_
    """
    speak("I don't know what you mean by "+query)
def take_command():
    """_summary_
    Returns:
        _type_: _description_
    """
    return input("You (type your command): ").lower()

speak('llave')