import speech_recognition as sr

def voiceToText():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        #clearning noise
        recognizer.adjust_for_ambient_noise(source,duration=1)
        recaudio = recognizer.listen(source)
    try:
        # creating to text using google rec
        text=recognizer.recognize_google(recaudio,language='en-US')
    except:
        text = "record again"
    return text