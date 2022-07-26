import speech_recognition as sr
import pyttsx3
import csv
listener = sr.Recognizer()
speaker = pyttsx3.init()
voices = speaker.getProperty('voices')
speaker.setProperty('voice',voices[1].id)

d=False
with sr.Microphone() as source:
    print("Recording audio")
    listener.adjust_for_ambient_noise(source, duration=1)
    recaudio = listener.listen(source)
    print("recognizing command")
    try :
        command = listener.recognize_google(recaudio)
        command = command.lower()
        print(f"You Spoke {command}")
    except:
        print("Error")
    
    if 'alexa get details' in command:
        command = command.replace('alexa get details', '')
        speaker.say("Tell your roll number")
        speaker.runAndWait()
        roll = listener.recognize_google(listener.listen(source))
        print(roll)
        with open('details.csv') as f:
            head = next(f)
            reader = csv.reader(f)
            for r in reader:
                for i in range(len(r)):
                    if r[i] == roll:
                        print("Name : ", r[1])
                        print("Percentage : ", r[2])
                        print("Attendence", r[3])
                        d=True
                        break
            if not d:
                speaker.say("No student Found")
                speaker.runAndWait()
