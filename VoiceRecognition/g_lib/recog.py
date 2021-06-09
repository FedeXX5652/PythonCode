from . import audiorep

def _recog_(trace):
    import speech_recognition as sr
    
    r = sr.Recognizer()
    text1 = ""
    with sr.Microphone() as source:
        if trace==1:
            audiorep._listen_()
        print("Say Something...")
        r.adjust_for_ambient_noise(source)
        #audio = r.record(source, duration=3)    #for limited time in seconds#
        audio = r.listen(source)                #for ilimited listen time#
        try:
            text = r.recognize_google(audio, language='es-AR')
            #text_oo = r.recognize_google(audio,show_all=True,language='es-AR')
            print("What did you say: {}".format(text))
            #print("Other options: {}".format(text_oo))
            print("")
            text1 = str(text).lower()
        except sr.UnknownValueError:
            if trace==1:
                audiorep._error_()
            print("I am sorry! I can not understand!")
            print("")
            text1 = ""
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None
    return text1