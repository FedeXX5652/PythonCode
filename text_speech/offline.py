import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("rate", 150)
engine.setProperty("voice", voices[1].id)
engine.say("Opening Spotify")
engine.say("CLOSING")
engine.runAndWait()
engine.stop()