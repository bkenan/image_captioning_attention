import pyttsx3


def read_text(text):
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    engine.setProperty('rate', 125)     # setting up new voice rate

    """VOLUME"""
    engine.setProperty('volume', 1.0)    # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')       # getting details of current voice
    engine.setProperty('voice', voices[1].id)   # changing index, changes voices. 1 for female

    engine.say(text)
    engine.runAndWait()
    engine.stop()
