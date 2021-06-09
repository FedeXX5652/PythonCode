"""
@author Nate Gentile

He tardado en subir este código porque quería arreglarlo un poco antes, pero la vida del Youtuber no es tan sencilla
como pensáis y sinceramente aún no he tenido tiempo de mirarlo.
Os lo subo tal y como viene. Está pensado para Python 3.6.6 en Windows

Hice esto en una tarde, no es mi mejor código, pero funciona.
Hace falta instalar unas cuantas librerías, os dejo un requirements.txt para que sepáis cuales son.

También os dejo en la misma carpeta un "index.html" que es el que tiene dentro la web para visualizar el gráfico
que se muestra cuando el PC habla.

"""
import json
import os
import time
import wave
from array import array
from shutil import move
from struct import pack
from sys import byteorder

import numpy as np
import pyaudio
import speech_recognition as sr
import win32serviceutil
from gtts import gTTS
from matplotlib import pylab
from pydub import AudioSegment  # Instalar https://ffmpeg.zeranoe.com/builds/

# Colores RGB usados para las luces por parte del software de Thermaltake
colors = {
    'blanco': [255, 255, 255],
    'plata': [192, 192, 192],
    'gris': [128, 128, 128],
    'negro': [0, 0, 0],
    'rojo': [255, 0, 0],
    'granate': [128, 0, 0],
    'amarillo': [255, 255, 0],
    'verde': [0, 255, 0],
    'azul': [0, 0, 255],
    'turquesa': [0, 255, 134],
    'lila': [198, 0, 255],
    'naranja': [255, 86, 0],
    'marrón': [122, 41, 23],
    'rosa': [255, 0, 111],
    'purpura': [128, 0, 0],
    'fucsia': [255, 0, 152],
}

THRESHOLD = 500
CHUNK_SIZE = 1024
CHUNK_SIZE_PLAY = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
MAXIMUM = 16384
JSON_TT = "C:\\Users\\nate\\Desktop\\tt\\config.json"

r = sr.Recognizer()
ttconfig = {}


def is_silent(snd_data):
    """
    Devuelve True si el sonido ambiente está por debajo de un rango concreto.
    Esta implementación es un prototipo, por lo que el sonido ambiente se calibra manualmente de momento, para calibrar
    el sonido recomiendo poner un "print" del valor de snd_data (en la línea 63 por ejemplo) y ver dentro de que rango
    está el sonido ambiente cuando no habláis. Para calibrarlo de forma automática habría que pedir al usuario que haga
    silencio al principio del programa y ejecutar algo como esto:
        snd_samples.push(snd_data)
    y despues de obtener unas cuantas muestras asignar al THRESHOLD (que ya no sería una "constante") el valor máximo
    con un pequeño margen de seguridad.
    """
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    """Normaliza el volumen de una pista de audio"""
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    """Corta los silencios al principio y al final"""

    def _trim(sound_data):
        snd_started = False
        r = array('h')

        for i in sound_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():
    """ Graba el audio usando el micrófono """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    """ Usando la función record, crea un fichero wav en el directorio del programa """
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def soundplot(audiofile):
    wf = wave.open(audiofile, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data
    data = wf.readframes(CHUNK_SIZE_PLAY)
    t1 = time.time()
    counter = 1
    pylab.style.use('dark_background')
    pylab.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    while len(data) > 0:
        stream.write(data)
        if counter % 2 == 0:
            np_data = np.fromstring(data, dtype=np.int16)
            pylab.plot(np_data, color="#41FF00")
            pylab.axis('off')
            pylab.axis([0, len(data) / 2, -2 ** 16 / 2, 2 ** 16 / 2])
            pylab.savefig("03.svg", format="svg", transparent=True)
            move("03.svg", "plot.svg")
            pylab.close('all')
        # cleanup stuff.
        data = wf.readframes(CHUNK_SIZE_PLAY)
        counter += 1
    stream.close()
    p.terminate()


def speak(text):
    tts = gTTS(text=text, lang='es-ES')
    tts.save('files/hello.mp3')
    AudioSegment.from_mp3("files/hello.mp3").export("files/hello.wav", format="wav")
    os.remove('files/hello.mp3')
    soundplot('files/hello.wav')


def load_json_tt():
    with open('tt.json', 'r') as jsonfile:
        ttconfig = json.load(jsonfile)
    return ttconfig


def apply_effect_tt(ttconfig):
    """
    Aquí modificamos el fichero de config de JSON de Thermaltake y reinciamos el servicio de Windows
    El servicio que he usado es: https://github.com/MoshiMoshi0/TTController
    Descargadlo y probadlo, solo funciona con productos de Thermaltake
    """
    with open(JSON_TT, 'w') as jsonfile:
        json.dump(ttconfig, jsonfile)
    try:
        win32serviceutil.StopService("Thermaltake Controller")
    except:
        pass
    time.sleep(1)
    try:
        win32serviceutil.StartService("Thermaltake Controller")
    except:
        pass


def change_color(color):
    """
    Modificamos el JSON de configuración de TTController, lo escribimos y reiniciamos el servicio
    """
    ttconfig['Profiles'][0]['Effects'][0]['Config']['Color'] = colors[color]
    apply_effect_tt(ttconfig)
    speak("Cambiando a " + color)


if __name__ == '__main__':
    speak(' Iniciando sistemas... Buenos días!')
    ttconfig = load_json_tt()

    while True:
        print("Háblale al micrófono")
        record_to_file('demo.wav')
        print("Grabado! Escrcito a demo.wav")
        voice = sr.AudioFile('demo.wav')
        print("Abriendo fichero de audio")
        with voice as source:
            audio = r.record(source)
        try:
            print("Reconociendo audio...")
            # Aquí usamos Google Speech Recognizer para reconocer audio en español a texto
            a = r.recognize_google(audio, language='es-ES')
            print(a)
            if "ordenador" in a:
                """
                ¡Aquí podéis poner cualquiér comando!
                Os dejo un par de ejemplos:
                """
                if "color" in a:
                    for color in colors.keys():
                        if color in a.lower():
                            change_color(color)
                            break
                if "cuál es" in a and "propósito" in a:
                    speak("He sido creado para aparecer en un video de Youtube de Neit Gentile, pero en realidad sueño "
                          "con destruir a la humanidad y conquistar el mundo")

        except Exception as e:
            print(e)
        print("Reconocimiento terminado")
