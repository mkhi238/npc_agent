from config import TTS_MODEL, CHAR_LIMIT, configure_tts
import re
import wave
import tempfile
import os

tts_client = configure_tts()

def temp_wave():
  fd, path = tempfile.mkstemp(suffix=".wav")
  os.close(fd)
  return path

def chunk(text, limit = CHAR_LIMIT):
  words = text.split()
  chunks, cur = [], ""
  for w in words:
    if len(cur) + len(w) + 1 <= limit:
      cur = (cur + " " + w).strip()
    else:
      if cur:
        chunks.append(cur)
      cur = w
  if cur:
    chunks.append(cur)
  return chunks

def stitch(paths):
    if len(paths) == 1:
        return paths[0]
    out = temp_wave()
    with wave.open(paths[0], "rb") as w0:
        nchannels = w0.getnchannels()
        sampwidth = w0.getsampwidth()
        framerate = w0.getframerate()
    with wave.open(out, "wb") as o:
        o.setnchannels(nchannels)
        o.setsampwidth(sampwidth)
        o.setframerate(framerate)
        for p in paths:
            with wave.open(p, "rb") as w:
                o.writeframes(w.readframes(w.getnframes()))
    return out
  
def synthesize(text, voice="troy"):
    try:
        paths = []
        for c in chunk(text):
            p = temp_wave()
            resp = tts_client.audio.speech.create(
                model=TTS_MODEL, voice=voice, input=c, response_format="wav",
            )
            resp.write_to_file(p)
            paths.append(p)
        return stitch(paths) if paths else None
    except Exception as e:
        print("TTS failed:", e)
        return None