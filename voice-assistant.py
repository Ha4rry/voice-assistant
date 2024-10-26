import whisper
import ollama
import pyaudio
import wave
import threading
import os

# TODO: make some things in a config e.g. model name and wpm.

ollama_model_name = 'llama3.2:3b'

context = "CONTEXT:\n"

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 16000 # Record at 44100 samples per second
seconds = 3
prompt_audio_file_location = "temp/user_prompt_audio.wav"

response_file_location = "temp/response.txt"

wpm = 175

def enter_to_submit():
    global stopped
    input("Enter to submit.\n")
    stopped = True
    

try:
    try:
      ollama.chat(ollama_model_name)
    except ollama.ResponseError as e:
      print('Error:', e.error)
      if e.status_code == 404:
        print("Pulling " + ollama_model_name + "...")
        ollama.pull(ollama_model_name)
        
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    whisper_model = whisper.load_model("turbo")

    while True:
        print('Listening... ')
        
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
        
        frames = []  # Initialize array to store frames
        
    
        stopped = False
        enter_to_submit_thread = threading.Thread(target=enter_to_submit)
        enter_to_submit_thread.start()
        while stopped == False:
            data = stream.read(chunk)
            frames.append(data)
        
        
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()

        
        # Save the recorded data as a WAV file
        wf = wave.open(prompt_audio_file_location, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        
        
        question = whisper_model.transcribe(prompt_audio_file_location, prompt="The sentence may be cut off, do not make up words or characters to fill in the rest of the sentence.")
        
        os.remove(prompt_audio_file_location)
        
        print("User:", question["text"])
        
        context += ( 'Question: ' +  question["text"] + "\n")
        
        response = ollama.chat(model=ollama_model_name, messages=[
          {
            'role': 'system',
            'content': context,
          },
          {
            'role': 'user',
            'content': question["text"],
          }
        ])
        context += ( "\n" + 'Response: ' + response['message']['content'] )
        
        print("\nResponse:\n")
        print(response['message']['content'])
        response_file = open(response_file_location, 'w')
        response_file.write(response['message']['content'])
        response_file.close()
        os.system(str("espeak -v gmw/en -s " + str(wpm) + " -f" + response_file_location))
        os.remove(response_file_location)
        
except KeyboardInterrupt:
    #try:
        #lock.release() # doesnt work
        #print("Released lock on enter to submit input thread")
    #except:
        #pass
    try:
        stream.stop_stream()
        stream.close()
        print("Closed audio recording stream")
    except: 
        pass
    try:
        wf.close()
        print("Closed audio file")
    except:
        pass
    try:
        response_file.close()
        print("Closed response file")
    except:
        pass
    try:
        # Terminate the PortAudio interface
        p.terminate()
        print("Terminated PortAudio interface")
    except:
        pass
    try: 
        os.remove(prompt_audio_file_location)
        print("Removed user audio file")
    except:
        pass
    try: 
        os.remove(response_file_location)
        print("Removed response file")
    except:
        pass
    print("bye bye")
    try:
        quit()
    except:
        pass
