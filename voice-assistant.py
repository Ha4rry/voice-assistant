import whisper
import ollama
import pyaudio
import wave
import threading
import os
import json
import subprocess
import multiprocessing

try:
    with open('config.json', 'r') as config_file:
        print("Config found!")
        config = json.load(config_file)
    print("Config loaded!")
except:
    print("Config not found! Creating...")
    
    config_file_path = "config.json"

    default_config = {
    "ollama_model_name": "llama3.2:3b", 
    "initial_system_prompt": "You are being spoken to a voice chat. Say everything in words, do NOT output symbols under any circumstances, always convert them to word form, because the text to speech only supports words. Remember, you are intended to be listened to.",
    "prompt_audio_file_location": "temp/user_prompt_audio.wav",
    "response_text_file_location": "temp/response.txt",
    "response_audio_file_location": "temp/response.wav",
    "whisper_model_type": "turbo",
    "whisper_prompt": "The sentence may be cut off, do not make up words or characters to fill in the rest of the sentence.",
    "piper_models_dir": "piper_models/",
    "piper_model_name": "en_GB-northern_english_male-medium.onnx",
    "piper_executable_location": "/opt/piper/piper",
    "wpm": 175
    }
    
    with open(config_file_path, 'w') as config_file:
        json.dump(default_config, config_file)
    print("Config created at " + config_file_path)
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    print("Config loaded!")

ollama_model_name = config.get('ollama_model_name')
initial_system_prompt = config.get('initial_system_prompt')
prompt_audio_file_location = config.get("prompt_audio_file_location")
response_text_file_location = config.get("response_text_file_location")
response_audio_file_location = config.get("response_audio_file_location")
whisper_model_type = config.get("whisper_model_type")
whisper_prompt = config.get("whisper_prompt")
piper_models_dir = config.get("piper_models_dir")
piper_model_name = config.get("piper_model_name")
piper_executable_location = config.get("piper_executable_location")
wpm = config.get("wpm")


context = "CONTEXT:\n"

chunk = 1024  # Record/play in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 16000 # Record at 44100 samples per second

def enter_to_submit():
    global stopped
    input("Enter to submit.\n")
    stopped = True

def start_ollama_server():
    subprocess.run(
    'ollama serve',
    shell=True,
    stdout=subprocess.DEVNULL, 
    stderr=subprocess.STDOUT,
    )

try:
    try:
        ollama.chat(ollama_model_name)
    except ollama.ResponseError as e:
        print('Error:', e.error)
        if e.status_code == 404:
            print("Pulling " + ollama_model_name + "...")
            ollama.pull(ollama_model_name)
    except:
        ollama_server_process = multiprocessing.Process(target=start_ollama_server)
        ollama_server_process.start()
        ollama_server_process_online=False
        while ollama_server_process_online == False:
            try:
                ollama.chat(ollama_model_name)
                ollama_server_process_online = True
            except:
                pass
                
        
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    whisper_model = whisper.load_model(whisper_model_type)

    while True:
        print('\nListening... ')
        
        recording_stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
        
        frames = []  # Initialize array to store frames
        
    
        stopped = False
        enter_to_submit_thread = threading.Thread(target=enter_to_submit)
        enter_to_submit_thread.start()
        while stopped == False:
            recording_data = recording_stream.read(chunk)
            frames.append(recording_data)

        
        # Stop and close the stream 
        recording_stream.stop_stream()
        recording_stream.close()
        


        
        # Save the recorded data as a WAV file
        with wave.open(prompt_audio_file_location, 'wb') as recording_wf:
            recording_wf.setnchannels(channels)
            recording_wf.setsampwidth(p.get_sample_size(sample_format))
            recording_wf.setframerate(fs)
            recording_wf.writeframes(b''.join(frames))
             
        question = whisper_model.transcribe(prompt_audio_file_location, prompt=whisper_prompt)
        
        os.remove(prompt_audio_file_location)
        
        print("User:", question["text"])
        
        context += ( 'Question: ' +  question["text"] + "\n")
        
        response = ollama.chat(model=ollama_model_name, messages=[
          {
            'role': 'system',
            'content': context + "\n SYSTEM: " + initial_system_prompt,
          },
          {
            'role': 'user',
            'content': question["text"],
          }
        ])
        context += ( "\n" + 'Response: ' + response['message']['content'] )
        
        print("\nResponse:\n")
        print(response['message']['content'])
        with open(response_text_file_location, 'w') as response_file:
            response_file.write(response['message']['content'])
        subprocess.run(
        str('echo """' + response['message']['content'] + '""" | ' + piper_executable_location + " --model "+ piper_models_dir + piper_model_name + " --output-file " + response_audio_file_location),
        shell=True,
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.STDOUT)

        os.remove(response_text_file_location)
        
        with wave.open(response_audio_file_location, 'rb') as playing_wf:
            playing_stream = p.open(format = p.get_format_from_width(playing_wf.getsampwidth()),
                    channels = playing_wf.getnchannels(),
                    rate = playing_wf.getframerate(),
                    output = True)
            
            playing_data = playing_wf.readframes(chunk)
            # Play the sound by writing the audio data to the stream
            try:
                while playing_data:
                    playing_stream.write(playing_data)
                    playing_data = playing_wf.readframes(chunk)
            except KeyboardInterrupt:
                print("INTERRUPTING SPEECH IS BROKEN AND RESULTS IN ERRORS.")
        # Close and terminate the stream
        playing_stream.stop_stream()
        playing_stream.close()
        os.remove(response_audio_file_location)
        
        
except KeyboardInterrupt:
    #try: # TODO: fix this
        #lock.release() # doesnt work
        #print("Released lock on enter to submit input thread")
    #except:
        #pass
    try:
        ollama_server_process.terminate()
        print("Terminated Ollama Server Process")
    except:
        pass
    try:
        ollama.chat(ollama_model_name, keep_alive=0)
        print("Cleared ollama model from memory")
    except:
        pass
    try:
        del whisper_model
        print("Cleared whisper model from memory")
    except:
        pass
    try:
        recording_stream.stop_stream()
        recording_stream.close()
        print("Closed audio recording stream")
    except: 
        pass
    try:
        playing_stream.stop_stream()
        playing_stream.close()
        print("Closed audio playing stream")
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
        os.remove(response_text_file_location)
        print("Removed response text file")
    except:
        pass
    try:
        os.remove(response_audio_file_location)
        print("Removed response audio file")
    except:
        pass
    print("bye bye")
    try:
        quit()
    except:
        pass

