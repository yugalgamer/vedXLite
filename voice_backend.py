from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import wave
import json
from vosk import Model, KaldiRecognizer
import subprocess
from vedix_core import get_vedix

app = Flask(__name__)
CORS(app)

# Initialize VediX
vedix = get_vedix()

# Load Vosk Model for English (adjust path if necessary)
vosk_model_path = "model/vosk-model-small-en-us-0.15"
model = Model(vosk_model_path)
recognizer = KaldiRecognizer(model, 16000)

@app.route('/api/voice-interact', methods=['POST'])
def voice_interact():
    """Process voice input with VediX offline AI assistant"""
    if 'audio' not in request.files:
        return jsonify(success=False, error="No audio file provided"), 400
    
    audio_file = request.files['audio']
    
    # Save uploaded audio to a temp file
    fd, temp_audio_path = tempfile.mkstemp(suffix='.webm')
    try:
        audio_file.save(temp_audio_path)
        
        # Convert webm to wav
        temp_wav_path = tempfile.mktemp(suffix='.wav')
        command = ['ffmpeg', '-y', '-i', temp_audio_path, '-ar', '16000', '-ac', '1', '-f', 'wav', temp_wav_path]
        subprocess.run(command, check=True)
        
        # Recognize using Vosk
        with wave.open(temp_wav_path, "rb") as wf:
            wf_content = wf.readframes(wf.getnframes())
            if recognizer.AcceptWaveform(wf_content):
                result = recognizer.Result()
                recognized_data = json.loads(result)
                recognized_text = recognized_data.get('text', '')
            else:
                return jsonify(success=False, error="No recognizable speech"), 400
        
        # Process with VediX
        vedix_response = vedix.process_voice_command(recognized_text)
        return jsonify(
            success=True, 
            reply=vedix_response,
            recognized_text=recognized_text,
            vedix_active=True
        )

    except subprocess.CalledProcessError as e:
        return jsonify(success=False, error="Error converting audio"), 500

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    
    finally:
        os.close(fd)
        os.remove(temp_audio_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

@app.route('/api/status')
def status():
    return jsonify(connected=True, message="Voice backend running!")

if __name__ == '__main__':
    app.run(port=5000, debug=True)

