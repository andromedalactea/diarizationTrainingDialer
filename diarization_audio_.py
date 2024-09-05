# api.py

from flask import Flask, request, jsonify
from your_module import speaker_transcription_and_identify  # Asegúrate de reemplazar 'your_module' con el nombre del archivo donde está tu función

app = Flask(__name__)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """
    Endpoint to process audio file and return transcription and speaker identification.
    
    Expects a form-data request with an 'audio_file' field containing the audio file.
    """
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio_file']
    
    # Read the audio file
    audio_data = audio_file.read()
    
    # Call the processing function
    try:
        result = speaker_transcription_and_identify(audio_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
