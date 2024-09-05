import os
import tempfile
import whisperx
from dotenv import load_dotenv
from scripts.process_results_to_openai_format import process_audio_to_openai_training_format

# Load the environment variables
load_dotenv(override=True)

# Get the Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Function to process the audio to OpenAI training format
def speaker_transcription_and_identify(audio_data: bytes):
    """
    This function takes audio data and returns the transcription of the audio
    and the speaker identification of the audio.

    Parameters:
    audio_data (bytes): The raw audio data.

    Returns:
    dict: A dictionary containing the transcription and speaker identification of the audio.
    """
    # Configuration parameters for the whisperx model
    device = "cuda"
    batch_size = 3  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

    # Create a temporary file to write the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file.flush()  # Ensure data is written to disk

        # Whisper processing using the temporary file path
        audio = whisperx.load_audio(temp_audio_file.name)

        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=batch_size)

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Diarization of the text
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Process the audio to OpenAI training format
        result = process_audio_to_openai_training_format(result["segments"])

        final_format = {"messages": result}
        print(final_format)

        return final_format

# Example of Usage
if __name__ == "__main__":
    # Path to the local audio file
    local_audio_file = "/home/aipc/TrainingDialer/diarizationTrainingDialer/not_save/audios_train_assistant-20240818T012707Z-001/20240322-135350_2524758817-all.mp3"

    with open(local_audio_file, "rb") as audio_file:
        audio_data = audio_file.read()

    result = speaker_transcription_and_identify(audio_data)

    print(result)
