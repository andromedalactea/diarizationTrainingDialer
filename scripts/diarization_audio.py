# Import pyhton Libraries
import os

# Import third party libraries
import whisperx
from dotenv import load_dotenv

# Import local libraries
from process_results_to_openai_format import process_audio_to_openai_training_format

# Load the environment variables
load_dotenv(override=True) 

# Get the Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Function to process the audio to OpenAI training format
def  speaker_transcription_and_identify(audio_file):
    """
    This function takes an audio file and returns the transcription of the audio
    and the speaker identification of the audio.


    Parameters:
    audio_file (str): The path to the audio file.


    Returns:
    dict: A dictionary containing the transcription and speaker identification of the audio.
    """
    # Configuration parameters
    device = "cuda"
    batch_size = 3 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)


    # Whisper procesing
    audio = whisperx.load_audio(audio_file)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)


    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)




    ## Diarization of the text
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN,
                                                device=device)


    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    # Process the audio to OpenAI training format
    result = process_audio_to_openai_training_format(result["segments"])


    final_format = {"messages": result}
    print(final_format)

    return final_format

# Exmplae of Usage 
if __name__ == "__main__":
    audio_file = "/home/aipc/TrainingDialer/diarizationTrainingDialer/not_save/audios_train_assistant-20240818T012707Z-001/20240313-145536_2157850688-all.mp3"

    speaker_transcription_and_identify(audio_file)