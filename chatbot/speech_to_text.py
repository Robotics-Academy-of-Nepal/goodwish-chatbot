# transcription_service.py
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

# Load environment variables
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass

def convert_to_wav(input_file):
    """Convert audio file to WAV format compatible with transcription service using soundfile"""
    try:
        import soundfile as sf
        import os
        import io
        
        # Create output filename
        output_file = os.path.splitext(input_file)[0] + '_converted.wav'
        
        # For WebM files, we need to use a different approach
        # since soundfile might not support WebM directly
        if input_file.endswith('.webm'):
            # If your frontend is sending WebM, you might need to install additional libraries
            # or use a subprocess to call ffmpeg directly
            raise NotImplementedError("WebM conversion requires ffmpeg. Please install ffmpeg or use a different format.")
        
        # For WAV files that might have incorrect headers
        try:
            # Read the audio data
            data, samplerate = sf.read(input_file)
            
            # Write to a new file with standard WAV format
            sf.write(output_file, data, samplerate, subtype='PCM_16', format='WAV')
            
            print(f"Successfully converted audio to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error in soundfile conversion: {e}")
            raise
    
    except Exception as e:
        print(f"Error converting audio: {e}")
        raise

def transcribe_audio(file_path):
    """Transcribe audio using Azure Speech Services"""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise TranscriptionError("Azure Speech credentials not found")
    
    try:
        # Configure speech services
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["en-US", "ne-NP"]
        )
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        
        # Create recognizer and transcribe
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config
        )
        
        # Get transcription result
        result = speech_recognizer.recognize_once_async().get()
        source_language = speechsdk.AutoDetectSourceLanguageResult(result)
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, source_language.language
        else:
            raise TranscriptionError(f"Speech recognition failed: {result.reason}")
            
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {str(e)}")