from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .chatbot_rag import get_chatbot_response  # Adjust import based on your project structure
import base64
from rest_framework.permissions import AllowAny
import os
import tempfile
from .speech_to_text import transcribe_audio, convert_to_wav, TranscriptionError


class ChatbotQueryView(APIView):
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)  # Support file uploads

    def post(self, request):
        query = request.data.get('query', '')
        image = request.FILES.get('image')

        # Validate: at least one of query or image must be provided
        if not query and not image:
            return Response(
                {'error': 'At least one of query or image is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Ensure session is available
            if not request.session.session_key:
                request.session.create()

            # Get chat history from session (default to empty list)
            chat_history = request.session.get('chat_history', [])

            # Prepare image data if provided
            image_data = None
            if image:
                if not image.content_type.startswith('image/'):
                    return Response(
                        {'error': 'Invalid file type. Please upload an image'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                # Convert image to base64 for Azure OpenAI
                image_data = base64.b64encode(image.read()).decode('utf-8')

            # Generate response
            response = get_chatbot_response(query, image_data, chat_history)

            # Update chat history (store image_data only if necessary)
            chat_history.append({
                'role': 'user',
                'content': query,
                'image': image_data  # Optional: omit if storage size is a concern
            })
            chat_history.append({'role': 'assistant', 'content': response})
            chat_history = chat_history[-10:]  # Limit to last 10 messages

            # Save updated chat history to session
            request.session['chat_history'] = chat_history
            request.session.modified = True

            return Response({'response': response}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': f'An error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        

class TranscribeAudioView(APIView):
    """Endpoint to transcribe audio files and send to chatbot"""
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        # Check if file is provided
        if 'audio_file' not in request.FILES:
            return Response({'error': 'No audio file provided'}, status=400)
            
        audio_file = request.FILES['audio_file']
        print(f"Received file: {audio_file.name}, size: {audio_file.size}, type: {audio_file.content_type}")
        temp_file = None
        wav_file = None
        
        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_file.read())
                temp_file_path = temp_file.name
            
            print(temp_file_path)
            
            # # Convert to WAV if needed
            wav_file = convert_to_wav(temp_file.name)
            
            # Transcribe
            transcription, language = transcribe_audio(wav_file)
            
            # Get chat history from session
            if not request.session.session_key:
                request.session.create()
            
            chat_history = request.session.get('chat_history', [])
            
            # Generate response using the transcribed text as query
            response = get_chatbot_response(transcription, None, chat_history)
            
            # Update chat history with transcribed text and response
            chat_history.append({
                'role': 'user',
                'content': f"[Audio Transcription ({language})]: {transcription}",
                'image': None
            })
            chat_history.append({'role': 'assistant', 'content': response})
            chat_history = chat_history[-10:]  # Limit to last 10 messages
            
            # Save updated chat history to session
            request.session['chat_history'] = chat_history
            request.session.modified = True
            
            return Response({
                'transcription': transcription,
                'language': language,
                'response': response
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=400)
            
        finally:
            # Clean up temp files
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            if wav_file and wav_file != temp_file.name and os.path.exists(wav_file):
                os.unlink(wav_file)

class ClearChatHistoryView(APIView):
    """Endpoint to clear chat history"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        if request.session.session_key:
            if 'chat_history' in request.session:
                del request.session['chat_history']
                request.session.modified = True
        
        return Response({'message': 'Chat history cleared'}, status=status.HTTP_200_OK)