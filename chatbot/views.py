from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .chatbot_rag import get_chatbot_response  
import base64
from rest_framework.permissions import AllowAny
import os
import tempfile
import threading
import time

# Session cache with expiration
SESSION_CACHE = {}
SESSION_EXPIRY = {}  # Track when cache entries should expire
SESSION_LOCK = threading.Lock()
CACHE_EXPIRY_SECONDS = 1800  # 30 minutes

class ChatbotQueryView(APIView):
    """Endpoint for multipart requests (text and/or image)"""
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)  # Support file uploads

    def post(self, request):
        start_time = time.time()
        query = request.data.get('query', '')
        image = request.FILES.get('image')

        # Validate: at least one of query or image must be provided
        if not query and not image:
            return Response(
                {'error': 'At least one of query or image is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Use session key as cache key
            session_key = request.session.session_key
            if not session_key:
                request.session.create()
                session_key = request.session.session_key

            # Get chat history from cache with expiry management
            chat_history = self._get_chat_history(session_key, request)

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

            # Update chat history in background (non-blocking)
            self._update_history_async(session_key, request, query, response, image_data, chat_history)

            processing_time = time.time() - start_time
            print(f"Response processed in {processing_time:.2f}s")
            return Response({'response': response}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error: {str(e)}")
            return Response(
                {'error': f'An error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _get_chat_history(self, session_key, request):
        """Get chat history with cache expiry management"""
        with SESSION_LOCK:
            # Clean expired cache entries
            current_time = time.time()
            expired_keys = [k for k, v in SESSION_EXPIRY.items() if v < current_time]
            for k in expired_keys:
                if k in SESSION_CACHE:
                    del SESSION_CACHE[k]
                if k in SESSION_EXPIRY:
                    del SESSION_EXPIRY[k]
            
            # Get or create chat history
            if session_key in SESSION_CACHE:
                chat_history = SESSION_CACHE[session_key]
            else:
                chat_history = request.session.get('chat_history', [])
                SESSION_CACHE[session_key] = chat_history
                
            # Update expiry time
            SESSION_EXPIRY[session_key] = time.time() + CACHE_EXPIRY_SECONDS
            
            return chat_history.copy()  # Return a copy to avoid race conditions

    def _update_history_async(self, session_key, request, query, response, image_data, chat_history):
        """Update chat history asynchronously"""
        def update_history():
            with SESSION_LOCK:
                # Get the latest chat history (it might have been updated by other requests)
                current_history = SESSION_CACHE.get(session_key, chat_history)
                
                # Append new messages
                current_history.append({
                    'role': 'user',
                    'content': query,
                    'image': True if image_data else None
                })
                current_history.append({'role': 'assistant', 'content': response})
                
                # Limit to last 6 messages
                current_history = current_history[-6:]
                
                # Update cache
                SESSION_CACHE[session_key] = current_history
                
                # Update session
                request.session['chat_history'] = current_history
                request.session.modified = True
        
        thread = threading.Thread(target=update_history)
        thread.daemon = True
        thread.start()

class TextOnlyChatbotView(APIView):
    """Endpoint for text-only JSON requests (more efficient)"""
    permission_classes = [AllowAny]
    parser_classes = (JSONParser,)  # Support JSON requests only

    def post(self, request):
        start_time = time.time()
        query = request.data.get('query', '')

        # Validate query
        if not query:
            return Response(
                {'error': 'Query is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Use session key as cache key
            session_key = request.session.session_key
            if not session_key:
                request.session.create()
                session_key = request.session.session_key

            # Get chat history from cache with expiry management
            chat_history = self._get_chat_history(session_key, request)

            # Generate response (no image data for this endpoint)
            response = get_chatbot_response(query, None, chat_history)
            
            # Filter out links and brackets from response
            filtered_response = self._filter_response(response)

            # Update chat history in background (non-blocking)
            self._update_history_async(session_key, request, query, filtered_response, chat_history)

            processing_time = time.time() - start_time
            print(f"Text-only response processed in {processing_time:.2f}s")
            return Response({'response': filtered_response}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error: {str(e)}")
            return Response(
                {'error': f'An error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _filter_response(self, response):
        """Filter out links and brackets from response"""
        import re
        
        # Replace URLs with a placeholder
        url_pattern = r'https?://[^\s<>"\')]+|www\.[^\s<>"\')]+\.[^\s<>"\')]+' 
        filtered_text = re.sub(url_pattern, "[link removed]", response)
        
        # Remove brackets and their contents
        bracket_pattern = r'\[.*?\]'
        filtered_text = re.sub(bracket_pattern, "", filtered_text)
        
        # Clean up any extra whitespace caused by the removals
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        filtered_text = filtered_text.strip()
        
        return filtered_text

    def _get_chat_history(self, session_key, request):
        """Get chat history with cache expiry management"""
        with SESSION_LOCK:
            # Clean expired cache entries
            current_time = time.time()
            expired_keys = [k for k, v in SESSION_EXPIRY.items() if v < current_time]
            for k in expired_keys:
                if k in SESSION_CACHE:
                    del SESSION_CACHE[k]
                if k in SESSION_EXPIRY:
                    del SESSION_EXPIRY[k]
            
            # Get or create chat history
            if session_key in SESSION_CACHE:
                chat_history = SESSION_CACHE[session_key]
            else:
                chat_history = request.session.get('chat_history', [])
                SESSION_CACHE[session_key] = chat_history
                
            # Update expiry time
            SESSION_EXPIRY[session_key] = time.time() + CACHE_EXPIRY_SECONDS
            
            return chat_history.copy()  # Return a copy to avoid race conditions

    def _update_history_async(self, session_key, request, query, response, chat_history):
        """Update chat history asynchronously"""
        def update_history():
            with SESSION_LOCK:
                # Get the latest chat history (it might have been updated by other requests)
                current_history = SESSION_CACHE.get(session_key, chat_history)
                
                # Append new messages
                current_history.append({'role': 'user', 'content': query})
                current_history.append({'role': 'assistant', 'content': response})
                
                # Limit to last 6 messages
                current_history = current_history[-6:]
                
                # Update cache
                SESSION_CACHE[session_key] = current_history
                
                # Update session
                request.session['chat_history'] = current_history
                request.session.modified = True
        
        thread = threading.Thread(target=update_history)
        thread.daemon = True
        thread.start()

class ClearChatHistoryView(APIView):
    """Endpoint to clear chat history"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        session_key = request.session.session_key
        
        if session_key:
            # Clear from session
            if 'chat_history' in request.session:
                del request.session['chat_history']
                request.session.modified = True
            
            # Clear from cache
            with SESSION_LOCK:
                if session_key in SESSION_CACHE:
                    del SESSION_CACHE[session_key]
                if session_key in SESSION_EXPIRY:
                    del SESSION_EXPIRY[session_key]
        
        return Response({'message': 'Chat history cleared'}, status=status.HTTP_200_OK)