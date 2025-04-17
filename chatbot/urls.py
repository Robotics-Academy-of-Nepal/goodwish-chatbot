from django.urls import path, include
from .views import ChatbotQueryView, TranscribeAudioView, ClearChatHistoryView


urlpatterns = [
    path('query-chatbot/', ChatbotQueryView.as_view(), name="Chatbot Query"),
    path('transcribe/', TranscribeAudioView.as_view(), name='transcribe_audio'),
    path('clear-history/', ClearChatHistoryView.as_view(), name='clear_history'),
]