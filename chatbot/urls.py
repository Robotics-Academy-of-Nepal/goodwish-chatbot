from django.urls import path, include
from .views import ChatbotQueryView, ClearChatHistoryView, TextOnlyChatbotView


urlpatterns = [
    path('query-chatbot/', ChatbotQueryView.as_view(), name="Chatbot Query"),
     path('chat/text/', TextOnlyChatbotView.as_view(), name='chatbot-text-query'),
    path('clear-history/', ClearChatHistoryView.as_view(), name='clear_history'),
]