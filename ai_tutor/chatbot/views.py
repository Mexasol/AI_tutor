from django.shortcuts import render

# Create your views here.
def student_login(request):
    return render(request,'chatbot/index.html')

def chatbot(request):
    return render(request,'chatbot/chatbot.html')