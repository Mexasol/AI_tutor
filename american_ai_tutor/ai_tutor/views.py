from django.shortcuts import render,redirect,get_object_or_404
from django.core.mail import send_mail
from django.contrib import messages
from .forms import UserRegisterForm
from django.views import View
from .models import Pdf_Model
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.contrib.auth.models import User, Group, Permission
import os
from django.http import JsonResponse
import threading
import queue
from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

# Create your views here.
pdfs_dir = os.path.join(settings.MEDIA_ROOT, 'pdfs')
@login_required
def home(request):
    document_names = []
    if request.method == "POST":
        user = request.user
        documents = request.FILES.getlist("pdf")
        document_names = [document.name for document in documents]
        documents_list = [Pdf_Model(file=document, user=user) for document in documents]
        Pdf_Model.objects.bulk_create(documents_list)
        print("Uploaded PDF names:", document_names)
        messages.success(request, "PDF Uploaded")
    return render(request, 'ai_tutor/index.html', {'document_names': document_names})

def delete_pdf(request, pdf_id):
    pdf = get_object_or_404(Pdf_Model, id=pdf_id, user=request.user)
    pdf.delete()
    return redirect('profile')

OPENAI_API_KEY= ""
def qa_llm():
    pdfs = Pdf_Model.objects.all()
    documents_pdf = []
    for pdf in pdfs:
        pdf_path = pdf.file.path
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        documents_pdf.append(document)

    loader = PyPDFDirectoryLoader(pdfs_dir)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    print(len(docs))
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )   
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa
 

@login_required
def question_answering(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        print(question)
        
        chat_history = request.session.get('chat_history', [])
        qa = qa_llm()
        generated_text = qa(question)
        answer = generated_text['result']

        if answer:
            print('----------------------------------')
            print(answer)
            print('----------------------------------')
            chat_history.append({'question': question, 'answer': answer})
            request.session['chat_history'] = chat_history
            request.session.save()
        else:
            answer = "No matching found."

        return JsonResponse({'answer': answer, 'chat_history': chat_history})
    else:
        return render(request, 'ai_tutor/chatbot.html', {'answer': ''})




@login_required
def profile(request):
    user_pdfs = Pdf_Model.objects.filter(user=request.user)
    return render(request, 'ai_tutor/profile.html', {'user_pdfs': user_pdfs})

import csv

def register_users_from_csv(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            first_name = row['First_Name']
            last_name = row['Last_Name']
            email = row['Email']
            password = first_name  # or any other logic for password

            # Check if user already exists
            if not User.objects.filter(username=email).exists():
                # Create user if not exists
                user = User.objects.create_user(
                    username=email, email=email, password=password,
                    first_name=first_name, last_name=last_name
                )
                user.is_staff = False
                user.save()
                print(f'Account created for {email}')
            else:
                print(f'User with email {email} already exists.')

# Call the function with the path to your CSV file
# register_users_from_csv('D:\\Office Work\\ai_tutor\\american_ai_tutor\\ai_tutor\\users.csv')

class User_Registration_view(View):
    def get(self, request):
        form = UserRegisterForm()
        return render(request, 'ai_tutor/register.html', {'form': form})

    def post(self, request):
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_staff = False  
            user.save()
            group, created = Group.objects.get_or_create(name='pdfuploadpermissions')
            permissions = Permission.objects.filter(codename__in=['add_pdf', 'change_pdf', 'delete_pdf'])
            group.permissions.set(permissions)
            user.groups.add(group)
            email = form.cleaned_data.get('email')
            messages.success(request, f'Account created for {email}!')
            return redirect('login')
        else:
            return render(request, 'ai_tutor/register.html', {'form': form})

