# third party imports
import os
import re
import csv
import fitz
import json
import concurrent.futures
# django  imports
from .models import Pdf_Model,bannend_word
from django.views import View
from django.conf import settings
from django.contrib import messages
from .forms import UserRegisterForm
from django.http import JsonResponse
from django.core.mail import send_mail
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User, Group, Permission
from django.shortcuts import render,redirect,get_object_or_404

# langchain imports
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers import pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True,)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             )  
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


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    doc.close()
    return text

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'\s*([.,;!?])\s*', r'\1 ', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    cleaned_text = re.sub(r'CHAP TER \d+', r'\n\g<0>', cleaned_text)

    return cleaned_text

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
    return chunks

def process_pdf(pdf_model):
    pdf_text = extract_text_from_pdf(pdf_model.file.path)
    cleaned_text = clean_text(pdf_text)
    chunks = split_text(cleaned_text)

    # Create Document objects from chunks
    docs = [Document(page_content=chunk) for chunk in chunks]

    return docs

global_documents_pdf = None
global_qa = None


def llm_pipeline():
    pipe=pipeline("text-generation",
              model=model,
              tokenizer=tokenizer,
              torch_dtype=torch.bfloat16,
              device_map='auto',
              max_new_tokens=512,
              min_new_tokens=-1,
              top_k=30

    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def initialize_system():
    global global_documents_pdf, global_qa

    pdfs = Pdf_Model.objects.all()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        global_documents_pdf = [doc for sublist in executor.map(process_pdf, pdfs) for doc in sublist]

    # Initialize the QA model here
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(global_documents_pdf, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    global_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Prompt for the chatbot
aha_info_prompt = """
American High School Academy (AHSA) is a fully accredited, private high school located in Miami, Florida. The school is committed to providing high-quality education to prepare students both academically and personally for college and beyond. It offers a wide range of courses and learning pathways to cater to various student needs, including virtual and blended learning, credit recovery, dropout prevention, alternative education, English language learning, and summer school programs.

The school also offers NCAA approved courses, meeting the standards of quality and effectiveness set by the National Collegiate Athletic Association, Florida Standards, and the Common Core State Standards. This indicates a focus on comprehensive and rigorous academic programs.

The student population of American High School Academy includes about 600 students in grades 6-12, with a student-teacher ratio of 26 to 1. Approximately 70% of graduates from this school go on to attend a 4-year college, highlighting the school's effectiveness in preparing students for higher education.
"""

# Call this function when the server starts
    # initialize_system()

@login_required
def question_answering(request):
    chat_history = request.session.get('chat_history', [])

    if request.method == 'POST':
        question = request.POST.get('question')
        print(question)
        if "ahsa" in question.lower() or "american high school academy" in question.lower():
            answer = aha_info_prompt
        else:
            generated_text = global_qa(question)
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

def Academy_stats(request):
    # Count all users
    user_count = User.objects.filter(is_staff=False, is_superuser=False).count()

    # Count staff (non-admin) users
    staff_count = User.objects.filter(is_staff=True).count()

    # Count banned words
    banned_words_count = bannend_word.objects.count()

    data = {
        'user_count': user_count,
        'staff_count': staff_count,
        # 'admin_count': admin_count,
        'banned_words_count': banned_words_count,
    }

    return render(request, 'ai_tutor/stats.html', {'data': json.dumps(data), 'staff_count': staff_count, 'user_count': user_count, 'banned_words_count': banned_words_count})
    

@login_required  # Use this decorator to ensure the user is logged in
def Banneds_words(request):
    if request.method == 'POST':
        banned_word = request.POST.get('banned_words')
        if banned_word:
            # Associate the current user with the banned word
            bannend_word.objects.create(user=request.user, word=banned_word)
            messages.success(request, f'The word "{banned_word}" has been added to the banned list.')

    banned_words = bannend_word.objects.all()
    return render(request, 'ai_tutor/banned_words.html', {'banned_words': banned_words})


@login_required
def delete_banned_word(request, banned_word_id):
    # Delete: Remove a banned word
    banned_word = get_object_or_404(bannend_word, id=banned_word_id)
    if banned_word.user == request.user:
        banned_word.delete()
        messages.success(request, f'The word "{banned_word.word}" has been removed from the banned list.')
    else:
        messages.error(request, 'You are not authorized to delete this word.')

    return redirect('banned_words')

@login_required
def edit_banned_word(request, banned_word_id):
    banned_word = get_object_or_404(bannend_word, id=banned_word_id, user=request.user)

    if request.method == 'POST':
        new_word = request.POST.get('new_word')
        if new_word:
            banned_word.word = new_word
            banned_word.save()
            messages.success(request, f'The word has been updated to "{new_word}".')

    banned_words = bannend_word.objects.filter(user=request.user)
    return render(request, 'ai_tutor/banned_words.html', {'banned_words': banned_words})


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

