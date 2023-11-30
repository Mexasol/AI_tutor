from django.shortcuts import render,redirect
from django.contrib import messages
from .forms import UserRegisterForm
from django.views import View
from .models import Pdf_Model
from django.contrib.auth.decorators import login_required
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import InstructionSerializer
# Import your process_answer function here

# Create your views here.
pdfs_dir = os.path.join(settings.MEDIA_ROOT, 'pdfs')
def home(request):
    document_names = []
    if request.method == "POST":
        documents = request.FILES.getlist("pdf")
        
        # Extracting the names of the uploaded documents
        document_names = [document.name for document in documents]
        
        documents_list = [Pdf_Model(file=document) for document in documents]
        Pdf_Model.objects.bulk_create(documents_list)

        # Log or use the document names as needed
        print("Uploaded PDF names:", document_names)  # Example: Printing the names

        messages.success(request, "PDF Uploaded")
       
    return render(request, 'ai_tutor/index.html', {'document_names': document_names})




# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
#                                           use_auth_token=True,)


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
#                                              device_map='auto',
#                                              torch_dtype=torch.float16,
#                                              use_auth_token=True,
#                                              )

checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token="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")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
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
documents_load_pdf = []

def pdf_view():
    pdfs = Pdf_Model.objects.all()
    documents_pdf = []
    for pdf in pdfs:
        pdf_path = pdf.file.path  # Replace 'your_file_field' with your actual field name
        loader = PyPDFLoader(pdf_path)  # Initialize PyPDFLoader with the file path
        document = loader.load()
        documents_pdf.append(document)

    loader = PyPDFDirectoryLoader(pdfs_dir)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(data)
    documents_load_pdf.append(docs)


def qa_llm():
  llm = llm_pipeline()
  pdfs = Pdf_Model.objects.all()
  documents_pdf = []
  for pdf in pdfs:
    pdf_path = pdf.file.path  # Replace 'your_file_field' with your actual field name
    loader = PyPDFLoader(pdf_path)  # Initialize PyPDFLoader with the file path
    document = loader.load()
    documents_pdf.append(document)

  loader = PyPDFDirectoryLoader(pdfs_dir)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  docs = text_splitter.split_documents(data)
  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  db = Chroma.from_documents(docs, embeddings)
  retriever = db.as_retriever()
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
  return qa

def process_answer(instruction):
    response = ''
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer



@api_view(['POST'])
def answer_view(request):
    if request.method == 'POST':
        serializer = InstructionSerializer(data=request.data)
        if serializer.is_valid():
            instruction = serializer.validated_data['instruction']
            answer = process_answer(instruction)
            return Response({"answer": answer})
        return Response(serializer.errors, status=400)
# def process_question(request):
#     if request.method == 'POST':
#         question = request.POST.get('question')
#         print(question)
#         answer = process_answer(question)
#         return render(request, 'ai_tutor/answer.html', {'answer': answer})
#     else:
#         return render(request, 'ai_tutor/answer.html', {'answer': ''})   


# def qa_llm():
#   llm = llm_pipeline()
#   embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#   db = Chroma.from_documents(docs, embeddings)
#   retriever = db.as_retriever()
#   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#   return qa
@login_required
def profile(request):
    return render(request,'ai_tutor/profile.html')



class User_Registration_view(View):
    def get(self, request):
        form = UserRegisterForm()
        return render(request, 'ai_tutor/register.html', {'form': form})

    def post(self, request):
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            email = form.cleaned_data.get('email')
            messages.success(request, f'Account created for {email}!')
            return redirect('login')
        else:
            return render(request, 'ai_tutor/register.html', {'form': form})
