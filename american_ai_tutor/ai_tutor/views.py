from django.shortcuts import render,redirect,get_object_or_404
from django.contrib import messages
from .forms import UserRegisterForm
from django.views import View
from .models import Pdf_Model
from django.contrib.auth.decorators import login_required
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings
from django.contrib.auth.models import User, Group, Permission
import os

# Import your process_answer function here

# Create your views here.
pdfs_dir = os.path.join(settings.MEDIA_ROOT, 'pdfs')
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



# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
#                                           use_auth_token=True,)


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
#                                              device_map='auto',
#                                              torch_dtype=torch.float16,
#                                              use_auth_token=True,
#                                              )

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



def qa_llm():
  llm = llm_pipeline()
  pdfs = Pdf_Model.objects.all()
  documents_pdf = []
  for pdf in pdfs:
    pdf_path = pdf.file.path  
    loader = PyPDFLoader(pdf_path) 
    document = loader.load()
    documents_pdf.append(document)

  loader = PyPDFDirectoryLoader(pdfs_dir)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  docs = text_splitter.split_documents(data)
  print(len(docs))
  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  db = Chroma.from_documents(docs, embeddings)
  retriever = db.as_retriever()
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
  return qa
 

def question_answering(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        print(question)
        qa = qa_llm()
        generated_text = qa(question)
        answer = generated_text['result']
        return render(request, 'ai_tutor/chatbot.html', {'answer': answer})
    else:
        return render(request, 'ai_tutor/chatbot.html', {'answer': ''})
    

@login_required
def profile(request):
    user_pdfs = Pdf_Model.objects.filter(user=request.user)
    return render(request, 'ai_tutor/profile.html', {'user_pdfs': user_pdfs})



class User_Registration_view(View):
    def get(self, request):
        form = UserRegisterForm()
        return render(request, 'ai_tutor/register.html', {'form': form})

    def post(self, request):
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_staff = True  
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
