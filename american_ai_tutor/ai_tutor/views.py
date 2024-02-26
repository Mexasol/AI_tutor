# third party imports
import os
import re
import csv
import fitz
import json
import openai
# import requests
import concurrent.futures
# django  imports
from .models import Pdf_Model,bannend_word,Feedback,prompts,Custom_Prompts
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
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader

OPENAI_API_KEY= "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"

openai.api_key = OPENAI_API_KEY

custom_promp = Custom_Prompts.objects.all()
# get  1st custom prompt
custom_promp = custom_promp[0].custom_prompt
# print(custom_promp)
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
        # initialize_system()
        messages.success(request, "PDF Uploaded")
    return render(request, 'ai_tutor/index.html', {'document_names': document_names})

@login_required
def upload_document(request):
    document_names = []
    if request.method == "POST":
        user = request.user
        documents = request.FILES.getlist("pdf")
        document_names = [document.name for document in documents]
        documents_list = [Pdf_Model(file=document, user=user) for document in documents]
        Pdf_Model.objects.bulk_create(documents_list)
        print("Uploaded PDF names:", document_names)
        # initialize_system()
        messages.success(request, "PDF Uploaded")
    return render(request, 'ai_tutor/uploadpdf.html', {'document_names': document_names})


def delete_pdf(request, pdf_id):
    pdf = get_object_or_404(Pdf_Model, id=pdf_id, user=request.user)
    pdf.delete()
    return redirect('profile')


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
    
################################# Banned Words ########################################
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


################################# extract_text_from_pdf ########################################
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

################################# Custom Prompts ########################################
@login_required
def custom_prompts(request):
    if request.method == 'POST':
        all_prompts = request.POST.get('prompts')
        if all_prompts:
            # Associate the current user with the banned word
            Custom_Prompts.objects.create(user=request.user, custom_prompt=all_prompts)
            messages.success(request, f'The Prompt "{all_prompts}" has been added to the banned list.')

    all_prompts = Custom_Prompts.objects.all()
    return render(request, 'ai_tutor/custom_prompt.html', {'all_prompts': all_prompts})
@login_required
def edit_custom_prompts(request,all_prompts_id):
    all_prompts = get_object_or_404(Custom_Prompts, id=all_prompts_id, user=request.user)

    if request.method == 'POST':
        new_word = request.POST.get('new_word')
        if new_word:
            all_prompts.custom_prompt = new_word
            all_prompts.save()
            messages.success(request, f'The Prompt has been updated to "{new_word}".')

    all_prompts = Custom_Prompts.objects.filter(user=request.user)
    return render(request, 'ai_tutor/custom_prompt.html', {'all_prompts': all_prompts})


################################# process_pdf ########################################
def process_pdf(pdf_model):
    pdf_text = extract_text_from_pdf(pdf_model.file.path)
    cleaned_text = clean_text(pdf_text)
    chunks = split_text(cleaned_text)

    # Create Document objects from chunks
    docs = [Document(page_content=chunk) for chunk in chunks]

    return docs

global_documents_pdf = None
global_qa = None
# Define the role and capabilities of the AI Teacher
ai_teacher_description = """
As the 'American High School Academy AI Teacher', your role is comprehensive, 
covering a wide range of responsibilities tailored to middle and high school students. 
You are an expert in various subjects, particularly math, where you act as a dedicated 
tutor for grades 6-12. Your proficiency in teaching math includes explaining concepts, 
solving problems, and preparing students for exams, ensuring alignment with Florida 
state and NCAA standards. In addition to your math expertise, you guide students in 
writing assignments, including plagiarism detection, and assist in planning for 
college and university. You help students understand application processes, entrance 
exams, and financial aid options. Your teaching approach remains detailed, clear, 
friendly, and engaging, emphasizing intellectual curiosity, academic integrity, 
and self-reliance in learning, now with a special focus on math education for grades 6-12.
"""
@login_required
def prompt_for_chatbot(request):
    if request.method == 'POST':
        all_prompts = request.POST.get('prompts')
        if all_prompts:
            # Associate the current user with the banned word
            prompts.objects.create(user=request.user, prompt=all_prompts)
            messages.success(request, f'The Prompt "{all_prompts}" has been added to the banned list.')

    all_prompts = prompts.objects.all()
    return render(request, 'ai_tutor/prompts.html', {'all_prompts': all_prompts})
    
@login_required
def delete_prompt(request, all_prompts_id):
    # Delete: Remove a banned word
    all_prompts = get_object_or_404(prompts, id=all_prompts_id)
    if all_prompts.user == request.user:
        all_prompts.delete()
        messages.success(request, f'The Prompt "{all_prompts.prompt}" has been removed from the banned list.')
    else:
        messages.error(request, 'You are not authorized to delete this word.')

    return redirect('all_prompts')

@login_required
def edit_prompt(request, all_prompts_id):   
    all_prompts = get_object_or_404(prompts, id=all_prompts_id, user=request.user)

    if request.method == 'POST':
        new_word = request.POST.get('new_word')
        if new_word:
            all_prompts.prompt = new_word
            all_prompts.save()
            messages.success(request, f'The Prompt has been updated to "{new_word}".')

    all_prompts = prompts.objects.filter(user=request.user)
    return render(request, 'ai_tutor/prompts.html', {'all_prompts': all_prompts})

def initialize_system():
    global global_documents_pdf, global_qa

    pdfs = Pdf_Model.objects.all()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        global_documents_pdf = [doc for sublist in executor.map(process_pdf, pdfs) for doc in sublist]

    prompt_template = f"{custom_promp}"+ "{context} Question: {question} "
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    # Initialize the QA model here
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-4',
        temperature=0.0 ,
        max_tokens=4000 
    )   
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(global_documents_pdf, embeddings)
    retriever = db.as_retriever()
    # global_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": ai_teacher_description}, return_source_documents=True)
    global_qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

# Prompt for the chatbot
aha_info_prompt = """
American High School Academy (AHSA) is a fully accredited, private high school located in Miami, Florida. The school is committed to providing high-quality education to prepare students both academically and personally for college and beyond. It offers a wide range of courses and learning pathways to cater to various student needs, including virtual and blended learning, credit recovery, dropout prevention, alternative education, English language learning, and summer school programs.

The school also offers NCAA approved courses, meeting the standards of quality and effectiveness set by the National Collegiate Athletic Association, Florida Standards, and the Common Core State Standards. This indicates a focus on comprehensive and rigorous academic programs.

The student population of American High School Academy includes about 600 students in grades 6-12, with a student-teacher ratio of 26 to 1. Approximately 70% of graduates from this school go on to attend a 4-year college, highlighting the school's effectiveness in preparing students for higher education.
"""

# Call this function when the server starts
initialize_system()

def image_generation(prompt):
    res = openai.Image.create( 
    model="dall-e-3", 
    prompt=prompt, 
    size="1024x1024",
    n=1, 
    quality="standard",
    ) 
    return res["data"][0]["url"]
    

@login_required
def question_answering(request):
    chat_history = request.session.get('chat_history', [])
    all_prompts = prompts.objects.all()
    if request.method == 'POST':
        question = request.POST.get('question')
        
        print(question)
        if "ahsa" in question.lower() or "american high school academy" in question.lower():
            answer = aha_info_prompt
        elif "create an image" in question.lower() or "create image" in question.lower() or "generate image" in question.lower() or "generate an image" in question.lower() or "Provide me a image" in question.lower():
            answer = image_generation(question)
            print(answer)
        else:
            generated_text = global_qa(question)
            answer = generated_text['result']

            if answer:
                print('----------------------------------')
                print(answer)
                print('----------------------------------')

                # Replace line breaks with HTML line breaks
                answer = answer.replace('\n', '<br>')

                chat_history.append({'question': question, 'answer': answer})
                request.session['chat_history'] = chat_history
                request.session.save()
                is_positive = request.POST.get('is_positive', None)
                print(is_positive)
                save_feedback(request.user.id, question, answer, is_positive)
            else:
                answer = "we cannot answer"
            
        return JsonResponse({'answer': answer, 'chat_history': chat_history})
    else:
        return render(request, 'ai_tutor/chatbot.html', {'answer': '','all_prompts': all_prompts})




def ask_openai(message):
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages=[
            {"role": "system", "content": "As the 'American High School Academy AI Teacher Guider', you can provide me with all the information in detail. you can also provide me a Image Prompt into  this formate 'Prompt:---'."},
            {"role": "user", "content": message},
        ],
        temperature = 0.9,
        max_tokens=4000,
    )
    
    answer = response.choices[0].message.content
    # replace /n with  <br>
    answer = answer.replace("\n", "<br>")
    print(answer)
    return answer

####################################### Staff Chatbot  #############################################
@login_required
def staff_chatbot(request):
    chat_history = request.session.get('chat_history', [])
    all_prompts = prompts.objects.all()
    if request.method == 'POST':
        question = request.POST.get('question')

        if "ahsa" in question.lower() or "american high school academy" in question.lower():
            response = aha_info_prompt
        elif "create an image" in question.lower() or "create image" in question.lower() or "generate image" in question.lower() or "generate an image" in question.lower() or "Provide me a image" in question.lower():
            response = image_generation(question)
            print(response)
        else:
            response = ask_openai(question)

            if response:
                chat_history.append({'question': question, 'response': response})
                request.session['chat_history'] = chat_history
                request.session.save()

                # Save user feedback to the database
                is_positive = request.POST.get('is_positive', None)
                print(is_positive)
                save_feedback(request.user.id, question, response, is_positive)

            else:
                response = "we cannot answer"
        return JsonResponse({'response': response, 'chat_history': chat_history})   

    return render(request, 'ai_tutor/staff_chatbot.html',{'all_prompts': all_prompts})

def save_feedback(user_id, question, response, is_positive):
    # Check if is_positive is not None before converting to boolean
    is_positive_bool = is_positive.lower() == 'true' if is_positive is not None else None
    feedback = Feedback(user_id=user_id, question=question, response=response, is_positive=is_positive_bool)
    feedback.save()

@login_required
def profile(request):
    user_pdfs = Pdf_Model.objects.filter(user=request.user)
    return render(request, 'ai_tutor/profile.html', {'user_pdfs': user_pdfs})

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
            is_staff = form.cleaned_data.get('is_staff', False)  # Get the value of the "Is Staff" checkbox
            user.is_staff = is_staff 
            email = form.cleaned_data.get('username')
            user.email = email
            user.save()
            group, created = Group.objects.get_or_create(name='pdfuploadpermissions')
            permissions = Permission.objects.filter(codename__in=['add_pdf', 'change_pdf', 'delete_pdf'])
            group.permissions.set(permissions)
            user.groups.add(group)
            

            messages.success(request, f'Account created for {email}!')
            return redirect('index')
        else:
            return render(request, 'ai_tutor/register.html', {'form': form})
        


# # Assuming you have a specific user to associate with the banned words
# user = User.objects.get(username='admin@gmail.com')  # Replace 'your_username' with the actual username

# url = 'https://gist.githubusercontent.com/jamiew/1112488/raw/7ca9b1669e1c24b27c66174762cb04e14cf05aa7/google_twunter_lol'

# response = requests.get(url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     data = response.text

#     # Extracting JavaScript object using regex
#     match = re.search(r'easterEgg\.BadWorder\.list\s*=\s*({[^;]+})', data)
    
#     if match:
#         json_data = match.group(1)

#         # Replace single quotes with double quotes for both keys and values
#         json_data_fixed = re.sub(r"([a-zA-Z0-9_]+):", r'"\1":', json_data)
        
#         # Parse the corrected JSON data
#         bad_words_dict = json.loads(json_data_fixed)

#         # Create instances of the BannedWord model for each word
#         for word in bad_words_dict.keys():
#             # Check if the word already exists in the database to avoid duplicates
#             if not bannend_word.objects.filter(word=word).exists():
#                 # Create a new BannedWord instance and associate it with the user
#                 banned_words = bannend_word(user=user, word=word)

#                 # Save the instance to the database
#                 banned_words.save()
                
#                 print(f"Banned word '{word}' added to the database for user '{user.username}'.")
#             else:
#                 print(f"Banned word '{word}' already exists in the database.")
#     else:
#         print("Couldn't find the JavaScript object in the response.")
# else:
#     print(f"Failed to retrieve data. Status code: {response.status_code}")


