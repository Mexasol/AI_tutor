from django.urls import path
from . import views
from django.contrib.auth import views as auth_view
from .forms import UserLoginForm,UserPasswordChangeForm,UserPasswordResetForm,UserSetPasswordForm
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='index'),
    path('documents/', views.upload_document, name='upload_documents'),
    path('delete_pdf/<int:pdf_id>/', views.delete_pdf, name='delete_pdf'),
    path('profile/', views.profile, name='profile'),
    path('answer/', views.question_answering, name='answer'),
    path('stats/', views.Academy_stats, name='stats'),
    path('banned_words/', views.Banneds_words, name='banned_words'),
    path('banned_words/delete/<int:banned_word_id>/', views.delete_banned_word, name='delete_banned_word'),
    path('banned_words/edit/<int:banned_word_id>/', views.edit_banned_word, name='edit_banned_word'),
    path('all_prompts/', views.prompt_for_chatbot, name='all_prompts'),
    path('all_prompts/delete/<int:all_prompts_id>/', views.delete_prompt, name='delete_prompt'),
    path('all_prompts/edit/<int:all_prompts_id>/', views.edit_prompt, name='edit_prompt'),
    path('custom_prompts/', views.custom_prompts, name='custom_prompts'),
    path('custom_prompts/edit/<int:all_prompts_id>/', views.edit_custom_prompts, name='edit_custom_prompts'),
    path('staffs/', views.staff_chatbot, name='staffs'),
    path('accounts/login/', auth_view.LoginView.as_view(template_name='ai_tutor/login.html',authentication_form=UserLoginForm) , name='login'),
    # path('accounts/login/', views.UserLoginView.as_view(), name='login'),
    # path('verify-otp/', views.verify_otp, name='verify_otp'),
    path('register/', views.User_Registration_view.as_view(), name='register'),
    path('logout/', auth_view.LogoutView.as_view(next_page='index'), name='logout'),
    # path('logout/', views.CustomLogoutView.as_view(), name='logout'),

    ############################# End Account Login ###################################
    ############################# Change Password #####################################
    path('passwordchange/', auth_view.PasswordChangeView.as_view(template_name='ai_tutor/passwordchange.html',form_class=UserPasswordChangeForm,success_url='/passwordchangedone/'), name='password-change'),
    path('passwordchangedone/', auth_view.PasswordChangeDoneView.as_view(template_name='ai_tutor/passwordchangedone.html'), name='password-change-done'),
    ############################# End Change Password ################################
    ############################# Reset Password #####################################

    path('password-reset/', auth_view.PasswordResetView.as_view(template_name='ai_tutor/password_reset.html',html_email_template_name='ai_tutor/password_reset_email.html',form_class=UserPasswordResetForm), name='password-reset'),
    path('password-reset/done/', auth_view.PasswordResetDoneView.as_view(template_name='ai_tutor/password_reset_done.html'), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', auth_view.PasswordResetConfirmView.as_view(template_name='ai_tutor/password_reset_confirm.html',form_class=UserSetPasswordForm), name='password_reset_confirm'),
    path('password-reset-complete/', auth_view.PasswordResetCompleteView.as_view(template_name='ai_tutor/password_reset_complete.html'), name='password_reset_complete'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
