from django.urls import path
from . import views
from django.contrib.auth import views as auth_view
from .forms import UserLoginForm,UserPasswordChangeForm,UserPasswordResetForm,UserSetPasswordForm
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='index'),
    path('profile/', views.profile, name='profile'),
    path('answer/', views.question_answering, name='answer'),
    path('accounts/login/', auth_view.LoginView.as_view(template_name='ai_tutor/login.html',authentication_form=UserLoginForm) , name='login'),
    path('register/', views.User_Registration_view.as_view(), name='register'),
    path('logout/', auth_view.LogoutView.as_view(next_page='index'), name='logout'),
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