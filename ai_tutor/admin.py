from django.contrib import admin
from .models import Pdf_Model,bannend_word,Feedback,Custom_Prompts

# Register your models here.

class Pdf_Admin(admin.ModelAdmin):
    list_display = ['file','user']
class bannend_word_Admin(admin.ModelAdmin):
    list_display = ['word','user']

admin.site.register(Custom_Prompts)
admin.site.register(Feedback)
admin.site.register(Pdf_Model, Pdf_Admin)
admin.site.register(bannend_word, bannend_word_Admin)

