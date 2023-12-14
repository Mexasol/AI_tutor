from django.contrib import admin
from .models import Pdf_Model

class Pdf_Admin(admin.ModelAdmin):
    list_display = ['file']

admin.site.register(Pdf_Model,Pdf_Admin)



