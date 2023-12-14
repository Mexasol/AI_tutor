from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Pdf_Model(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)    
    file = models.FileField(verbose_name='PDFS',upload_to='pdfs/')


    def __str__(self):
        return f"{self.user.username}'s PDF"
