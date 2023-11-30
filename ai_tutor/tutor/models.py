from django.db import models

# Create your models here.

class Pdf_Model(models.Model):
    file = models.FileField(verbose_name='PDFS',upload_to='pdfs/')


    def __str__(self):
        return self.file.name
