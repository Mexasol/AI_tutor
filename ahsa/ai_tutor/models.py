from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Pdf_Model(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)    
    file = models.FileField(verbose_name='PDFS',upload_to='pdfs/')


    def __str__(self):
        return f"{self.user.username}'s PDF"
    

class bannend_word(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    word = models.TextField()

    # length of word
    def __len__(self):
        return len(self.word)
    def __str__(self):
        return f"{self.word}"
    



class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    question = models.TextField(null=True, blank=True)
    response = models.TextField(null=True, blank=True)
    is_positive = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}"

    class Meta:
        ordering = ['-created_at']


class prompts(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.prompt[:50]}"

class Custom_Prompts(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    custom_prompt = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.custom_prompt[:50]}"