from django.db import models

# Create your models here.

class Character(models.Model):
    name = models.CharField(max_length=100)
    img_path = models.CharField(max_length=100)
    audio_path = models.CharField(max_length=300)
