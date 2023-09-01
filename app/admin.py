from django.contrib import admin
from . models import Character 
# Register your models here.

@admin.register(Character)
class CharacterModelAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'img_path', 'audio_path']
