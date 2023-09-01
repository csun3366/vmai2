from django.urls import path
from . import views

urlpatterns = [
    path("", views.home),
    path("characters", views.CharactersView.as_view(), name="characters"),
    path("checkout", views.CheckoutView.as_view(), name="checkout"),
    path("transform/<int:pk>", views.CharacterTransform.as_view(), name="characterId"),
    path("fetch", views.Fetch.as_view(), name="fetchId"),
    path("convert", views.convert, name="convert"),
]
