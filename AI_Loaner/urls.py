# AI_Loaner/urls.py
from django.urls import path
from .views import LoanPredictView

urlpatterns = [
    path('predict/', LoanPredictView.as_view(), name='loan_predict'),
]