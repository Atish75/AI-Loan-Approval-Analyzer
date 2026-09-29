# AI_Loaner/serializers.py
from rest_framework import serializers

class LoanApplicationSerializer(serializers.Serializer):
    ApplicantIncome = serializers.FloatField(required=True, min_value=1000.0)
    CoapplicantIncome = serializers.FloatField(default=0.0, min_value=0.0)
    LoanAmount = serializers.FloatField(required=True, min_value=5000.0)
    CIBIL_Score = serializers.IntegerField(required=True, min_value=300, max_value=900)
    Existing_EMI = serializers.FloatField(default=0.0, min_value=0.0)
    Loan_Amount_Term = serializers.FloatField(default=360.0)