# AI_Loaner/views.py
import os
import joblib
import pandas as pd
import numpy as np
import shap
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import LoanApplicationSerializer

MODEL_PATH = os.path.join(settings.BASE_DIR, 'loan_model.joblib')

class LoanPredictView(APIView):
    def get(self, request):
        return Response({
            "engine": "Enterprise Loan Underwriting Engine v2.0",
            "features_evaluated": ["Financial Ratios (DTI, LTI)", "CIBIL Bureau Score", "Income Metrics"],
            "status": "Ready for POST requests"
        })

    def post(self, request):
        if not os.path.exists(MODEL_PATH):
            return Response({"error": "Model file not found."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        artifacts = joblib.load(MODEL_PATH)
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_cols = artifacts['features']

        serializer = LoanApplicationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        d = serializer.validated_data

        # 1. Feature Engineering (Pipeline Ratios)
        total_income = d['ApplicantIncome'] + d['CoapplicantIncome']
        term = d['Loan_Amount_Term'] if d['Loan_Amount_Term'] > 0 else 360
        approx_monthly_emi = (d['LoanAmount'] / term) * 1.15
        dti_ratio = round((d['Existing_EMI'] + approx_monthly_emi) / total_income, 4)
        lti_ratio = round(d['LoanAmount'] / (total_income * 12), 4)

        eval_data = {
            'ApplicantIncome': d['ApplicantIncome'],
            'CoapplicantIncome': d['CoapplicantIncome'],
            'LoanAmount': d['LoanAmount'],
            'CIBIL_Score': d['CIBIL_Score'],
            'Loan_Amount_Term': d['Loan_Amount_Term'],
            'Existing_EMI': d['Existing_EMI'],
            'DTI_Ratio': dti_ratio,
            'LTI_Ratio': lti_ratio
        }

        input_df = pd.DataFrame([eval_data])[feature_cols]
        scaled_input = scaler.transform(input_df)

        # 2. Probability & Tiered Verdict
        prob = model.predict_proba(scaled_input)[0][1]
        score_pct = round(float(prob) * 100, 2)

        if score_pct >= 75.0:
            decision = "Approved (STP)"
            tier = "Tier-1 Low Risk"
        elif score_pct >= 45.0:
            decision = "Referred (Manual Review)"
            tier = "Tier-2 Moderate Risk"
        else:
            decision = "Rejected"
            tier = "Tier-3 High Risk"

        # 3. SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)

        if isinstance(shap_values, list):
            class_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, 1]
        else:
            class_shap = shap_values[0]

        factors = []
        for feature, val in zip(feature_cols, class_shap):
            factors.append({
                "feature": feature,
                "impact_score": round(float(val), 4),
                "direction": "Positive" if val > 0 else "Negative",
                "summary": "Improves Creditworthiness" if val > 0 else "Increases Default Risk"
            })

        factors.sort(key=lambda x: abs(x['impact_score']), reverse=True)

        return Response({
            "verdict": decision,
            "risk_tier": tier,
            "confidence_score": score_pct,
            "financial_indicators": {
                "DTI_Ratio": f"{round(dti_ratio * 100, 1)}%",
                "LTI_Ratio": f"{round(lti_ratio, 2)}x",
                "Total_Household_Income": f"₹{int(total_income):,}"
            },
            "shap_attribution": {
                "base_prior": round(float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value), 3),
                "primary_risk_driver": factors[0]["feature"],
                "factors": factors
            }
        }, status=status.HTTP_200_OK)