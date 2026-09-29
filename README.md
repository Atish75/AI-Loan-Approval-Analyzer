# LoanAI - Enterprise Loan Underwriting & Explainability Engine

An automated, regulatory-compliant Machine Learning microservice built with **Django REST Framework (DRF)**, **Scikit-learn**, and **SHAP (SHapley Additive exPlanations)**. 

This service evaluates credit risk using derived financial ratios (DTI, LTI), CIBIL scores, and delivers straight-through processing (STP) decisions alongside local feature attributions for adverse action reporting.

---

## 🏛️ System Architecture

* **Primary Backend (Spring Boot):** Handles user authentication, database persistence, and client business logic.
* **ML Microservice (This Repository):** Exposes stateless prediction endpoints for algorithmic risk scoring and SHAP explainability.
* **Frontend (React.js):** Consumes decision payloads to render interactive underwriting dashboards and visual attribution charts.

---

## 🚀 Key Features

* **Real-Time Underwriting Engine:** Random Forest Classifier trained on key financial parameters and synthetic bureau behaviors.
* **Derived Financial Risk Ratios:** Automatic calculation of:
  * **Debt-to-Income (DTI):** Monthly debt obligation vs total household income.
  * **Loan-to-Income (LTI):** Multiplier of requested loan relative to annual earnings.
* **Tri-Tier Decisioning:**
  * `Approved (STP)`: $\ge 75\%$ probability (Low risk automated disbursal).
  * `Referred (Manual Review)`: $45\% - 74\%$ probability (Referred to credit underwriter).
  * `Rejected`: $< 45\%$ probability (High risk default trajectory).
* **Explainable AI (SHAP):** Computes exact percentage attribution per feature using `shap.TreeExplainer` for adverse action notices.

---

## 📡 API Contract (For Spring Boot & React)

### Endpoint
`POST /predict/`

### Headers
```http
Content-Type: application/json