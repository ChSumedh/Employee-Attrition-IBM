from flask import Flask,render_template,request
from joblib import load
import pandas as pd
import numpy as np
app = Flask(__name__,template_folder='templates',static_folder='static')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/add/<n1>/<n2>")
def add(n1,n2):
    sum=int(n1)+int(n2)
    return f"{sum}"

@app.route("/form")
def form():
    return render_template('form.html')

@app.route("/model",methods=["POST"])
def result():
    model=load("model/model.pkl")
    imputer_num=load("model/imputer_num.pkl")
    imputer_cat=load("model/inputer_cat.pkl")
    scaler=load("model/scaler.pkl")
    encoder=load("model/encoder.pkl")

    import numpy as np

    def get_float(field):
        val = request.form.get(field)
        return float(val) if val not in [None, ""] else np.nan

    def get_str(field):
        val = request.form.get(field)
        return val if val not in [None, ""] else np.nan

    monthly_income = get_float("MonthlyIncome")

    daily_rate = monthly_income / 30 if not np.isnan(monthly_income) else np.nan
    hourly_rate = daily_rate / 8 if not np.isnan(daily_rate) else np.nan

    num_data = {
        "Age": get_float("Age"),
        "DailyRate": daily_rate,
        "DistanceFromHome": get_float("DistanceFromHome"),
        "Education": get_float("Education"),
        "EnvironmentSatisfaction": get_float("EnvironmentSatisfaction"),
        "HourlyRate": hourly_rate,
        "JobInvolvement": get_float("JobInvolvement"),
        "JobLevel": get_float("JobLevel"),
        "JobSatisfaction": get_float("JobSatisfaction"),
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": get_float("NumCompaniesWorked"),
        "PercentSalaryHike": get_float("PercentSalaryHike"),
        "RelationshipSatisfaction": get_float("RelationshipSatisfaction"),
        "StockOptionLevel": get_float("StockOptionLevel"),
        "TotalWorkingYears": get_float("TotalWorkingYears"),
        "TrainingTimesLastYear": get_float("TrainingTimesLastYear"),
        "WorkLifeBalance": get_float("WorkLifeBalance"),
        "YearsAtCompany": get_float("YearsAtCompany"),
        "YearsInCurrentRole": get_float("YearsInCurrentRole"),
        "YearsSinceLastPromotion": get_float("YearsSinceLastPromotion"),
        "YearsWithCurrManager": get_float("YearsWithCurrManager")
    }
    num_df = pd.DataFrame([num_data]).astype(float)

    cat_data = {
        "BusinessTravel": get_str("BusinessTravel"),
        "Department": get_str("Department"),
        "EducationField": get_str("EducationField"),
        "Gender": get_str("Gender"),
        "JobRole": get_str("JobRole"),
        "MaritalStatus": get_str("MaritalStatus"),
        "OverTime": get_str("Overtime")
    }

    cat_df=pd.DataFrame([cat_data])

    num_df=pd.DataFrame(
        imputer_num.transform(num_df),
        columns=num_df.columns
    )

    cat_df=pd.DataFrame(
        imputer_cat.transform(cat_df),
        columns=cat_df.columns
    )

    num_df=pd.DataFrame(
        scaler.transform(num_df),
        columns=num_df.columns
    )

    cat_df=pd.DataFrame(
        encoder.transform(cat_df),
        columns=encoder.get_feature_names_out(cat_df.columns)
    )

    final_input = pd.concat([num_df, cat_df], axis=1)

    prediction = model.predict(final_input)[0]
    prob=model.predict_proba(final_input)
    
    return render_template("result.html",pred=prediction,prob=prob[0][1]*100)
if __name__ == "__main__":
    app.run(debug=True)