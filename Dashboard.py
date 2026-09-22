# CREDIT RISK PREDICTION SYSTEM

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import sqlite3
from datetime import datetime
import time
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import letter

from io import BytesIO


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Credit Risk Assessment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)



# SESSION AUTHENTICATION

if "authenticated" not in st.session_state:

    st.session_state.authenticated = False


# LOGIN SCREEN

if not st.session_state.authenticated:

    st.title("Secure Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if (
            username == "staff"
            and password == "staff123"
        ):

            st.session_state.authenticated = True

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.stop()

    
st.markdown("""

<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f2937;
    font-weight: 700;
}

/* SIDEBAR STYLING */

section[data-testid="stSidebar"] {
    background-color: #1e293b;
}

section[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebarNav"] {
    background-color: #1e293b;
    padding-top: 20px;
}

[data-testid="stSidebarNav"]::before {
    content: "Navigation";
    font-size: 24px;
    font-weight: bold;
    color: white;
    display: block;
    margin-bottom: 20px;
    padding-left: 20px;
}

[data-testid="stSidebarNav"] a {
    background-color: #334155;
    color: white !important;
    border-radius: 10px;
    margin: 8px;
    padding: 10px;
    transition: 0.3s;
}

[data-testid="stSidebarNav"] a:hover {
    background-color: #2563eb;
    color: white !important;
}
        

/* BUTTON STYLING */

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* INPUT BOXES */

.stNumberInput,
.stSelectbox,
.stTextInput {
    border-radius: 10px;
}

/* SUCCESS / INFO BOXES */

.stSuccess {
    border-radius: 12px;
}

.stInfo {
    border-radius: 12px;
}
            
/* LOGOUT BUTTON SIDEBAR STYLE */

section[data-testid="stSidebar"] .stButton > button {
    background-color: #334155;
    color: white;
    border-radius: 10px;
    margin: 8px auto;
    padding: 12px;
    width: 95%;
    border: none;
    font-weight: 600;
    transition: 0.3s;
    display: block;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #b8142f;
    color: white;
}

</style>

""", unsafe_allow_html=True)



# PAGE HEADER

col1, col2 = st.columns([1.25, 6])

with col1:

    st.markdown(
        "<div style='margin-top:15px'></div>",
        unsafe_allow_html=True
    )

    st.image(
        "FinTrust.png",
        width=120
    )

    

with col2:

    st.title(
        "Credit Risk Assessment Dashboard"
    )

    st.write(
        "Predict borrower default risk using "
        "Machine Learning and Explainable AI."
    )

st.markdown("<br>", unsafe_allow_html=True)


# SIDEBAR LOGOUT BUTTON

with st.sidebar:

    st.sidebar.success(
        "Logged in as Staff"
        )

    st.markdown(
        """
        <div style='margin-top:210px;'></div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Logout", use_container_width=True):

        st.session_state.authenticated = False

        st.rerun()


        
# LOAD MODEL FILES

model = joblib.load(
    "xgb_credit_risk_model.pkl"
)

model_features = joblib.load(
    "model_features.pkl"
)

input_template = joblib.load(
    "input_template.pkl"
)


# DATABASE CONNECTION

conn = sqlite3.connect(
    "credit_risk.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor = conn.cursor()

# ADD NEW COLUMN IF NOT EXISTS

try:

    cursor.execute("""

    ALTER TABLE prediction_history

    ADD COLUMN application_status TEXT

    """)

    conn.commit()

except:

    pass


# CREATE TABLE

cursor.execute("""

CREATE TABLE IF NOT EXISTS prediction_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    prediction_time TEXT,

    income REAL,
    credit REAL,
    loan_years REAL,
    interest_rate REAL,
    goods_price REAL,
    loan_purpose TEXT,

    children INTEGER,
    family_members INTEGER,
    employment_years REAL,
    age INTEGER,

    family_status TEXT,
    education TEXT,
    occupation TEXT,

    annuity REAL,
    devs REAL,
    dpi REAL,
    lsr REAL,

    loan_to_income REAL,
    annuity_to_income REAL,
    goods_to_credit REAL,
    income_per_person REAL,
    credit_per_person REAL,

    risk_score REAL,
    risk_level TEXT,
    application_status TEXT,
    interpretation TEXT,
    explanations TEXT

)

""")

conn.commit()


# LOAD SHAP COMPONENTS

fitted_pipeline = (
    model.calibrated_classifiers_[0]
    .estimator
)

preprocessor = (
    fitted_pipeline.named_steps[
        "preprocessor"
    ]
)

xgb_clf = (
    fitted_pipeline.named_steps[
        "model"
    ]
)

explainer = shap.TreeExplainer(
    xgb_clf
)


# USER INPUTS

income = st.number_input(
    "Annual Household Income",
    min_value=1000,
    value=250000
)

credit = st.number_input(
    "Requested Credit Amount",
    min_value=1000,
    value=500000
)

loan_years = st.number_input(
    "Loan Duration (Years)",
    min_value=1,
    value=5
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=1.0,
    value=12.0
)

goods_price = st.number_input(
    "Requested Loan Purpose Amount",
    min_value=1000,
    value=480000
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    [
        "Business",
        "Education",
        "Agriculture",
        "Household",
        "Medical",
        "Personal",
        "Other"
    ]
)
if loan_purpose == "Other":

    other_purpose = st.text_input(
        "Specify Loan Purpose"
    )

children = st.number_input(
    "Number of Children",
    min_value=0,
    value=2
)

family_members = st.number_input(
    "Family Members (Including Children)",
    min_value=1,
    value=4
)

if children >= family_members:

    st.error(
        "Family members must be greater than "
        "number of children."
    )

    st.stop()


employment_years = st.number_input(
    "Continuous Employment Duration (In Years)",
    min_value=0.0,
    value=2.0,
    step=0.5
)

# Convert years to days
days_employed = int(
    employment_years * 365
)

age = st.number_input(
    "Age",
    min_value=18,
    value=35
)

family_status = st.selectbox(
    "Family Status",
    [
        "Married",
        "Single / not married",
        "Civil marriage",
        "Separated",
        "Widow"
        
    ]
)

education = st.selectbox(
    "Education Level",
    [
        "No Formal Education",
        "Primary Education",
        "Secondary Education",
        "Higher Education"
    ]
)


occupation = st.selectbox(
    "Occupation Type",
    [
         "Small Business Owner",
        "Self-Employed",
        "Vendor / Trader",
        "Laborer",
        "Driver",
        "Tailor / Dressmaker",
        "Factory Worker",
        "Sales Worker",
        "Food Service Worker",
        "Unemployed",
        "Other"
    ]
)

if occupation == "Other":

    other_occupation = st.text_input(
        "Specify Occupation"
    )



# Monthly interest rate calculation

monthly_rate = (
    interest_rate / 100
) / 12

# Total monthly payments
months = loan_years * 12

# EMI calculation
if monthly_rate > 0:

    monthly_payment = (
        credit
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / (
        ((1 + monthly_rate) ** months) - 1
    )

else:

    monthly_payment = credit / months

# Convert to annual repayment
annuity = monthly_payment * 12

# Display calculated repayment
st.write(
    f"Estimated Annual Loan Repayment: "
    f"{annuity:,.2f}"
)



# FEATURE LABELS

FEATURE_LABELS = {

    "AMT_ANNUITY":
    "annual loan repayment",

    "AMT_GOODS_PRICE":
    "loan amount for purchased goods",

    "AMT_CREDIT":
    "total loan amount",

    "AMT_INCOME_TOTAL":
    "reported household income",

    "DAYS_EMPLOYED":
    "employment duration",

    "AGE":
    "age",

    "CNT_CHILDREN":
    "number of children",

    "CNT_FAM_MEMBERS":
    "household size",

    "DEVS":
    "income volatility",

    "LSR":
    "loan repayment burden",

    "DPI":
    "household dependency pressure",

    "NAME_FAMILY_STATUS":
    "family status",

    "NAME_EDUCATION_TYPE":
    "education level",

    "OCCUPATION_TYPE":
    "occupation type"
}



# HUMAN EXPLANATION FUNCTION

def human_explanation(
    feature,
    value,
    shap_value
):

    increase = shap_value > 0


 
    # EMPLOYMENT HISTORY

    if feature == "employment duration":

        years = round(value / 365, 1)

        if value < 365:

            return (
                f"The applicant has a short employment "
                f"history ({years} years), which may indicate "
                f"limited income stability and increased "
                f"repayment risk."
            )

        elif value < 1825:

            return (
                f"The applicant demonstrates moderate "
                f"employment history ({years} years), "
                f"indicating average employment stability."
            )

        else:

            return (
                f"The applicant demonstrates stable "
                f"long-term employment history "
                f"({years} years), supporting stronger "
                f"repayment capability."
            )


   
    # INCOME VOLATILITY

    elif feature == "income volatility":

        if value > 0.40:

            return (
                f"The applicant demonstrates relatively "
                f"high income volatility "
                f"(DEVS = {value:.2f}), which may reduce "
                f"financial stability."
            )

        else:

            return (
                f"The applicant demonstrates relatively "
                f"stable income patterns "
                f"(DEVS = {value:.2f}), supporting "
                f"consistent repayment capacity."
            )


    # LOAN REPAYMENT BURDEN

    elif feature == "loan repayment burden":

        if value > 0.40:

            return (
                f"A high proportion of the applicant's "
                f"annual income is allocated toward "
                f"loan repayments (LSR = {value:.2f}), "
                f"which may increase repayment pressure."
            )

        elif value > 0.20:

            return (
                f"The applicant demonstrates a moderate "
                f"annual repayment burden "
                f"(LSR = {value:.2f})."
            )

        else:

            return (
                f"The applicant demonstrates a manageable "
                f"annual repayment burden "
                f"(LSR = {value:.2f}), supporting "
                f"repayment affordability."
            )



    # DEPENDENCY PRESSURE

    elif feature == "household dependency pressure":

        if value > 1.0:

            return (
                f"The applicant supports multiple "
                f"dependents (DPI = {value:.2f}), "
                f"which may increase household "
                f"financial obligations."
            )

        else:

            return (
                f"The applicant demonstrates relatively "
                f"low household dependency pressure "
                f"(DPI = {value:.2f}), supporting "
                f"repayment capacity."
            )



    # HOUSEHOLD INCOME

    elif feature == "reported household income":

        if value < 120000:

            return (
                f"The applicant reports relatively low "
                f"annual household income ({int(value)}), "
                f"which may reduce repayment affordability."
            )

        elif value < 300000:

            return (
                f"The applicant demonstrates moderate "
                f"annual household income ({int(value)}), "
                f"supporting average repayment capacity."
            )

        else:

            return (
                f"The applicant demonstrates strong "
                f"annual household income ({int(value)}), "
                f"supporting repayment capability."
            )



    # ANNUAL LOAN REPAYMENT

    elif feature == "annual loan repayment":

        if value > 150000:

            return (
                f"The applicant has a relatively high "
                f"annual loan repayment obligation "
                f"({int(value)}), which may increase "
                f"financial repayment pressure."
            )

        elif value > 60000:

            return (
                f"The applicant demonstrates a moderate "
                f"annual loan repayment obligation "
                f"({int(value)}), requiring stable "
                f"financial capacity."
            )

        else:

            return (
                f"The applicant's annual repayment "
                f"obligation ({int(value)}) appears "
                f"manageable relative to income."
            )



    # TOTAL LOAN AMOUNT

    elif feature == "total loan amount":

        if value > 1000000:

            return (
                f"The requested loan amount is relatively "
                f"large ({int(value)}), increasing overall "
                f"repayment exposure."
            )

        else:

            return (
                f"The requested loan amount "
                f"({int(value)}) appears manageable "
                f"within the applicant's financial capacity."
            )


    # EDUCATION LEVEL

    elif feature == "education level":

        education_value = str(value)

        if education_value == "Higher Education":

            return (
                f"The applicant demonstrates higher "
                f"educational attainment, which is commonly "
                f"associated with improved employment "
                f"stability and repayment capability."
            )

        elif education_value == "Secondary Education":

            return (
                f"The applicant demonstrates secondary-level "
                f"education, representing moderate financial "
                f"and employment stability."
            )

        elif education_value == "Primary Education":

            return (
                f"The applicant demonstrates basic educational "
                f"attainment, which may limit access to stable "
                f"income opportunities."
            )

        elif education_value == "No Formal Education":

            return (
                f"The applicant demonstrates limited formal "
                f"educational background, which may increase "
                f"financial vulnerability and repayment uncertainty."
            )

        else:

            return (
                f"The applicant's educational background "
                f"contributes to the overall credit "
                f"risk evaluation."
            )


    # OCCUPATION TYPE

    elif feature == "occupation type":

        occupation_value = str(value)

        stable_jobs = [
            "Factory Worker",
            "Sales Worker",
            "Small Business Owner"
        ]

        moderate_jobs = [
            "Driver",
            "Vendor / Trader",
            "Tailor / Dressmaker",
            "Food Service Worker",
            "Self-Employed"
        ]

        high_risk_jobs = [
            "Laborer",
            "Unemployed"
        ]

        if occupation_value in stable_jobs:

            return (
                f"The applicant's occupation demonstrates "
                f"relatively stable income conditions and "
                f"consistent repayment capability."
            )

        elif occupation_value in moderate_jobs:

            return (
                f"The applicant's occupation demonstrates "
                f"moderate income stability according to "
                f"historical borrower patterns."
            )

        elif occupation_value in high_risk_jobs:

            return (
                f"The applicant's occupation may involve "
                f"seasonal or variable income conditions, "
                f"which can influence repayment stability."
            )

        elif occupation_value in high_risk_jobs:

            return (
                f"The applicant's occupation profile may "
                f"indicate unstable or limited income "
                f"conditions, which can influence "
                f"repayment stability."
            )


    # AGE

    elif feature == "age":

        if value < 25:

            return (
                f"The applicant belongs to a younger "
                f"age group ({int(value)} years), which "
                f"may demonstrate limited credit history."
            )

        elif value < 55:

            return (
                f"The applicant belongs to a financially "
                f"stable working-age category "
                f"({int(value)} years)."
            )

        else:

            return (
                f"The applicant belongs to an older age "
                f"group ({int(value)} years), which may "
                f"influence long-term repayment capacity."
            )


    # DEFAULT

    else:

        return (
            f"This financial factor contributed "
            f"to the overall credit risk evaluation."
        )


# GENERATE HUMAN EXPLANATIONS

def generate_human_explanations(
    shap_vals,
    feature_names,
    input_df,
    top_k=20
):

    explanations = []

    shap_df = pd.DataFrame({

        "Feature": feature_names,
        "SHAP": shap_vals

    })

    shap_df["ABS_SHAP"] = (
        shap_df["SHAP"].abs()
    )

    shap_df = shap_df.sort_values(
        "ABS_SHAP",
        ascending=False
    ).head(top_k)

    for _, row in shap_df.iterrows():

        raw_feature = row["Feature"]

        clean_feature = (
            raw_feature
            .replace("num__", "")
            .replace("cat__", "")
        )

        if clean_feature.startswith(
            "NAME_FAMILY_STATUS"
        ):

            clean_feature = (
                "NAME_FAMILY_STATUS"
            )

        elif clean_feature.startswith(
            "NAME_EDUCATION_TYPE"
        ):

            clean_feature = (
                "NAME_EDUCATION_TYPE"
            )

        elif clean_feature.startswith(
            "OCCUPATION_TYPE"
        ):

            occupation_name = (
                clean_feature
                .replace("OCCUPATION_TYPE_", "")
            )

            clean_feature = (
                "OCCUPATION_TYPE"
            )

            value = occupation_name

        feature_label = FEATURE_LABELS.get(
            clean_feature,
            clean_feature
        )

        if "value" not in locals():

            if clean_feature in input_df.columns:

                value = (
                    input_df.iloc[0][clean_feature]
                )

            else:

                value = 0

        explanation = human_explanation(
            feature_label,
            value,
            row["SHAP"]
        )

        if "occupation_name" in locals():
            del occupation_name

        if "value" in locals():
            del value

        if (
            explanation not in explanations
            and "This financial factor" not in explanation
        ):

            explanations.append(
                explanation
            )

        if len(explanations) >= 7:

            break

    return explanations



# INTERPRETATION FUNCTION

def generate_interpretation(
    risk_level
):

    if risk_level == "Low Risk":

        return (
            "The applicant demonstrates relatively "
            "stable annual financial characteristics "
            "and manageable repayment capacity."
        )

    elif risk_level == "Medium Risk":

        return (
            "The applicant demonstrates moderate "
            "annual repayment risk with some "
            "financial pressure indicators."
        )

    else:

        return (
            "The applicant demonstrates elevated "
            "financial risk indicators and increased "
            "annual repayment uncertainty."
        )



# PDF REPORT GENERATOR

def generate_pdf_report(
    risk_score,
    risk_level,
    interpretation,
    explanations,
    income,
    credit,
    loan_years,
    interest_rate,
    occupation,
    education,
    age,
    application_id,
    application_status,
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=10
    )

    risk_style = ParagraphStyle(
        "RiskStyle",
        parent=styles["Heading1"],
        alignment=1
    )

    elements = []

    # TITLE

    title = Paragraph(
        "<b>Credit Risk Assessment Report</b>",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    # APPLICANT DETAILS

    elements.append(
        Paragraph(
            "Applicant Information",
            heading_style
        )
    )

    applicant_table = Table([
        ["Annual Income", f"{income:,.2f}"],
        ["Requested Credit", f"{credit:,.2f}"],
        ["Loan Duration", f"{loan_years} Years"],
        ["Interest Rate", f"{interest_rate}%"],
        ["Occupation", occupation],
        ["Education", education],
        ["Age", age]
    ])

    applicant_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold")
    ]))

    elements.append(applicant_table)
    elements.append(Spacer(1,15))

    # Risk banner
    risk_color = colors.green

    if risk_level == "Medium Risk":
        risk_color = colors.orange

    elif risk_level == "High Risk":
        risk_color = colors.red

    elements.append(
        Paragraph(
            f'<font color="{risk_color}"><b>{risk_level}</b></font>',
            risk_style
        )
    )

    # PREDICTION RESULTS

    elements.append(
        Paragraph(
            "Assessment Summary",
            heading_style
        )
    )

    summary_table = Table([
        ["Application ID", application_id],
        ["Risk Probability", f"{risk_score*100:.1f}%"],
        ["Risk Level", risk_level],
        ["Recommendation", application_status]
    ])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold")
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1,15))

    # EXPLANATION FACTORS

    explanation_text = (
        "<b>Key Explanation Factors</b><br/><br/>"
    )

    for e in explanations:

        explanation_text += f"• {e}<br/><br/>"

    elements.append(
        Paragraph(
            explanation_text,
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 20))

    # FOOTER

    footer = Paragraph(
        f"""
        Report Generated:
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """,
        styles['Italic']
    )

    elements.append(footer)

    # BUILD PDF

    doc.build(elements)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf



# PREDICTION BUTTON

if st.button("Predict Credit Risk"):

    with st.spinner(
        "Analyzing borrower financial profile..."
    ):

        time.sleep(4)

        input_df = input_template.copy()

        input_df.loc[:, "AMT_INCOME_TOTAL"] = income
        input_df.loc[:, "AMT_CREDIT"] = credit
        input_df.loc[:, "AMT_ANNUITY"] = annuity
        input_df.loc[:, "AMT_GOODS_PRICE"] = goods_price
        input_df.loc[:, "CNT_CHILDREN"] = children
        input_df.loc[:, "CNT_FAM_MEMBERS"] = family_members
        input_df.loc[:, "DAYS_EMPLOYED"] = days_employed
        input_df.loc[:, "AGE"] = age
        input_df.loc[:, "NAME_FAMILY_STATUS"] = family_status
        input_df.loc[:, "NAME_EDUCATION_TYPE"] = education
        input_df.loc[:, "OCCUPATION_TYPE"] = occupation



    # ESTIMATED INCOME VOLATILITY

    # Base volatility score

    devs = 0.50


    # Occupation-based adjustment

    stable_jobs = [
        "Factory Worker",
        "Sales Worker",
        "Small Business Owner"
    ]

    moderate_jobs = [
        "Driver",
        "Vendor / Trader",
        "Tailor / Dressmaker",
        "Food Service Worker",
        "Self-Employed"
    ]

    high_risk_jobs = [
        "Laborer",
        "Unemployed"
    ]


    if occupation in stable_jobs:

        devs -= 0.20

    elif occupation in moderate_jobs:

        devs -= 0.10

    elif occupation in high_risk_jobs:

        devs += 0.15


    # Employment stability adjustment

    if employment_years >= 5:

        devs -= 0.10

    elif employment_years < 1:

        devs += 0.10


    # Education adjustment

    if education == "Higher Education":

        devs -= 0.10

    elif education == "No Formal Education":

        devs += 0.10


    # Keep value within realistic range

    devs = max(
        0.10,
        min(devs, 1.00)
    )

    input_df.loc[:, "DEVS"] = devs



    income_earners = max(
        family_members - children,
        1
    )

    input_df.loc[:, "DPI"] = (
        children / (income_earners + 1)
    )

    input_df.loc[:, "LSR"] = (
        annuity / income
    )

    input_df.loc[:, "LOAN_TO_INCOME"] = (
        credit / income
    )

    input_df.loc[:, "ANNUITY_TO_INCOME"] = (
        annuity / income
    )

    input_df.loc[:, "GOODS_TO_CREDIT"] = (
        goods_price / credit
    )

    input_df.loc[:, "INCOME_PER_PERSON"] = (
        income / family_members
    )

    input_df.loc[:, "CREDIT_PER_PERSON"] = (
        credit / family_members
    )

    input_df = input_df[
        model_features
    ]


    # PREDICTION

    risk_score = model.predict_proba(
        input_df
    )[:, 1][0]




    # RISK CATEGORY

    if risk_score < 0.40:

        risk_level = "Low Risk"

    elif risk_score < 0.70:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"



    # APPLICATION STATUS

    if risk_level == "Low Risk":

        application_status = (
            "Approved"
        )

    elif risk_level == "Medium Risk":

        application_status = (
            "Review Required"
        )

    else:

        application_status = (
            "Rejected"
        )



    # SHAP EXPLAINABILITY

    input_transformed = (
        preprocessor.transform(input_df)
    )

    shap_values = explainer.shap_values(
        input_transformed
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    explanations = (
        generate_human_explanations(
            shap_values[0],
            feature_names,
            input_df
        )
    )

    interpretation = (
        generate_interpretation(
            risk_level
        )
    )

   
    # SAVE PREDICTION TO DATABASE

    explanations_text = " | ".join(
        explanations
    )

    cursor.execute("""

    INSERT INTO prediction_history (

        prediction_time,

        income,
        credit,
        loan_years,
        interest_rate,
        goods_price,
        loan_purpose,

        children,
        family_members,
        employment_years,
        age,

        family_status,
        education,
        occupation,

        annuity,
        devs,
        dpi,
        lsr,

        loan_to_income,
        annuity_to_income,
        goods_to_credit,
        income_per_person,
        credit_per_person,

        risk_score,
        risk_level,
        application_status,
        interpretation,
        explanations

    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        income,
        credit,
        loan_years,
        interest_rate,
        goods_price,
        loan_purpose,

        children,
        family_members,
        employment_years,
        age,

        family_status,
        education,
        occupation,

        annuity,
        devs,
        input_df["DPI"].iloc[0],
        input_df["LSR"].iloc[0],

        input_df["LOAN_TO_INCOME"].iloc[0],
        input_df["ANNUITY_TO_INCOME"].iloc[0],
        input_df["GOODS_TO_CREDIT"].iloc[0],
        input_df["INCOME_PER_PERSON"].iloc[0],
        input_df["CREDIT_PER_PERSON"].iloc[0],

        float(risk_score),
        risk_level,
        application_status,
        interpretation,
        explanations_text

    ))


    application_id = cursor.lastrowid

    conn.commit()




# DISPLAY RESULTS

    st.subheader("Prediction Result")

   
    # RISK SCORE DISPLAY CARD

    if risk_level == "Low Risk":

        risk_bg = "#d4edda"
        risk_color = "#155724"

    elif risk_level == "Medium Risk":

        risk_bg = "#fff3cd"
        risk_color = "#856404"

    else:

        risk_bg = "#f8d7da"
        risk_color = "#721c24"  

    st.components.v1.html(
        f"""
        <div style="
            background-color: {risk_bg};
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
            font-family: Arial;
        ">

            <h2 style="
                color: {risk_color};
                margin-bottom: 10px;
            ">
                Risk Probability
            </h2>

            <h1 style="
                color: {risk_color};
                font-size: 42px;
                margin-bottom: 10px;
            ">
                {risk_score * 100:.1f}%
            </h1>

            <h3 style="
                color: {risk_color};
            ">
                {risk_level}
            </h3>

        </div>
        """,
        height=220
    )

    

    # ASSESSMENT SUMMARY CARDS

    st.markdown("### Assessment Summary")

    col1, col2, col3, col4 = st.columns(4)

    card_style = """
    background-color:#f3f4f6;
    padding:12px;
    border-radius:10px;
    text-align:center;
    border:1px solid #d1d5db;
    """

    with col1:

        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:13px;">
                Risk Probability
            </div>
            <div style="font-size:20px;
                        font-weight:700;">
                {risk_score * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:13px;">
                Annual Repayment
            </div>
            <div style="font-size:20px;
                        font-weight:700;">
                {annuity:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:13px;">
                Repayment Burden
            </div>
            <div style="font-size:20px;
                        font-weight:700;">
                {(annuity / income) * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:13px;">
                Recommendation
            </div>
            <div style="font-size:20px;
                        font-weight:700;">
                {application_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    




    # OVERALL INTERPRETATION

    st.subheader(
        "Overall Interpretation"
    )

    st.write(
        interpretation
    )


    # KEY EXPLANATION FACTORS

    st.subheader(
        "Key Explanation Factors"
    )

    for e in explanations:

        st.write(f"• {e}")



    # PDF REPORT DOWNLOAD

    pdf_report = generate_pdf_report(
        risk_score,
        risk_level,
        interpretation,
        explanations,
        income,
        credit,
        loan_years,
        interest_rate,
        occupation,
        education,
        age,
        application_id,
        application_status,
        
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_report,
        file_name="credit_risk_report.pdf",
        mime="application/pdf"
    )


# FOOTER

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center;
                color: #6b7280;
                font-size: 14px;
                padding-top: 10px;
                padding-bottom: 10px;'>

    <b>AI-Powered Credit Risk Assessment System</b><br>

    An Explainable Machine Learning Framework for Credit Risk Assessment
    in FinTrust Microfinance Institution Using Field Estimable Financial Indicators.<br><br>

    Supporting intelligent borrower risk evaluation and sustainable lending decision-making in microfinance institutions.

    </div>
    """,
    unsafe_allow_html=True
)