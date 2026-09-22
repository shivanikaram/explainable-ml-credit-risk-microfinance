import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Applicants History",
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
            username == "admin"
            and password == "admin123"
        ):

            st.session_state.authenticated = True

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.stop()
    


# CUSTOM STYLING

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
}

[data-testid="stSidebarNav"] a:hover {
    background-color: #2563eb;
    color: white !important;
}

/* DATAFRAME */

.stDataFrame {
    border-radius: 12px;
}

/* METRIC CARDS */

.metric-card {
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}

.total-card {
    background-color: #dbeafe;
}

.low-card {
    background-color: #d4edda;
}

.medium-card {
    background-color: #fff3cd;
}

.high-card {
    background-color: #f8d7da;
}

.metric-title {
    font-size: 16px;
    color: #6b7280;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #111827;
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
        "Borrower Assessment History"
    )

    st.write(
        "Historical borrower credit risk "
        "assessment records and prediction history."
    )

st.markdown("<br>", unsafe_allow_html=True)


# DATABASE CONNECTION

conn = sqlite3.connect(
    "credit_risk.db",
    check_same_thread=False
)


# LOAD DATA

query = """

SELECT *

FROM prediction_history

ORDER BY id DESC

"""

df = pd.read_sql_query(
    query,
    conn
)


# DASHBOARD METRICS

total_applicants = len(df)

high_risk = len(
    df[df["risk_level"] == "High Risk"]
)

medium_risk = len(
    df[df["risk_level"] == "Medium Risk"]
)

low_risk = len(
    df[df["risk_level"] == "Low Risk"]
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(f"""

    <div class="metric-card total-card">
        <div class="metric-title">
            Total Applicants
        </div>
        <div class="metric-value">
            {total_applicants}
        </div>
    </div>

    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""

    <div class="metric-card low-card">
        <div class="metric-title">
            Low Risk
        </div>
        <div class="metric-value">
            {low_risk}
        </div>
    </div>

    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""

    <div class="metric-card medium-card">
        <div class="metric-title">
            Medium Risk
        </div>
        <div class="metric-value">
            {medium_risk}
        </div>
    </div>

    """, unsafe_allow_html=True)

with col4:

    st.markdown(f"""

    <div class="metric-card high-card">
        <div class="metric-title">
            High Risk
        </div>
        <div class="metric-value">
            {high_risk}
        </div>
    </div>

    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# RENAME COLUMNS

df = df.rename(columns={

    "id":
    "Applicant ID",

    "prediction_time":
    "Assessment Time",

    "income":
    "Annual Income",

    "credit":
    "Requested Credit",

    "loan_years":
    "Loan Duration (Years)",

    "interest_rate":
    "Interest Rate (%)",

    "goods_price":
    "Requested Loan Purpose Amount",

    "loan_purpose":
    "Loan Purpose",

    "children":
    "Children",

    "family_members":
    "Family Members",

    "employment_years":
    "Continuous Employment Duration",

    "age":
    "Applicant Age",

    "family_status":
    "Family Status",

    "education":
    "Education Level",

    "occupation":
    "Occupation Type",

    "annuity":
    "Estimated Annual Repayment",

    "devs":
    "Income Volatility",

    "dpi":
    "Dependency Pressure",

    "lsr":
    "Repayment Burden",

    "loan_to_income":
    "Loan-To-Income Ratio",

    "annuity_to_income":
    "Repayment-To-Income Ratio",

    "goods_to_credit":
    "Goods-To-Credit Ratio",

    "income_per_person":
    "Income Availability Per Household Member",

    "credit_per_person":
    "Household Credit Burden",

    "risk_score":
    "Risk Probability",

    "risk_level":
    "Risk Category",

    "application_status":
    "Application Status",

    "interpretation":
    "Overall Interpretation",

    "explanations":
    "Key Explanation Factors"

})


# CREATE COPY FOR VISUALIZATIONS

chart_df = df.copy()


# FORMAT PERCENTAGES

percentage_columns = [

    "Risk Probability",
    "Income Volatility",
    "Repayment Burden",
    "Repayment-To-Income Ratio"

]

for col in percentage_columns:

    df[col] = (
        df[col]
        .fillna(0)
        * 100
    ).round(1).astype(str) + "%"


# FORMAT DECIMAL RATIOS

ratio_columns = [

    "Dependency Pressure",
    "Loan-To-Income Ratio",
    "Goods-To-Credit Ratio"

]

for col in ratio_columns:

    df[col] = (
        df[col]
        .fillna(0)
        .round(2)
    )


# FORMAT NORMAL NUMERIC COLUMNS

normal_number_columns = [

    "Annual Income",
    "Requested Credit",
    "Loan Duration (Years)",
    "Interest Rate (%)",
    "Requested Loan Purpose Amount",
    "Estimated Annual Repayment",
    "Income Availability Per Household Member",
    "Household Credit Burden"

]

for col in normal_number_columns:

    if col in df.columns:

        df[col] = (
            df[col]
            .fillna(0)
            .map(lambda x: f"{x:.2f}")
        )


# VISUAL ANALYTICS

st.subheader("Borrower Assessment Insights")

col1, col2 = st.columns(2)


# RISK CATEGORY DISTRIBUTION

with col1:

    risk_counts = (
        chart_df["Risk Category"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Category",
        "Count"
    ]

    fig_risk = px.pie(
        risk_counts,
        names="Risk Category",
        values="Count",
        hole=0.5,
        title="Risk Category Distribution"
    )

    fig_risk.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


# LOAN PURPOSE DISTRIBUTION

with col2:

    purpose_counts = (
        chart_df["Loan Purpose"]
        .value_counts()
        .reset_index()
    )

    purpose_counts.columns = [
        "Loan Purpose",
        "Count"
    ]

    fig_purpose = px.bar(
        purpose_counts,
        x="Loan Purpose",
        y="Count",
        title="Loan Purpose Distribution"
    )

    fig_purpose.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_purpose,
        use_container_width=True
    )


# INCOME VS CREDIT ANALYSIS

fig_income = px.line(
    chart_df.sort_values("Annual Income"),
    x="Annual Income",
    y="Requested Credit",
    color="Risk Category",
    markers=True,
    title="Income vs Requested Credit"
)

fig_income.update_layout(
    height=500
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)


# AGE DISTRIBUTION

fig_age = px.histogram(
    chart_df,
    x="Applicant Age",
    color="Risk Category",
    title="Applicant Age Distribution",
    nbins=20
)

fig_age.update_layout(
    height=450
)

st.plotly_chart(
    fig_age,
    use_container_width=True
)


# RISK COLOR STYLING

def highlight_risk(val):

    if val == "High Risk":

        return (
            "background-color: #f8d7da; "
            "color: #721c24; "
            "font-weight: bold;"
        )

    elif val == "Medium Risk":

        return (
            "background-color: #fff3cd; "
            "color: #856404; "
            "font-weight: bold;"
        )

    elif val == "Low Risk":

        return (
            "background-color: #d4edda; "
            "color: #155724; "
            "font-weight: bold;"
        )

    return ""


# SEARCH HIGHLIGHTING

def highlight_search(val):

    if search:

        if search.lower() in str(val).lower():

            return (
                "background-color: yellow; "
                "color: black;"
            )

    return ""


# APPLY TABLE STYLING

styled_df = (
    df.style
    .map(
        highlight_risk,
        subset=["Risk Category"]
    )
    .map(highlight_search)
)


# DISPLAY TABLE

st.subheader("Applicant Assessment Records")


# SEARCH FILTER

search = st.text_input(
    "Search Applicant Details"
)

filtered_df = df.copy()

if search:

    filtered_df = filtered_df[
        filtered_df.astype(str)
        .apply(
            lambda row:
            row.str.contains(
                search,
                case=False,
                na=False
            ).any(),
            axis=1
        )
    ]

df = filtered_df


# DISPLAY TABLE

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    height=600
)


# DOWNLOAD CSV BUTTON

csv = df.to_csv(
    index=False
).encode("utf-8")

left_space, right_button = st.columns([6, 2])

with right_button:

    st.download_button(
        label="Download Assessment History CSV",
        data=csv,
        file_name="assessment_history.csv",
        mime="text/csv",
        use_container_width=True
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