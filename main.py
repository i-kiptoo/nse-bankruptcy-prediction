import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
from reduced_features import FinancialData
from transformers import DataFrameTransformer # Used in pipeline  # noqa: F401
import joblib
from PIL import Image

pipeline = joblib.load('lightgbm_pipeline.pkl')

def plotly_chart(fig, use_static=True):
    config = {
        'staticPlot': use_static,
        'displayModeBar': False,
        'scrollZoom': False,
        'doubleClick': False
    }
    st.plotly_chart(fig, config=config, use_container_width=True)

def upload_file():
    st.header("📂 Upload Financial Data")
    return st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            df.set_index(df.columns[0], inplace=True)
            st.success("✅ File Uploaded Successfully!")
            st.subheader("**Preview of Uploaded Data:**")
            st.dataframe(df)
            
            financial_data = FinancialData(
                revenue=df.loc["Revenue"],
                ebit=df.loc["Operating Profit/Loss"],
                net_income=df.loc["Profit/Loss of the Year"],
                current_receivables=df.loc["Trade And Other Receivables(Current Assets)"],
                cash_and_equivalents=df.loc["Cash And Bank Balances"],
                current_assets=df.loc['Total Current Assets'],
                total_assets=df.loc["Total Assets"],
                current_liabilities=df.loc['Total Current Liabilities'],
                total_liabilities=df.loc['Total Liabilities'],
                retained_earnings=df.loc["Retained Earnings"],
                net_cash_from_ops=df.loc["Net Cash From/Used in Operating Activities"]
                
            )
            return financial_data
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    return None

def manual_input():
    st.subheader("Enter Financial Data")

    col1, col2, col3 = st.columns(3, vertical_alignment="top", border=True, gap="small")
    with col1:
        "Consolidated Statement of Profit or Loss"
        revenue = st.number_input("Revenue/Sales", step=1000.0, value=None, placeholder="Required")
        ebit = st.number_input("Operating Profit/Loss", step=1000.0, value=None)
        with st.expander("Don't have Operating Profit/Loss? Let's Estimate it"):
            st.markdown("*Provide values to estimate Operating Profit:*")
            ebt = st.number_input("Profit/Loss Before Tax", step=1000.0, value=0.0)
            finance_costs = st.number_input("Finance Costs/Interest Expenses", step=1000.0, value=0.0)
            finance_income = st.number_input("Finance Income/Interest Income", step=1000.0, value=0.0)
            monetary_gains = st.number_input("Monetary Gains/Losses", step=1000.0, value=0.0)
            
            estimated_ebit = ebt - finance_costs - finance_income - monetary_gains
            st.success(f"Estimated Operating Profit: {estimated_ebit:.2f}")

            ebit = estimated_ebit
        net_income = st.number_input("Profit/Loss of the Year", step=1000.0, value=None, placeholder="Required")

    with col2:
        "Consolidated Statement of Financial Position"
        current_receivables = st.number_input("Trade And Other Receivables", step=1000.0, value=None, placeholder="Required")
        cash_and_equivalents = st.number_input("Cash And Bank Balances", step=1000.0, value=None, placeholder="Required")
        current_assets = st.number_input("Total Current Assets", step=1000.0, value=None, placeholder="Required")
        total_assets = st.number_input("Total Assets", step=1000.0, value=None, placeholder="Required")
        retained_earnings = st.number_input("Retained Earnings/Accumulated Losses", step=1000.0, value=None, placeholder="Required")
        current_liabilities = st.number_input("Total Current Liabilities", step=1000.0, value=None, placeholder="Required")
        total_liabilities = st.number_input("Total Liabilities", step=1000.0, value=None, placeholder="Required")
    with col3:
        "Consolidated Statement of Cash Flows"
        net_cash_from_ops = st.number_input("Net Cash From/Used in Operating Activities", step=1000.0, value=None, placeholder="Required")

    return FinancialData(
        revenue=pd.Series(revenue),
        ebit=pd.Series(ebit),
        net_income=pd.Series(net_income),
        current_receivables=pd.Series(current_receivables),
        cash_and_equivalents=pd.Series(cash_and_equivalents),
        current_assets=pd.Series(current_assets),
        total_assets=pd.Series(total_assets),
        current_liabilities=pd.Series(current_liabilities),
        total_liabilities=pd.Series(total_liabilities),
        retained_earnings=pd.Series(retained_earnings),
        net_cash_from_ops=pd.Series(net_cash_from_ops)
    )


def display_prediction_card(label: str, probability: float):
    """
    Display prediction result in a styled card with color-coded label and probability.

    Args:
        label (str): 'Bankrupt' or 'Non-Bankrupt'
        probability (float): Probability of bankruptcy (between 0 and 1)
    """
    if label == "Bankrupt":
        card_color = "#ffe6e6"  # light red
        text_color = "red"
        status = f"Bankruptcy is Likely within 3 years with a confidence of {probability * 100:.2f}%"

    elif label == "Non-Bankrupt":
        card_color = "#e6ffed"  # light green
        text_color = "green"
        status = f"Bankruptcy is Unlikely within 3 years with a confidence of {(1-(probability)) * 100:.2f}%"

    else:
        st.error(f"Unknown label '{label}' provided.")
        return

    st.markdown(
        f"""
        <div style="background-color: {card_color}; padding: 1rem; border-radius: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.1);">
            <h4 style="color: {text_color}; margin-bottom: 0.5rem;">{status}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )


def predict_bankruptcy_3yr(ratios_df, model):
    """
    Predict bankruptcy within 3 years using a trained model.

    Parameters:
    - ratios_df (pd.DataFrame): DataFrame containing financial ratios.
    - model: Trained classification model with predict and predict_proba.

    Outputs:
    - Streamlit UI elements showing predictions and probabilities.
    """
    # Predict labels and probabilities
    predictions = model.predict(ratios_df)
    probabilities = model.predict_proba(ratios_df)[:, 1]  # Probabilities for class 1 (bankrupt)

    if ratios_df.shape[0] == 1:
        # Handle single row prediction
        prediction = predictions[0]
        probability = probabilities[0]
        label = "Bankrupt" if prediction == 1 else "Non-Bankrupt"
        
        st.subheader("**Computed Financial Ratios:**")
        st.dataframe(ratios_df.T)
        
        display_prediction_card(probability=probability, label=label)
        plot_gauge(probability)
    
    else:
        # Handle multiple row prediction
        ratios_df = ratios_df.copy()  # avoid modifying original
        ratios_df['3-Year Prediction'] = np.where(
            predictions == 1,
            "bankutpycy is likely within 3 years",
            "bankruptcy is unlikely within 3 years"
        )
        ratios_df['Probability of Bankruptcy (%)'] = np.round(probabilities * 100, 2)

        def categorize_risk(prob):
            if prob >= 80:
                return "High Risk"
            elif prob >= 50:
                return "Moderate Risk"
            else:
                return "Low Risk"

        
        ratios_df['Risk Level'] = ratios_df['Probability of Bankruptcy (%)'].apply(categorize_risk)

        st.subheader("Batch Predictions (3-Year Window)")
        st.dataframe(ratios_df[['3-Year Prediction', 'Probability of Bankruptcy (%)', 'Risk Level']])


def plot_feature_importance(pipeline, feature_names):
    with st.expander("What does this chart mean?"):
        st.markdown("""
                    ### Feature Importance (Permutation Method)

                    This chart shows how much each feature contributes to the model’s performance.  
                    - The importance is measured by how much the model's accuracy drops when each feature is randomly shuffled.
                    - A **higher score** means the feature is **more important** to making accurate predictions.
                    - Features are ranked from most to least important.

                    Use this to understand which financial ratios the model considers most impactful in predicting bankruptcy.
        """)
    # Accessing the feature importances from the model in the pipeline
    feature_importances = pipeline.named_steps['clf'].feature_importances_
    
    sorted_idx = np.argsort(feature_importances)[::-1]  # Sorting in descending order
    sorted_importances = feature_importances[sorted_idx]
    sorted_features = np.array(feature_names)[sorted_idx]
    
    # st.subheader("Feature Contribution to Prediction")
    
    # Create the bar chart
    fig = px.bar(
        x=sorted_features, 
        y=sorted_importances, 
        
    )
    fig.update_layout(
        title="Feature Importance (Permutation Method)",
        xaxis_title="Feature", 
        yaxis_title="Importance"
    )
    
    plotly_chart(fig)
    
def plot_gauge(probability, threshold=0.5):
    """
    Creates a gauge chart for bankruptcy probability.
    
    Args:
        probability: Float value between 0 and 1
        threshold: Cutoff for bankruptcy classification
    """
    color = "red" if probability >= threshold else "green"
    # st.subheader("Bankruptcy Risk")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Bankruptcy Risk", 'font': {'size': 25}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgreen"},
                {'range': [threshold * 100, 100], 'color': "lightcoral"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    plotly_chart(fig)
 

def display_tooltips():
    st.markdown("### Financial Ratio Definitions")

    with st.expander("Liquidity Ratios"):
        st.markdown("""
        **Quick Assets to Total Assets**  
        *Formula:* Quick Assets / Total Assets  
        Shows the proportion of **liquid assets** (like cash and receivables) to the company’s total assets.

        **Operating Cash Flow Ratio**  
        *Formula:* Operating Cash Flow / Current Liabilities  
        Measures the ability to **cover short-term liabilities** with cash generated from operations.

        **Current Ratio**  
        *Formula:* Current Assets / Current Liabilities  
        Classic measure of **short-term financial health**. A ratio above 1 indicates the company can meet its short-term obligations.

        **Cash and Equivalents to Total Assets**  
        *Formula:* Cash & Equivalents / Total Assets  
        Reflects the portion of total assets held in **cash or highly liquid instruments**.
        """)

    with st.expander("Profitability Ratios"):
        st.markdown("""
        **Non-operating Income Margin**  
        *Formula:* Non-operating Income / Revenue  
        Measures income from **non-core operations**, like interest or investment income.

        **Operating Cash Flow to Revenue**  
        *Formula:* Operating Cash Flow / Revenue  
        Measures how effectively a company converts **sales into actual cash** from operations.

        **Receivables Turnover Ratio**  
        *Formula:* Revenue / Current Receivables  
        Indicates how often **accounts receivable** are collected. Higher = faster collection.

        **Retained Earnings to Total Assets**  
        *Formula:* Retained Earnings / Total Assets  
        Indicates how much of the company’s assets are financed by **retained profits**.
        """)

    with st.expander("Leverage & Solvency Ratios"):
        st.markdown("""
        **Non-Current Liabilities to Current Assets**  
        *Formula:* Non-Current Liabilities / Current Assets  
        Assesses how well current assets can **cover long-term obligations**.

        **Liabilities to Asset Ratio**  
        *Formula:* Total Liabilities / Total Assets  
        Indicates the degree to which a company is **funded by debt**.
        """)


def show_disclaimer():
    """
    Display a disclaimer regarding the use of the bankruptcy prediction tool inside an expander.
    """
    st.markdown("### Disclaimer")

    with st.expander("click to expand)"):
        st.markdown(
            """
            The **Bankruptcy Prediction Tool** is provided for informational and educational purposes only.

            ### !Important Notes!
            - The predictions generated by this tool are **not financial advice**.
            - While the model is based on historical financial data and uses machine learning techniques, it **does not guarantee accuracy** or future outcomes.
            - This tool should **not** be used as the sole basis for making business, investment, or credit decisions.
            - Users are encouraged to consult with a **qualified financial advisor or professional** before making any decisions based on the model’s output.

            ---
            By using this tool, you acknowledge that the creators are **not liable** for any decisions or actions taken based on its predictions.
            """
        )


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def plot_shap_contributions(pipeline, ratios_df):
    with st.expander("What does this chart mean?"):
        st.markdown("""
                    #### SHAP Feature Contributions
                    This chart displays the **financial ratios** that most influenced the model's prediction **for this specific input**.

                    - **Red bars** indicate a **positive contribution** toward predicting bankruptcy (i.e., pushing the prediction closer to 1).
                    - **Green bars** represent **negative contributions** (i.e., pushing the prediction toward 0).
                    - The longer the bar, the **stronger the impact** of that feature on the model’s output.

                    SHAP (SHapley Additive exPlanations) values are a way to explain individual predictions by showing how each feature shifts the prediction from the base value (average prediction) to the final predicted probability.
        """)
    # Extract raw model from pipeline
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        raw_model = pipeline.named_steps['clf']
    else:
        raise ValueError("Pipeline must contain a 'clf' step with LightGBM as the final estimator.")

    explainer = get_shap_explainer(raw_model)
    shap_values = explainer.shap_values(ratios_df)

    # Handle binary classification
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1][0]  # class 1
    elif isinstance(shap_values, np.ndarray):
        shap_vals = shap_values[0]  # SHAP now returns [n_features] instead of [2][n_instances][n_features]
    else:
        raise ValueError("Unexpected SHAP output format.")

    contributions = dict(zip(ratios_df.columns, shap_vals))
    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

    colors = ['red' if val > 0 else 'green' for val in sorted_contribs.values()]
    fig = go.Figure(go.Bar(
        x=list(sorted_contribs.values()),
        y=list(sorted_contribs.keys()),
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(title='SHAP Feature Contributions', xaxis_title='SHAP Value')
    plotly_chart(fig)

def main():
    st.set_page_config(
        page_title='Bankruptcy Prediction Dashboard',
        page_icon=':money:',
        layout='wide',
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
    <style>
    .custom-tab-container {
        width: 50%;
        margin: 0 auto; /* centers the container */
    }
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        min-width: 0;
        text-align: center;
    }
    </style>
    <div class="custom-tab-container">
""", unsafe_allow_html=True)
    tabs = st.tabs(["Data Input", "Prediction", "Info"])
            
    with tabs[0]:  # "Data Input" tab
        st.info('### BANKRUPTCY PREDICTION FOR FIRMS LISTED ON NAIROBI STOCK EXCHANGE MARKET(NSE)')
        
        st.warning('ALL SECTORS SUPPORTED EXCEPT(BANKING, ENERGY AND INSURANCE SECTOR)')
        
        st.header("Choose Input Method")
        input_method = st.radio("", ["Manual Input", "Upload File(CSV/XLXS)"], horizontal=True)

        if input_method == "Manual Input":
            financial_data = manual_input()
            if st.button("Submit", use_container_width=True):
                has_missing = financial_data.ratios().isna().all().all()
                if has_missing:
                    st.warning('Fill all the Required Fields')
                else:
                    st.session_state["submitted_data"] = financial_data
                    st.session_state["prediction_ready"] = True
                    st.success("Data submitted. Check the Prediction tab.")

        elif input_method == "Upload File(CSV/XLXS)":
            uploaded_file = upload_file()
            if uploaded_file is not None:
                financial_data = process_uploaded_file(uploaded_file)
                if st.button("Submit", use_container_width=True):
                    st.session_state["submitted_data"] = financial_data
                    st.session_state["prediction_ready"] = True
                    st.success("Data submitted. Check the Prediction tab.")
                    
    with tabs[1]:  # "Prediction" tab
        st.header("Prediction")

        if st.session_state.get("prediction_ready"):
            data = st.session_state["submitted_data"]
            ratios_df = data.ratios()
            
            predict_bankruptcy_3yr(ratios_df, pipeline)
            image = Image.open('images/lgbm_eval.png')
            with st.expander('What does this image mean?'):
                st.markdown("""
                            ### Model Evaluation Explained

                            This panel provides a comprehensive evaluation of the model’s performance on test data:

                            ---

                            #### 1️.ROC AUC Curve *(Top-Left)*  
                            - Plots **True Positive Rate (Recall)** vs. **False Positive Rate**.  
                            - The **AUC (Area Under Curve)** measures overall model ability to distinguish between bankrupt and non-bankrupt firms.  
                            - A perfect model has AUC = 1.0; random guessing = 0.5.

                            ---

                            #### 2️.Confusion Matrix *(Top-Right)*  
                            - Shows how many predictions the model got **correct and incorrect**.  
                            - Rows = actual values, Columns = predicted values.  
                            - You want high values on the **diagonal** (correct predictions).

                            ---

                            #### 3️.Precision-Recall Curve *(Bottom-Left)*  
                            - Focuses on performance in **imbalanced datasets**.  
                            - Higher area under the curve = better model for identifying bankrupt firms with fewer false alarms.  
                            - Useful when **False Negatives** (missing bankrupt firms) are very costly.

                            ---

                            #### 4️.Probability Distribution *(Bottom-Right)*  
                            - Shows predicted probability scores for both classes.  
                            - Helps visualize **model confidence**:  
                            - Blue = not bankrupt  
                            - Orange = bankrupt  
                            - Good separation = model assigns high probabilities correctly.
                """)
            st.image(image, use_container_width=True)
            plot_feature_importance(pipeline, ratios_df.columns)
            plot_shap_contributions(pipeline, ratios_df)
            
        else:
            st.info("Submit data from the 'Data Input' tab to see the prediction.")
        
    with tabs[2]:          
        show_disclaimer()
        display_tooltips()
            
if __name__ == '__main__':
    main()
