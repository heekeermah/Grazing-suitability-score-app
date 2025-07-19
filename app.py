import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import os
import traceback

# --- GPT Suggestion Function ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

def generate_ai_suggestion(diagnosis, biomass, shrub, grazing, woody):
    prompt = f"""
    A grazing land plot has the following issues:
    - Diagnosis: {diagnosis}
    - Available biomass: {biomass}
    - Shrub cover percentage: {shrub}%
    - Grazing pressure (animals/ha): {grazing}
    - Woody plant count: {woody}

    Suggest a specific, practical solution that a rangeland manager or farmer can apply to improve this plot.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are an expert in rangeland management."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error getting suggestion: {e}"
# --- Helper Functions ---
def calculate_gss(df, weights=None):
    try:
        if weights is None:
            weights = {
                'biomass': 0.4,
                'shrub': 0.2,
                'grazing': 0.2,
                'woody': 0.2
            }

        data = df[['available_biomass', 'Shrub %', 'grazing_pressure', 'total woody count']].copy()

        # Debug view of input data
        st.write("🔍 Input data for scaling:")
        st.dataframe(data.head())

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled, columns=['biomass_score', 'shrub_raw', 'grazing_raw', 'woody_raw'])

        # Debug view of scaled data
        st.write("🔍 Scaled values:")
        st.dataframe(scaled_df.head())

        scaled_df['shrub_score'] = 1 - scaled_df['shrub_raw']
        scaled_df['grazing_score'] = 1 - scaled_df['grazing_raw']
        scaled_df['woody_score'] = 1 - scaled_df['woody_raw']

        scaled_df['GSS'] = (
            weights['biomass'] * scaled_df['biomass_score'] +
            weights['shrub'] * scaled_df['shrub_score'] +
            weights['grazing'] * scaled_df['grazing_score'] +
            weights['woody'] * scaled_df['woody_score']
        )

        def diagnose(row):
            if row.GSS < 0.3:
                if row.shrub_raw > 0.7:
                    return "Too much shrub cover"
                elif row.grazing_raw > 0.7:
                    return "Excessive grazing pressure"
                elif row.woody_raw > 0.7:
                    return "High woody plant density"
                else:
                    return "Very low biomass"
            elif row.GSS < 0.5:
                return "Poor condition, needs intervention"
            elif row.GSS < 0.75:
                return "Moderately suitable"
            else:
                return "Highly suitable"

        scaled_df['Diagnosis'] = scaled_df.apply(diagnose, axis=1)
        result = pd.concat([df.reset_index(drop=True), scaled_df[['GSS', 'Diagnosis']]], axis=1)
        result['AI_Suggestion'] = result.apply(
            lambda row: generate_ai_suggestion(
                row['Diagnosis'],
                row['available_biomass'],
                row['Shrub %'],
                row['grazing_pressure'],
                row['total woody count']
            ),
            axis=1
        )
        return result

    except Exception as e:
        st.error("❌ Error during GSS calculation:")
        st.error(traceback.format_exc())
        st.error("⚠️ Data passed to function:")
        st.dataframe(df[['available_biomass', 'Shrub %', 'grazing_pressure', 'total woody count']].head())
        return pd.DataFrame()

# --- Streamlit App
def main():
    st.set_page_config(page_title="Grazing Suitability Checker", layout="wide")
    st.title("🌾 Grazing Suitability Score (GSS) Calculator with AI Suggestions")
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully.")
            st.write("📋 Available columns:", df.columns.tolist())
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
        else:
            required_cols = ['Plot Name', 'available_biomass', 'Shrub %', 'grazing_pressure', 'total woody count']
            if not set(required_cols).issubset(df.columns):
                missing = set(required_cols) - set(df.columns)
                st.error(f"❌ Missing columns: {missing}")
            elif df[required_cols].isnull().any().any():
                st.error("❌ Your file contains missing values in required columns. Please clean the data and try again.")
            else:
                st.write("📋 First 5 rows of your data:")
                st.dataframe(df.head())

                with st.spinner("🧠 Calculating GSS and generating AI suggestions..."):
                    result = calculate_gss(df)

                if not result.empty:
                    st.subheader("📊 Results")
                    st.dataframe(result)

                    csv = result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download results as CSV",
                        data=csv,
                        file_name='gss_results_with_ai.csv',
                        mime='text/csv'
                    )

                    st.subheader("📈 Grazing Suitability Score Distribution")
                    st.bar_chart(result['GSS'])
                    st.subheader("Top and Bottom Plots by GSS")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Top 5 Plots")
                        st.dataframe(result.sort_values('GSS', ascending=False).head(5))
                    with col2:
                        st.markdown("#### Bottom 5 Plots")
                        st.dataframe(result.sort_values('GSS', ascending=True).head(5))
                    
    else:
        st.info("📂 Upload a CSV or Excel file to begin.")
        st.markdown("Required columns: `Plot Name`, `grazing_pressure`, `Shrub %`, `total woody count`, `available_biomass`.")

if __name__ == "__main__":
    main()
