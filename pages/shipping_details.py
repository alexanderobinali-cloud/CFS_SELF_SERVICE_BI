
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.cloud import bigquery
from google.oauth2 import service_account

from People_vox import login_screen

month_end_query= """SELECT d.item_weight, d.item_code,d.salesorder_number,d.site_reference, d.carrier, d.service, d.destination_country, d.salesorder_number , s.order_date FROM `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.pvx_despatch_summary` as d  left join `classic-football-shirts.reporting.sales_orders` as s on d.salesorder_number = s.order_no
WHERE
    s.order_date >= '2026-01-01' 
    AND s.order_date < '2026-01-31''-- Replace with specific date or range
"""
@st.cache_resource
def get_bigquery_client():
    # 1. Load the dictionary from st.secrets
    # st.secrets['gcp_service_account'] will return the dictionary we defined in TOML
    try:
        service_account_info = st.secrets["gcp_service_account"]

        st.success("GCP Big Query connection successful...please wait for the query to finish. This may take a couple of minutes.")
    except KeyError:
        st.error("GCP Service Account secret not found. Check your .streamlit/secrets.toml file.")
        st.stop()
        
    # 2. Create credentials object from the info
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    
    # 3. Initialize the BigQuery client with credentials
    client = bigquery.Client(
        credentials=credentials,
        project=credentials.project_id # Ensure the project ID is used
    )
    return client

# --- Page Configuration ---
st.set_page_config(
    page_title="AI BigQuery Data Generator",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Gemini API Configuration ---
# NOTE: To run this app, create a .streamlit/secrets.toml file with your API key:
# API_KEY = "YOUR_GOOGLE_API_KEY"
try:
    API_KEY = st.secrets["API_KEY"]
    genai.configure(api_key=API_KEY)
except (KeyError, FileNotFoundError):
    st.error(f"API_KEY not found. Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()
##localised session state
if 'sql_query' not in st.session_state:
    # 2. If it does not exist, initialize it as an empty dictionary.
    st.session_state.sql_query = {}


# --- Gemini Helper Functions ---
@st.cache_resource
def generate_sql_query(prompt: str) -> str | None:
    """Generates a BigQuery SQL query from a natural language prompt."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    full_prompt = f"""
        Based on the following user request, generate a single, valid Google BigQuery SQL query string.
        - The query should be standard SQL compatible with Google BigQuery.
        - Do NOT include any explanation, comments, or markdown formatting like ```sql.
        - Only output the raw SQL query string.
        -The query is a modification of this standard report: {month_end_query}
        -**STRICTLY FORBIDDEN ACTIONS:**
You must not include, generate, or suggest any SQL command that modifies the database, its data, or its structure.
This includes, but is not limited to, the following keywords and commands:

* **Data Manipulation Language (DML) other than SELECT:**
    * `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `TRUNCATE`
* **Data Definition Language (DDL):**
    * `CREATE` (e.g., `CREATE TABLE`, `CREATE DATABASE`, `CREATE INDEX`)
    * `ALTER` (e.g., `ALTER TABLE`, `ALTER COLUMN`)
    * `DROP` (e.g., `DROP TABLE`, `DROP DATABASE`)
    * `RENAME`
* **Data Control Language (DCL):**
    * `GRANT`, `REVOKE`
* **Transaction/Session Management:**
    * `COMMIT`, `ROLLBACK`, `BEGIN TRANSACTION`
   

        User request: "{prompt}"
    """
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=GenerationConfig(temperature=0.1),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        sql_query = response.text.strip()
        if not sql_query.lower().startswith('select'):
            st.session_state.error = "Generated text does not appear to be a valid SQL query. Please try again."
            return None
        return sql_query
    except Exception as e:
        st.session_state.error = f"Failed to generate SQL query: {e}"
        return None

@st.cache_resource
def generate_data_from_query(sql_query: str) -> pd.DataFrame | None:
    """Simulates a BigQuery query execution to generate a dataset."""
    try:
        client = get_bigquery_client()
    # Execute the SQL script
        query_job = client.query(sql_query)  # API request
        response= query_job.result()
        
        output= response.to_dataframe()
        # data = response.candidates[0].content.parts[0].json
        # df = pd.DataFrame(data['rows'], columns=data['headers'])
        
        return output
    except Exception as e:
        st.session_state.error = f"Failed to generate data from query: {e}"
        return None

# --- App State Initialization ---
# if 'sql_query' not in st.session_state:
#     st.session_state.sql_query = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'error' not in st.session_state:
    st.session_state.error = None

st.session_state.prompt = "Run the depatch summary dataset query for last month"

# --- UI Layout ---


if not st.user.is_logged_in:
    login_screen()
elif '@classicfootballshirts.co.uk' not in st.user.email:
    st.error("Access denied. Please log in with your @classicfootballshirts.co.uk account.", icon="🚨")
    st.stop()


# Header
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5rem;">🛒</span>
        <div>
            <h1 style="margin-bottom: 0;">Stock Take Data</h1>
            <p style="margin-top: 0; color: #888;">Retrieve stock data for specific shops and dates </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# st.write("### Available Tables and Descriptions")
# df = pd.DataFrame(
#     list(table_descriptions.items()),
#     columns=['Table ID', 'Description']
# )
# st.dataframe(
#     df,
#     hide_index=True, # Hides the default Pandas index (0, 1, 2, ...)
#     use_container_width=True # Makes the table fill the width of the app
# )

st.write("### Modify the report using natural language")
# Step 1: Generate Query
with st.form(key="depatch_query_form"):
    prompt = st.text_area(
        "Describe the dataset you want to query:",
        value=st.session_state.prompt,
        height=100,
        key="prompt_input_stock"
    )
    submit_query = st.form_submit_button("✨ Generate Query", use_container_width=True)

if submit_query:
   
    st.session_state.dataframe = None
    st.session_state.error = None
    st.session_state.prompt = prompt # Save user's prompt
    
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("AI is generating your SQL query..."):
            st.session_state.sql_query['stock'] = generate_sql_query(prompt)

# Display Error if any
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# Step 2: Display SQL and Generate Data
if st.session_state.sql_query.get('stock'):
    st.markdown("---")
    st.subheader("Generated BigQuery SQL")
    st.code(st.session_state.sql_query['stock'] , language="sql")
    
    if st.button("📈 Generate Data from Query", use_container_width=True, type="primary"):
        st.session_state.dataframe = None
        st.session_state.error = None
        with st.spinner("AI is simulating the query and generating data..."):
            st.session_state.dataframe = generate_data_from_query(st.session_state.sql_query['stock'] )

# if st.session_state.get('error'):
#     st.error(st.session_state.error, icon="❌")
#     # Crucial: Stop execution or return None to prevent accessing 'df'

# Step 3: Display Data and Download
if st.session_state.dataframe is None:
    st.markdown("---")
    st.subheader("No data returned")
    
if st.session_state.dataframe is not None:
    st.markdown("---")
    st.subheader("Generated Dataset Preview")
    
    df = st.session_state.dataframe
    
    # Show preview message if data is large
    PREVIEW_ROW_LIMIT = 20
    if len(df) > PREVIEW_ROW_LIMIT:
        st.info(f"Showing a preview of the first {PREVIEW_ROW_LIMIT} of {len(df)} total rows. The full dataset will be downloaded.", icon="ℹ️")
        st.dataframe(df.head(PREVIEW_ROW_LIMIT))
    else:
        st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Data as CSV",
        data=csv,
        file_name=f"despatch-data-{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Initial placeholder
if not st.session_state.sql_query and not submit_query:
     st.info("Enter a description above and click 'Generate Query' to start.", icon="👋")

# Footer
st.markdown(
    """
    <hr style="margin-top: 40px;">
    <p style="text-align: center; color: #888;">Powered by Streamlit & Google Gemini</p>
    """,
    unsafe_allow_html=True
)
