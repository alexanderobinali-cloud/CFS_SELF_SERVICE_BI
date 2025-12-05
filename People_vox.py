
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.cloud import bigquery
from google.oauth2 import service_account

table_descriptions = {
    'despatch_summary': 'Provides a summary of despatched items.',
    'despatched_items_by_package': 'Details items grouped by their despatch package.',
    'inventory_snapshot': 'Captures the state of inventory at a specific point in time.',
    'item_inventory_location_intraday': 'Tracks item inventory by location throughout the day.',
    'item_inventory_summary': 'Offers a summary of item inventory.',
    'item_inventory_summary_intraday': 'Provides an intraday summary of item inventory.',
    'item_movement_history': 'Records the history of item movements.',
    'mcr_pending_orders_archive_raw': 'Contains raw archived data for pending orders.',
    'mvw_quantity_by_sku': 'A materialized view showing quantities aggregated by SKU.',
    'sales_order_allocated': 'Lists sales orders that have had stock allocated to them.',
    'sales_order_summary': 'Summarizes sales order information.',
    'sales_order_update_history': 'Logs changes and updates to sales orders.',
    'summary_test_dt': 'A view used for testing summaries.',
    'tapfilliate_data': 'Contains data related to tapfilliate.',
    'temp_duplicates': 'A temporary table likely used for managing duplicates.',
    'tmp_despatch_summary': 'A temporary table for despatch summaries.',
    'tmp_item_movement_history': 'A temporary table for item movement history.',
    'vw_inventory_summary_item_code_character_loop': 'A view providing an inventory summary, potentially involving item code character logic.',
    'vw_item_movement_history_max_history_id': 'A view that identifies the maximum history ID in item movement history.'
}

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

def inject_floating_button_css(button_key):
    """
    Injects CSS to make an element with a specific key float 
    in the bottom-right corner, regardless of page scroll.
    """
    st.markdown(
        f"""
        <style>
        /* Target the button container based on its unique key */
        div[data-testid*="{button_key}"] {{
            position: fixed;
            bottom: 30px; /* Distance from the bottom edge */
            right: 30px; /* Distance from the right edge */
            z-index: 1000; /* Ensure it stays above all other content */
            
            /* Optional: Style adjustments for visibility */
            border: 2px solid #FF4B4B; 
            border-radius: 10px;
            padding: 5px 15px;
            background-color: #FFFFFF;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )
# 4b. Inject CSS for the floating button
FLOATING_BUTTON_KEY = "floating_logout_btn"
inject_floating_button_css(FLOATING_BUTTON_KEY)
# login screen
@st.dialog("🔐 App Login Required", width="small", dismissible=False)
def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.write("Current user object:", st.user)
    st.button("Log in with Google", on_click=st.login)
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

# --- Gemini Helper Functions ---

def generate_sql_query(prompt: str) -> str | None:
    """Generates a BigQuery SQL query from a natural language prompt."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    full_prompt = f"""
        Based on the following user request, generate a single, valid Google BigQuery SQL query string.
        - The query should be standard SQL compatible with Google BigQuery.
        - Do NOT include any explanation, comments, or markdown formatting like ```sql.
        - Only output the raw SQL query string.
        -The database is classic-football-shirts.peoplevox.+ table name from + {table_descriptions}
   

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


#logout
def local_logout():
    """Function to handle logout."""
    if st.user:
        st.logout()
    st.rerun()

# --- App State Initialization ---
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = {}
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'error' not in st.session_state:
    st.session_state.error = None



if not st.user.is_logged_in:
    login_screen()
elif '@classicfootballshirts.co.uk' not in st.user.email:
    st.error("Access denied. Please log in with your @classicfootballshirts.co.uk account.", icon="🚨")
    st.stop()



st.session_state.prompt = "I want all the orders dispatched in the last month"

# --- UI Layout ---

# Header
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5rem;">📦</span>
        <div>
            <h1 style="margin-bottom: 0;">People Vox</h1>
            <p style="margin-top: 0; color: #888;">Directly query people vox data and download to csv</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("### Available Tables and Descriptions")
df = pd.DataFrame(
    list(table_descriptions.items()),
    columns=['Table ID', 'Description']
)
st.dataframe(
    df,
    hide_index=True, # Hides the default Pandas index (0, 1, 2, ...)
    use_container_width=True # Makes the table fill the width of the app
)

st.write("### Query the peoplevox database using natural language")
# Step 1: Generate Query
with st.form(key="query_form"):
    prompt = st.text_area(
        "Describe the dataset you want to query:",
        value=st.session_state.prompt,
        height=100,
        key="prompt_input"
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
            st.session_state.sql_query['vox'] = generate_sql_query(prompt)

# Display Error if any
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# Step 2: Display SQL and Generate Data
if st.session_state.sql_query.get('vox'):
    st.markdown("---")
    st.subheader("Generated BigQuery SQL")
    st.code(st.session_state.sql_query['vox'], language="sql")
    
    if st.button("📈 Generate Data from Query", use_container_width=True, type="primary"):
        st.session_state.dataframe = None
        st.session_state.error = None
        with st.spinner("AI is simulating the query and generating data..."):
            st.session_state.dataframe = generate_data_from_query(st.session_state.sql_query['vox'])

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
        file_name=f"people-vox_data-{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
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
# 4e. The Floating Button (Place it anywhere, the CSS positions it)
st.button(
    "👋 Log Out", 
    on_click=local_logout, 
    key=FLOATING_BUTTON_KEY # Uses the key targeted by the CSS
)
