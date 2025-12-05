
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.cloud import bigquery
from google.oauth2 import service_account

from People_vox import login_screen

month_end_query= """WITH LatestProductInfo AS (
  SELECT
    product_sku,
    sizes
  FROM
    `classic-football-shirts.reference.product`
  WHERE
    marked_deleted IS NOT TRUE -- Handles both FALSE and NULL
  QUALIFY ROW_NUMBER() OVER(PARTITION BY product_sku ORDER BY product_id DESC) = 1 -- IMPORTANT: Change 'updated_at' if your recency column is named differently
),

CreditMemoItems AS (
  SELECT
    cm.entity_id AS creditmemo_id,
    so.increment_id AS salesorder_increment_id,
    cm.increment_id AS creditmemo_increment_id,
    so.created_at AS so_created_at,
    cm.created_at AS cm_created_at,
    cm.base_to_global_rate,
    cm.order_currency_code,
    (cm.base_shipping_amount * cm.base_to_global_rate) AS base_shipping_total,
    (cm.base_shipping_tax_amount * cm.base_to_global_rate) AS base_shipping_tax_amount_total,
    (cm.base_shipping_incl_tax * cm.base_to_global_rate) AS base_shipping_incl_tax_total,
    (cm.base_adjustment * cm.base_to_global_rate) AS base_adjustment_total,
    (cm.base_adjustment_positive * cm.base_to_global_rate) AS base_adjustment_positive_total,
    (cm.base_adjustment_negative * cm.base_to_global_rate) AS base_adjustment_negative_total,
    so.country,
    cmi.sku,
    cmi.name,
    cmi.qty,
    cmi.base_price * cm.base_to_global_rate AS price,
    cmi.base_row_total_incl_tax * cm.base_to_global_rate AS subtotal,
    cmi.tax_amount,
    cmi.base_tax_amount * cm.base_to_global_rate AS tax_amount_calc,
    COALESCE(cmi.base_discount_amount * cm.base_to_global_rate, 0) AS discount,
    (cmi.base_row_total_incl_tax * cm.base_to_global_rate) - COALESCE(cmi.base_discount_amount * cm.base_to_global_rate, 0) AS row_total,
    cpe.type_id AS prod_type,
    cpe.sku AS cpe_sku,
    COUNT(*) OVER (PARTITION BY cmi.sku, cm.entity_id) AS sku_count,
    CASE
      WHEN cmi.sku IS NULL AND cmi.name IS NULL AND cm.base_shipping_incl_tax > 0
      THEN TRUE
      ELSE FALSE
    END AS is_shipping_only
  FROM
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.sales_creditmemo` AS cm
  LEFT JOIN
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.sales_creditmemo_item` AS cmi
    ON cm.entity_id = cmi.parent_id
    and cmi.__hevo__marked_deleted = false
  JOIN
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.sales_order` AS so
    ON cm.order_id = so.entity_id
    and so.__hevo__marked_deleted = false
  LEFT JOIN
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_entity` AS cpe
    ON cmi.product_id = cpe.entity_id
    and cpe.__hevo__marked_deleted = false
  WHERE
    cm.created_at >= '2025-10-01'
    AND cm.created_at < '2025-11-01'
    and cm.__hevo__marked_deleted = false

)

SELECT
  cmi.creditmemo_id,
  cmi.so_created_at,
  cmi.cm_created_at,
  cmi.salesorder_increment_id,
  cmi.creditmemo_increment_id,
  cmi.base_to_global_rate,
  cmi.order_currency_code,
  cmi.base_adjustment_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_adjustment_per_line,
  cmi.base_adjustment_positive_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_adjustment_positive_per_line,
  cmi.base_adjustment_negative_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_adjustment_negative_per_line,
  cmi.base_shipping_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_shipping_per_line,
  cmi.base_shipping_tax_amount_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_shipping_tax_amount_per_line,
  cmi.base_shipping_incl_tax_total / COUNT(cmi.creditmemo_id) OVER (PARTITION BY cmi.creditmemo_id) AS base_shipping_incl_tax_per_line,
  cmi.sku,
  cmi.name,
  lpi.sizes AS product_size, -- <-- SIZE IS NOW JOINED FROM THE NEW CTE
  cmi.qty,
  cmi.price,
  cmi.subtotal,
  cmi.tax_amount,
  cmi.discount,
  cmi.row_total,
  cmi.prod_type,
  cmi.cpe_sku,
  cmi.country,
  -- Select columns from the reporting sales_orders table as needed
  so_n.*
FROM
  CreditMemoItems AS cmi
LEFT JOIN
  `classic-football-shirts.reporting.sales_orders` AS so_n
  ON cmi.salesorder_increment_id = so_n.order_no
     AND cmi.sku = so_n.product_sku
LEFT JOIN
  LatestProductInfo AS lpi ON cmi.sku = lpi.product_sku -- <-- JOINING ON SKU
WHERE
  cmi.is_shipping_only
  OR (cmi.price > 0 AND (cmi.prod_type = 'configurable' OR (cmi.prod_type = 'simple' AND cmi.sku_count = 1)))
ORDER BY
  cmi.cm_created_at;
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
        **STRICTLY FORBIDDEN ACTIONS:**
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
        # if not sql_query.lower().startswith('select'):
        #     st.session_state.error = "Generated text does not appear to be a valid SQL query. Please try again."
        #     return None
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
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = {}
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'error' not in st.session_state:
    st.session_state.error = None



st.session_state.prompt = "Run the credit memo report"

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
        <span style="font-size: 2.5rem;">💳</span>
        <div>
            <h1 style="margin-bottom: 0;">Credit Memo report</h1>
            <p style="margin-top: 0; color: #888;">Download the Credit memo report</p>
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
with st.form(key="credit_query_form"):
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
            st.session_state.sql_query['credit'] = generate_sql_query(prompt)

# Display Error if any
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# Step 2: Display SQL and Generate Data
if st.session_state.sql_query.get('credit'):
    st.markdown("---")
    st.subheader("Generated BigQuery SQL")
    st.code(st.session_state.sql_query['credit'] , language="sql")
    
    if st.button("📈 Generate Data from Query", use_container_width=True, type="primary"):
        st.session_state.dataframe = None
        st.session_state.error = None
        with st.spinner("AI is simulating the query and generating data..."):
            st.session_state.dataframe = generate_data_from_query(st.session_state.sql_query['credit'] )

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
        file_name=f"credit-memo-report-data-{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Initial placeholder
if not st.session_state.sql_query.get('credit') and not submit_query:
     st.info("Enter a description above and click 'Generate Query' to start.", icon="👋")

# Footer
st.markdown(
    """
    <hr style="margin-top: 40px;">
    <p style="text-align: center; color: #888;">Powered by Streamlit & Google Gemini</p>
    """,
    unsafe_allow_html=True
)
