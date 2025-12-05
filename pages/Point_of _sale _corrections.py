
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.cloud import bigquery
from google.oauth2 import service_account

from People_vox import login_screen

month_end_query= """/*
Step 1: Identify all orders that were CREATED in the previous calendar month
and had their record UPDATED in the current calendar month. This is our initial
pool of candidates.
*/
WITH
  potentially_updated_orders AS (
    SELECT DISTINCT
      po.order_increment_id AS order_no
    FROM
      `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.ebizmarts_pos_orders` AS po
    WHERE
      -- Condition 1: The order was created in the previous calendar month.
      DATE_TRUNC(CAST(po.created_at AS DATE), MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
      -- Condition 2: The order was last updated in the current calendar month.
      AND DATE_TRUNC(CAST(po.updated_at AS DATE), MONTH) = DATE_TRUNC(CURRENT_DATE(), MONTH)
  ),
  /*
  Step 2: Get a distinct list of all order numbers that were ALREADY INCLUDED
  in the official month-end report for the previous month. This is our exclusion list.
  */
  already_reported_orders AS (
    SELECT DISTINCT
      order_no
    FROM
      `classic-football-shirts.reporting.vw_sales_orders_month_end`
    WHERE
      -- Filter the month-end report to the same period as the creation date above.
      order_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
      AND order_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
  )

/*
Step 3: Select the full, correctly formatted data from the main reporting table.
We will filter this data to only include orders that are in our "potential updates"
list but are NOT in our "already reported" list.
*/
SELECT
  -- The SELECT list is an exact copy to ensure a matching schema for appending.
  ROW_NUMBER() OVER (ORDER BY so.order_date DESC, so.order_no) AS row_num,
  so.order_no,
  so.order_line_no,
  so.order_date,
  CAST(so.order_time AS STRING) AS order_time,
  so.local_order_date,
  CAST(so.local_order_time AS STRING) AS local_order_time,
  so.despatched_date,
  CAST(so.despatched_time AS STRING) AS despatched_time,
  so.order_header_status,
  so.order_line_status,
  so.sales_channel,
  so.store_name,
  so.store_id,
  so.customer_id,
  so.customer_name,
  so.customer_email,
  so.customer_is_guest,
  so.billing_city,
  so.billing_country,
  so.shipping_street,
  so.shipping_city,
  so.shipping_region,
  so.shipping_postcode,
  so.shipping_country,
  so.shipping_description,
  so.shipping_courier,
  so.tendered_currency,
  so.promo_code,
  so.base_to_global_rate,
  so.base_to_order_rate,
  so.qty_ordered,
  so.qty_refunded,
  so.qty_canceled,
  so.qty_invoiced,
  so.transaction_id,
  so.stripe_charges_amount,
  so.stripe_charges_amount_captured,
  so.stripe_charges_amount_refunded,
  so.stripe_fees,
  so.stripe_exchange_rate,
  so.stripe_charge_id,
  so.stripe_invoice_id,
  so.stripe_currency,
  so.stripe_datasource,
  so.vat_rate,
  so.VAT_number,
  so.VAT_Status,
  so.payment_method,
  so.product_id,
  so.product_sku,
  so.parent_sku,
  so.product_description,
  so.product_cost,
  so.product_printing_cost,
  so.product_price,
  so.product_source,
  so.product_buying_status,
  so.product_department,
  so.product_season,
  so.product_condition,
  so.unit_cost_price_gbp,
  so.unit_cost_price_local,
  so.total_cost_price_refunded_gbp,
  so.total_cost_price_refunded_local,
  so.total_cost_price_gbp,
  so.total_cost_price_local,
  so.unit_sale_price_gbp,
  so.unit_sale_price_local,
  so.total_sale_price_gbp,
  so.total_sale_price_local,
  so.discount_gbp,
  so.discount_local,
  so.giftcard_gbp,
  so.giftcard_local,
  so.product_refund_gbp,
  so.product_refund_local,
  so.discount_refund_gbp,
  so.discount_refund_local,
  so.unit_revenue_inc_vat_gbp,
  so.unit_revenue_inc_vat_local,
  so.total_revenue_inc_vat_gbp,
  so.total_revenue_inc_vat_local,
  so.unit_product_revenue_inc_vat_gbp,
  so.unit_product_revenue_inc_vat_local,
  so.total_product_revenue_inc_vat_gbp,
  so.total_product_revenue_inc_vat_local,
  so.unit_vat_gbp,
  so.unit_vat_local,
  so.total_vat_gbp,
  so.total_vat_local,
  so.unit_product_vat_gbp,
  so.unit_product_vat_local,
  so.total_product_vat_gbp,
  so.total_product_vat_local,
  so.unit_revenue_ex_vat_gbp,
  so.unit_revenue_ex_vat_local,
  so.total_revenue_ex_vat_gbp,
  so.total_revenue_ex_vat_local,
  so.unit_product_revenue_ex_vat_gbp,
  so.unit_product_revenue_ex_vat_local,
  so.total_product_revenue_ex_vat_gbp,
  so.total_product_revenue_ex_vat_local,
  so.total_margin_gbp,
  so.total_margin_local,
  so.product_margin_gbp,
  so.product_margin_local,
  so.shipping_charge_inc_vat_gbp,
  so.shipping_charge_inc_vat_local,
  so.shipping_charge_vat_gbp,
  so.shipping_charge_vat_local,
  so.shipping_charge_ex_vat_gbp,
  so.shipping_charge_ex_vat_local,
  so.shipping_refunded_gbp,
  so.shipping_refunded_local,
  so.free_shipping,
  so.duties_base_fee_gbp,
  so.duties_base_fee_local,
  so.duties_base_fee_tax_gbp,
  so.duties_base_fee_tax_local,
  so.duties_total_gbp,
  so.duties_total_local,
  so.total_refunded_gbp,
  so.total_refunded_local,
  so.is_giftcard,
  so.base_adjustment,
  so.pos_multiple.currency_code AS pos_currency_code,
  so.pos_multiple.pos_izettle,
  so.pos_multiple.pos_cash,
  so.pos_multiple.pos_payment_express,
  so.b2b_pc_code,
  so.b2c_pc_code
FROM
  `classic-football-shirts.reporting.sales_orders` AS so
  -- We join to our list of candidate orders. This acts as an initial filter.
  INNER JOIN potentially_updated_orders puo
    ON so.order_no = puo.order_no
  -- Now we LEFT JOIN to our exclusion list (orders already reported).
  LEFT JOIN already_reported_orders aro
    ON so.order_no = aro.order_no
-- The crucial final filter: we only keep rows where the order was NOT found
-- in the `already_reported_orders` list (i.e., the join returned NULL).
WHERE
  aro.order_no IS NULL;

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

st.session_state.prompt = "Run the point of sale corrections report"

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
        <span style="font-size: 2.5rem;">🧾</span>
        <div>
            <h1 style="margin-bottom: 0;">Point of sale report</h1>
            <p style="margin-top: 0; color: #888;">This identifies orders created in previous month but processed/updated after month-end </p>
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
with st.form(key="pos_query_form"):
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
            st.session_state.sql_query['point_of_sale'] = generate_sql_query(prompt)

# Display Error if any
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# Step 2: Display SQL and Generate Data
if st.session_state.sql_query.get('point_of_sale'):
    st.markdown("---")
    st.subheader("Generated BigQuery SQL")
    st.code(st.session_state.sql_query['point_of_sale'] , language="sql")
    
    if st.button("📈 Generate Data from Query", use_container_width=True, type="primary"):
        st.session_state.dataframe = None
        st.session_state.error = None
        with st.spinner("AI is simulating the query and generating data..."):
            st.session_state.dataframe = generate_data_from_query(st.session_state.sql_query['point_of_sale'] )

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
        file_name=f"point-of-sale-data-{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
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
