
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.cloud import bigquery
from google.oauth2 import service_account

from People_vox import login_screen

month_end_query= """-- Step 1-3: Calculate the grandparent SKU based on rules
WITH product_base_info AS (
  -- Step 1: Get basic product info, including product_id, brand, parent SKU, and department
  SELECT
    cpe.entity_id AS product_id, -- Alias entity_id as product_id
    cpe.sku AS product_sku,
    -- Normalize brand name to handle potential case differences ('adidas' vs 'Adidas')
    LOWER(eaov_brand.value) AS brand,
    parent_sku_subquery.parent_sku,
    departments.department -- Include department
  FROM
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_entity` AS cpe
  LEFT JOIN -- Join to get brand (attribute_id=763 is assumed for brand based on your original query)
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_entity_int` AS cpei_brand
    ON cpe.entity_id = cpei_brand.entity_id
    AND cpei_brand.attribute_id = 763 -- Make sure 763 is the correct attribute_id for 'brand'
    AND cpei_brand.store_id = 0
    AND cpei_brand.__hevo__marked_deleted = False
  LEFT JOIN
    `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.eav_attribute_option_value` AS eaov_brand
    ON cpei_brand.value = eaov_brand.option_id
    AND eaov_brand.store_id = 0
    AND eaov_brand.__hevo__marked_deleted = False
  LEFT JOIN -- Join to get parent SKU
    (
      SELECT
        cpr.child_id,
        ARRAY_AGG(cpe_parent.sku ORDER BY cpe_parent.entity_id DESC LIMIT 1)[OFFSET(0)] AS parent_sku
      FROM
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_relation` AS cpr
      JOIN
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_entity` AS cpe_parent
        ON cpr.parent_id = cpe_parent.entity_id AND cpe_parent.__hevo__marked_deleted = False
      WHERE
        cpr.__hevo__marked_deleted = False
      GROUP BY
        cpr.child_id
    ) AS parent_sku_subquery
    ON cpe.entity_id = parent_sku_subquery.child_id
  LEFT JOIN -- Join to get department
    (
      SELECT
        cpei_department.entity_id,
        STRING_AGG(DISTINCT eaov_department.value, ', ') AS department
      FROM
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_entity_int` AS cpei_department
      LEFT JOIN
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.eav_attribute_option_value` AS eaov_department
        ON cpei_department.value = eaov_department.option_id
        AND eaov_department.store_id = 0
        AND eaov_department.__hevo__marked_deleted = False
      WHERE
        cpei_department.attribute_id = 1702 -- Make sure 1702 is correct attribute_id for department
        AND cpei_department.store_id = 0
        AND cpei_department.__hevo__marked_deleted = False
      GROUP BY
        cpei_department.entity_id
    ) AS departments
    ON cpe.entity_id = departments.entity_id
  WHERE
    cpe.__hevo__marked_deleted = False
    AND cpe.type_id = 'simple'
    -- Use LOWER on brand value for consistent matching
    --AND LOWER(eaov_brand.value) IN ('adidas', 'puma')
    AND departments.department IN ('Clearance', 'Current Season')
),

removal_patterns AS (
  -- Step 2: Define patterns needed for cleaning logic
  SELECT
    r'-(?i)(LDN|MCR|NYC|NYC2|BER|POP|MMI|LAX|LAX2|BRM|SNC|MMI2|KOR|MKT|FBA)$' AS location_pattern_end,
    r'(-\\d{1,2}(?:[\\.-]?)[A-Z]+)' AS print_pattern_middle,
    r'-\\d+CM$' AS cm_size_pattern_end,
    r'-(?i)(cup|prem|euro|league|final|dom|copa|pol)$' AS league_cup_pattern_end,
    ( -- Dynamic size pattern
      SELECT
        CONCAT('-(?:', STRING_AGG(DISTINCT REGEXP_REPLACE(optval.value, r'[.\\\\*+?|()\\[\\]{}-]', r'\\\\\\0'), '|'), ')$')
      FROM
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.eav_attribute` AS eav
      JOIN
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.eav_attribute_option` AS opt ON eav.attribute_id = opt.attribute_id
      JOIN
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.eav_attribute_option_value` AS optval ON opt.option_id = optval.option_id AND optval.__hevo__marked_deleted = false
      JOIN
        `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.catalog_product_super_attribute` AS super ON super.attribute_id = eav.attribute_id AND eav.__hevo__marked_deleted = false
      WHERE
        eav.attribute_code = 'size' AND eav.__hevo__marked_deleted = false
    ) AS size_pattern_end
),

calculated_skus AS (
  -- Step 3: Apply cleaning logic and pass through essential IDs
  SELECT
    pbi.product_id, -- << Pass through product_id
    pbi.product_sku,
    pbi.parent_sku, -- << Pass through parent_sku
    pbi.brand,      -- << Keep brand/dept for potential debugging/ordering
    pbi.department, -- << Keep brand/dept for potential debugging/ordering
    -- COALESCE(NULLIF(TRIM(pbi.parent_sku), ''), pbi.product_sku) AS starting_sku, -- For debug if needed

    CASE
      -- Strategy 1: If Puma and starts with #NNNNNN-NN or NNNNNN-NN, extract NNNNNN-NN
      WHEN pbi.brand = 'puma' AND REGEXP_CONTAINS(COALESCE(NULLIF(TRIM(pbi.parent_sku), ''), pbi.product_sku), r'^#?\\d{6}-\\d{2}')
      THEN REGEXP_EXTRACT(COALESCE(NULLIF(TRIM(pbi.parent_sku), ''), pbi.product_sku), r'^#?(\\d{6}-\\d{2})') -- Extract only digits/hyphen part

      -- Strategy 2: Fallback sequential cleaning (Adidas/Other Puma)
      ELSE REGEXP_REPLACE( -- 7. Remove leading hash (applied to all results)
             REGEXP_REPLACE( -- 6. Cleanup trailing - or .
               REGEXP_REPLACE( -- 5. Remove league/cup suffix (from the end)
                 REGEXP_REPLACE( -- 4. Remove dynamic size pattern (from the end)
                   REGEXP_REPLACE( -- 3b. Remove explicit CM size (from the end)
                     REGEXP_REPLACE( -- 2. Remove print pattern (middle/anywhere)
                       REGEXP_REPLACE( -- 1. Remove location pattern (from the end)
                         COALESCE(NULLIF(TRIM(pbi.parent_sku), ''), pbi.product_sku),
                         rp.location_pattern_end, ''
                       ),
                       rp.print_pattern_middle, ''
                     ),
                     rp.cm_size_pattern_end, '' -- Apply CM size removal
                   ),
                   COALESCE(rp.size_pattern_end, '$^'), '' -- Apply dynamic size removal AFTER CM size
                 ),
                 rp.league_cup_pattern_end, ''
               ),
               r'[-.]+$', ''
             ),
             r'^#', '' -- Remove leading # if present
           )
    END AS calculated_grandparent_sku

  FROM
    product_base_info AS pbi
  CROSS JOIN
    removal_patterns AS rp
)

-- Step 4: Combine calculated results with manual overrides and select desired columns
SELECT
  -- Product ID: Primarily from calculated data. Will be NULL if the SKU only exists in overrides and wasn't selected in product_base_info.
  cs.product_id,

  -- Product SKU: Use from calculated data if exists, otherwise from override table
  COALESCE(cs.product_sku, ov.sku) AS product_sku,

  -- Parent SKU: Primarily from calculated data. Will be NULL if the SKU only exists in overrides and wasn't selected in product_base_info OR if the product has no parent.
  cs.parent_sku,

  -- Grandparent SKU: Prioritize manual override, then the calculated value
  COALESCE(ov.grandparent_sku, cs.calculated_grandparent_sku) AS grandparent_sku

  -- You can optionally include brand/department here if needed for review/debugging
  -- cs.brand,
  -- cs.department

FROM
  calculated_skus AS cs
FULL OUTER JOIN -- Use FULL OUTER JOIN to include SKUs from both sources
  `classic-football-shirts.hevo_dataset_classic_football_shirts_fNpk.grandparent_sku_manual_overrides` AS ov -- <<< REPLACE WITH YOUR ACTUAL TABLE NAME
  ON cs.product_sku = ov.sku -- Assumes SKUs match exactly between calculated and override

-- Optional: Add ORDER BY for review
ORDER BY
  -- Order primarily by the definitive product SKU
  COALESCE(cs.product_sku, ov.sku)
  -- You could add secondary sorting if needed, e.g., by product_id
  -- COALESCE(cs.product_id, -1) -- Put override-only SKUs (with potentially NULL ID) first or last

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
    st.session_state.sql_query = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'error' not in st.session_state:
    st.session_state.error = None

st.session_state.prompt = "Run the product data query"

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
        <span style="font-size: 2.5rem;">👕</span>
        <div>
            <h1 style="margin-bottom: 0;">Product Data Queries</h1>
            <p style="margin-top: 0; color: #888;"> Calculate grandparent SKU (grouping level) using Regex rules and manual overrides</p>
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
with st.form(key="product_query_form"):
    prompt = st.text_area(
        "Describe the dataset you want to query:",
        value=st.session_state.prompt,
        height=100,
        key="prompt_input_product"
    )
    submit_query = st.form_submit_button("✨ Generate Query", use_container_width=True)

if submit_query:
    st.session_state.sql_query['product'] = None
    st.session_state.dataframe = None
    st.session_state.error = None
    st.session_state.prompt = prompt # Save user's prompt
    
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("AI is generating your SQL query..."):
            st.session_state.sql_query['product'] = generate_sql_query(prompt)

# Display Error if any
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# Step 2: Display SQL and Generate Data
if st.session_state.sql_query.get('product'):
    st.markdown("---")
    st.subheader("Generated BigQuery SQL")
    st.code(st.session_state.sql_query['product'] , language="sql")
    
    if st.button("📈 Generate Data from Query", use_container_width=True, type="primary"):
        st.session_state.dataframe = None
        st.session_state.error = None
        with st.spinner("AI is simulating the query and generating data..."):
            st.session_state.dataframe = generate_data_from_query(st.session_state.sql_query['product'] )

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
        file_name=f"product-queries-data-{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
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
