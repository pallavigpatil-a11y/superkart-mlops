
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download

MODEL_REPO = "PallaviPatil0501/superkart-sales-model"

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.pkl"
    )

    return joblib.load(model_path)


st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="📈"
)

st.title("📈 SuperKart Sales Forecast")

st.write(
    "Predict product-store sales using the trained "
    "SuperKart machine learning model."
)

product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.0
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.05
)

product_type = st.selectbox(
    "Product Type",
    [
        "Frozen Foods",
        "Dairy",
        "Canned",
        "Baking Goods",
        "Health and Hygiene",
        "Snack Foods",
        "Meat",
        "Household",
        "Hard Drinks",
        "Fruits and Vegetables",
        "Breads",
        "Soft Drinks",
        "Breakfast",
        "Others",
        "Starchy Foods",
        "Seafood"
    ]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=150.0
)

store_id = st.text_input(
    "Store ID",
    value="OUT018"
)

store_establishment_year = st.number_input(
    "Store Establishment Year",
    value=1999,
    step=1
)

store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Departmental Store",
        "Food Mart"
    ]
)

if st.button("Predict Sales"):

    model = load_model()

    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type
    }])

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted Product Store Sales: ₹{prediction:,.2f}"
    )
