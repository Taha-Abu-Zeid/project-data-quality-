import io
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO
import re
from methods import *
# vsfbfs
# from Data_Quality_last_lab.methods import column_names_analysis, correlation_matrix, data_types_analysis, describe_data, download_dataset, handle_duplicates, handle_missing_values, handle_outliers, missing_value_analysis, outlier_analysis, reset_all_flags, visualize_data

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Data Quality Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        if 'data' not in st.session_state:
            try:
                if uploaded_file.name.endswith(".csv"):
                   csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                   df = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                   df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.sidebar.success("Dataset uploaded successfully!")
            except Exception as e:
               st.sidebar.error(f"Error: {e}")
        else:
            df = st.session_state['data'].copy()

        st.write("### Original DataFrame")
        st.write(df)
        if st.sidebar.button("Dataset info", key='show_data_btn'):
            reset_all_flags()
            st.session_state['show_data'] = True
            if df is not None:
                buffer = io.StringIO()
                df.info(buf=buffer)
                dataset_info = buffer.getvalue()

        if 'show_data' in st.session_state and st.session_state['show_data']:
            reset_all_flags()
            st.write("### Dataset Information:")
            st.text(dataset_info)

        if st.sidebar.button("Describe Data", key='describe_data_btn'):
             reset_all_flags()
             st.session_state['describe_data'] = True

        if 'describe_data' in st.session_state and st.session_state['describe_data']:
            reset_all_flags()
            st.header("Data Description")
            st.table(describe_data(df))

        if 'data' in st.session_state:
            df = st.session_state['data']
        else:
            st.warning("No data loaded. Please upload a dataset.")
            df = None

        if st.sidebar.button("Data Type Analysis", key='data_type_btn'):
            reset_all_flags()
            st.session_state['data_type_analysis_clicked'] = True

        if 'data_type_analysis_clicked' in st.session_state and st.session_state['data_type_analysis_clicked']:
            if df is not None:
                df = data_types_analysis(df)

        if 'type_converted' in st.session_state and st.session_state['type_converted']:
            st.write("Updated DataFrame:")
            st.write(df)
            st.session_state['type_converted'] = False



        if st.sidebar.button("Column Name Analysis", key='col_name_btn'):
             reset_all_flags()
             st.session_state['column_name_analysis_clicked']= True

        if 'column_name_analysis_clicked' in st.session_state and st.session_state['column_name_analysis_clicked']:
            df = column_names_analysis(df)           
            st.session_state['column_name_analysis_clicked']= True 


        if 'columns_renamed' in st.session_state and st.session_state['columns_renamed']:
            reset_all_flags()
            # st.session_state['column_name_analysis_clicked']= True
            st.write(df)
            # st.session_state['columns_renamed'] = True
            # st.session_state['column_name_analysis_clicked']= True  
            


        if st.sidebar.button("Missing Value Analysis", key='missing_val_btn'):
            reset_all_flags()
            st.session_state['missing_analysis_run'] = True
            
        if 'missing_analysis_run' in st.session_state and st.session_state['missing_analysis_run']:
           st.header("Missing Value Analysis")
           missing_value_analysis(df)
           st.session_state['missing_analysis_run']= False

        method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
        column = st.sidebar.selectbox("Select Column (optional)", df.columns, key="missing_col")
        if st.sidebar.button("Handle Missing Values", key='handle_missing_btn'):
            reset_all_flags()
            df = handle_missing_values(df, method, column)
            st.session_state['data'] = df
            st.session_state['missing_values_handled'] = True

        if 'missing_values_handled' in st.session_state and st.session_state['missing_values_handled']:
            st.header("Data after Handling Missing Values")
            st.write(df)
            missing_value_analysis(df)
            st.session_state['missing_values_handled']= False

        if st.sidebar.button("Handle Duplicates", key='handle_duplicates_btn'):
            reset_all_flags()
            st.session_state['handle_duplicates_clicked']=True
        if 'handle_duplicates_clicked' in st.session_state and st.session_state['handle_duplicates_clicked']:
            if df is not None:
                df = handle_duplicates(df)


        if 'duplicates_handled' in st.session_state and st.session_state['duplicates_handled']:
            st.header("Data after Handling Duplicates")
            st.write(df)
            st.session_state['duplicates_handled'] = False
            

        column_for_outlier = st.sidebar.selectbox("Select Column for Outlier Analysis", df.select_dtypes(include=['float64', 'int64']).columns, key="outlier_col")
        if st.sidebar.button("Outlier Analysis", key='outlier_analysis_btn'):
             reset_all_flags()
             st.session_state['outlier_analysis_run'] = True
        if 'outlier_analysis_run' in st.session_state and st.session_state['outlier_analysis_run']:
            st.header("Outlier Analysis")
            lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
            if lower_bound is not None and upper_bound is not None:
                outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method")
                if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
                    df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                    st.session_state['data'] = df
                    st.session_state['outliers_handled'] = True
                    reset_all_flags()

        if 'outliers_handled' in st.session_state and st.session_state['outliers_handled']:
             st.header("Data after Handling Outliers")
             st.write(df)
             st.session_state['outliers_handled']=False

        column_to_visualize = st.sidebar.selectbox("Select Column for Visualization", df.columns, key="visualize_col")
        if st.sidebar.button("Visualize Data", key='visualize_data_btn'):
            reset_all_flags()
            st.session_state['visualize_data_run'] = True

        if 'visualize_data_run' in st.session_state and st.session_state['visualize_data_run']:
            st.header("Data Visualization")
            fig1, fig2 = visualize_data(df, column_to_visualize)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.session_state['visualize_data_run'] = False

        if st.sidebar.button("Correlation Matrix", key='correlation_btn'):
            reset_all_flags()
            st.session_state['correlation_run'] = True

        if 'correlation_run' in st.session_state and st.session_state['correlation_run']:
            st.header("Correlation Matrix")
            fig = correlation_matrix(df)
            if fig is not None:
                st.pyplot(fig)
            st.session_state['correlation_run']= False

        if st.sidebar.button("Download dataset", key='download_btn'):
             download_dataset(df)

if __name__ == "__main__":
    main()