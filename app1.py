import streamlit as st
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Streamlit app
st.title('WeChat Investor Database - v0.1')

# File uploader
uploaded_file = st.file_uploader("上传 Citco 原始数据", type=["xlsx"])

st.divider()

user_ids = st.file_uploader("上传 Name 和 UID 对应文件", type=["xlsx"])

if user_ids is not None:
    # Read the Excel file into a DataFrame
    df = pd.read_excel(user_ids)

    # Assuming column A is 'Key' and column B is 'Value'
    user_ids = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()

tab1, tab2 = st.tabs(['净值报告','认购记录'])

with tab1:

    # Input parameters
    last_month_date = st.date_input('Enter last month date (YYYY/MM/DD):')
    this_month_date = st.date_input('Enter this month date (YYYY/MM/DD):')

    if uploaded_file is not None:
        # Load the raw data
        raw_data = pd.read_excel(uploaded_file)

        # Process data
        nav_statement_data = []

        for _, row in raw_data.iterrows():
            IA_NAME = row['IA_NAME']
            name = IA_NAME  # Assume Name is first letter of IA_NAME
            user_id = user_ids.get(name, None)
            if user_id is None:
                continue

            # Handle zero values
            if row['NAV_NET_VALUE_FROM'] == 0:
                row['NAV_NET_VALUE_FROM'] = 1000

            new_row = {
                'Name': name,
                'UserId': user_id,
                'Class': row['SPC_DESC'],
                'Product': row['SSS_DESC'],
                '期初单位净值 Date': last_month_date,
                '期初单位净值 净值': row['NAV_NET_VALUE_FROM'],
                '期末单位净值 Date': this_month_date,
                '期末单位净值 净值': row['NAV_NET_VALUE_TO'],
                '期间业绩表现': f"{(row['NAV_NET_VALUE_TO'] / row['NAV_NET_VALUE_FROM'] - 1) * 100:.2f}%",
                '单位数': row['HLD_SHR_BAL_TO'],
                '单位净值': row['NAV_NET_VALUE_TO'],
                '投资价值': row['HLD_NET_MRKT_VALUE_TO']
            }
            
            nav_statement_data.append(new_row)

        # Create a DataFrame for the new table
        new_table = pd.DataFrame(nav_statement_data)

        # Save to a new Excel file with openpyxl
        wb = Workbook()
        ws = wb.active

        # Write multi-level header manually
        ws.append(['Name', 'UserId', 'Class', 'Product', '期初单位净值', '', '期末单位净值', '', '期间业绩表现', '单位数', '单位净值', '投资价值'])
        ws.append(['', '', '', '', 'Date', '净值', 'Date', '净值', '', '', '', ''])

        # Write data
        for r in dataframe_to_rows(new_table, index=False, header=False):
            ws.append(r)

        # Adjust column widths (optional)
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter  # Get the column name
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[column].width = adjusted_width

        # Save file to a buffer
        from io import BytesIO
        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        # Download link
        st.download_button(
            label="Download Processed Data",
            data=buffer,
            file_name='nav_statement_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

with tab2:

        if uploaded_file is not None:
            raw_data = pd.read_excel(uploaded_file)

            raw_data['name_class'] = raw_data['IA_NAME'] + "&" + raw_data['SPC_DESC']

            # Process data
            subscription_record = raw_data[['IA_NAME', 'SPC_DESC', 'NET_AMT_IN', 'NET_AMT_OUT']]
            subscription_record['name_class'] = subscription_record['IA_NAME'] + "|" + subscription_record['SPC_DESC']


            subscription_record['net_amt'] = subscription_record['NET_AMT_IN'] + subscription_record['NET_AMT_OUT']

            temp = []

            for n in subscription_record['name_class'].unique():
                net_amount = subscription_record[subscription_record['name_class'] == n]['net_amt'].sum()
                temp.append(net_amount)

            sub_df = pd.DataFrame(subscription_record['name_class'].unique(), columns=['Name and Class'])
            sub_df['Net Amount'] = temp

            sub_df[['Name', 'Class']] = sub_df['Name and Class'].str.split('|', expand=True)

            sub_df.drop(columns=['Name and Class'], inplace=True)
            sub_df['UID'] = sub_df['Name'].map(user_ids)

            sub_df = sub_df[['Name', 'Class', 'UID','Net Amount']]

            st.dataframe(sub_df, height=1200)
        