# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:01:41 2025

@author: U6077840
"""
#%%input
import pandas as pd
import streamlit as st
import datetime
from openai import AzureOpenAI
import yaml
from langchain_openai import AzureChatOpenAI
from report_creation_asset_v2 import AnalysisProcessor
import utils
import altair as alt
#%%
logger=utils.get_logger('daily_report.log')
with open('./config.yaml') as f:
    cfg=yaml.safe_load(f)

llm_client = AzureOpenAI(
    azure_endpoint = cfg["AZURE_ENDPOINT"],    
    api_key=cfg["AZURE_OPENAI_API_KEY"],
    api_version=cfg['API_VERSION']
)

azure_llm_deployment='gpt-4o'
llm_name = 'gpt-4o'
api_version="2024-05-01-preview"
azure_embedding_deployment='embedding'
temparature=0
seed=42
llm = AzureChatOpenAI(azure_endpoint=cfg['AZURE_ENDPOINT'],azure_deployment=azure_llm_deployment,model_name=llm_name, 
                      temperature=temparature, api_version=api_version,seed=seed,api_key=cfg["AZURE_OPENAI_API_KEY"])

asset_file='asset_code.csv'
asset_data=pd.read_csv(asset_file)
root_path='./data/2025-01-20'
currency_pairs=asset_data.query("type=='currency'").name.values
company_values=asset_data.query("type=='company'").name.values
#%%
#input
p_language=st.session_state.get("language", None)
p_source_language=st.session_state.get("source_language", None)
p_input_date=st.session_state.get("input_date", None)
p_selected_asset=st.session_state.get("selected_asset", None)
p_mrr_value=st.session_state.get("mrr_value", None)
p_bullet_num=st.session_state.get("bullet_num", None)

asset_type=st.session_state.get("asset_type", None)
language=st.session_state.get("language", None)
source_language=st.session_state.get("source_language", None)
input_date=st.session_state.get("input_date", None)
selected_asset=st.session_state.get("selected_asset", None)
mrr_value=st.session_state.get("mrr_value", None)
report_result=st.session_state.get("report_result", None)
sel_headline=st.session_state.get("sel_headline", None)
news_num=st.session_state.get("news_num", None)
bullet_num=st.session_state.get("bullet_num", None)

# Sidebar - Time range selector
st.sidebar.header('Settings')
asset_type = st.sidebar.selectbox('Select asset_type', ['currency','company'])
st.session_state.asset_type = asset_type

language_ls=[ 'English',
 '简体中文', 
 '繁體中文', 
 '日本語'
]
language = st.sidebar.selectbox('Select language', language_ls,)
st.session_state.language = language

if asset_type=='currency':
    source_language_ls=['all','en']
elif asset_type=='company':
    source_language_ls=[ 'all',
     'en', 
     'ja', 
     'zh-Hans',
     'zh-Hant'
    ]
source_language = st.sidebar.selectbox('Select source language', source_language_ls,)
st.session_state.source_language = source_language

default_start_date = datetime.datetime(2024, 12, 2)
default_end_date = datetime.datetime(2024, 12, 8)
search_start_date = datetime.datetime(2024, 10, 1)
search_end_date = datetime.datetime(2024, 12, 31)
# default all time period
input_date = st.sidebar.date_input(
    "Source time range",
    (default_start_date, default_end_date),
    search_start_date,
    search_end_date,
    format="YYYY-MM-DD",
)
st.session_state.input_date = input_date
if len(input_date)==2:
    # Extract start_date and end_date from input_date
    start_date, end_date = input_date
    # Convert start_date and end_date to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Sidebar - Currency pair selector
    if asset_type=='currency':
        asset_ls=currency_pairs
    elif asset_type=='company':
        asset_ls=company_values
    selected_asset = st.sidebar.multiselect('Select asset', asset_ls,None)
    st.session_state.selected_asset = selected_asset
    
    mrr_value = st.sidebar.slider('Select answer diversity value', min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    st.session_state.mrr_value = mrr_value
    bullet_num= st.sidebar.slider('Select maximum number of bullet points for analysis', min_value=1, max_value=20, value=5, step=1)
    st.session_state.bullet_num = bullet_num
    #top_n=15
    #if p_asset_type==asset_type and p_language=language and 
    if selected_asset:
        if p_language==language and p_input_date==input_date and p_mrr_value==mrr_value and p_selected_asset==selected_asset and report_result and p_bullet_num==bullet_num and p_source_language==source_language:
            pass
        else:
            #result Generation
            report_processor=AnalysisProcessor(mrr_value,logger,llm_client,llm,cfg['ds_username'],cfg['ds_password'],language,bullet_num)
            task_ls=[[selected_asset,asset_type,asset_data,root_path+'/'+asset_type+'_LLM_news.json',start_date_str,end_date_str,False,source_language]]
            report_result=report_processor.report_arch(task_ls)
            news_num=report_processor.news_num
            
            st.session_state.report_result = report_result
            st.session_state.news_num = news_num
            
        cur_num=1
        headline_ls=[]
        for key_i,value_i in report_result[asset_type].items():
            dup_num=2
            a_news_num=news_num[key_i]
            st.markdown(f'Analyzed {a_news_num} pieces of news for {key_i}')
            # Filter pricing table based on selected time range
            filtered_pricing_table = value_i['pricing']
            filtered_pricing_table.index.name='date'
            
            st.markdown(f'### Asset: {key_i}')
            # Main page - Display price trend as a graph
            #st.line_chart(filtered_pricing_table['price'])
            
            chart = alt.Chart(filtered_pricing_table.reset_index()).mark_line().encode(
            x='date:T',
            y=alt.Y('price:Q', scale=alt.Scale(domain=[filtered_pricing_table['price'].min(), filtered_pricing_table['price'].max()]))
            ).properties(
            title='Price Trend'
            )
            
            if 'turn_info' in value_i.keys():
                for key_j,value_j in value_i['turn_info'].items():
                    if len(value_j):
                        if key_j=='peak':
                            color='red'
                        else:
                            color='green'
                        annotation=pd.DataFrame(value_j,columns=['date','price','note'])
    
                        # Highlight points with notes
                        highlight = alt.Chart(annotation).mark_point(
                            size=100,  # Increase the size of the points
                            color=color  # Change the color of the points
                        ).encode(
                            x='date:T',
                            y='price:Q',
                            tooltip=['date:T', 'price:Q', 'note']
                        )
                        chart = chart + highlight

                
            st.altair_chart(chart, use_container_width=True)
            
            # Calculate analytics
            latest_price = filtered_pricing_table['price'].iloc[-1]
            return_in_period = (filtered_pricing_table['price'].iloc[-1] / filtered_pricing_table['price'].iloc[0] - 1) * 100
            highest_price = filtered_pricing_table['price'].max()
            lowest_price = filtered_pricing_table['price'].min()
            
            # Main page - Display calculated analytics
            
            st.write(f'**Latest Price:** {latest_price:.2f}')
            st.write(f'**Return in the Period:** {return_in_period:.2f}%')
            st.write(f'**Highest Price:** {highest_price:.2f}')
            st.write(f'**Lowest Price:** {lowest_price:.2f}')
            
            print(value_i)
            # Main page - Display analysis from dictionary
            st.subheader('Analysis')
            st.write('**Historical Trend Analysis**')
            st.write(value_i['summary']['trend_analysis'])
            st.write('**Future View**')
            st.write(value_i['summary']['future_view'])
            
            # Main page - Display news headlines and bodies
            st.markdown("### Reference")
            raw_file_path=root_path+'/'+asset_type
            for ref_i in value_i['citations']:
                raw_news_path=raw_file_path+'/'+ref_i
                with open(raw_news_path,'r',encoding='utf-8')as f:
                    raw_news=f.read()
                    
                # Split the content into two parts by the first newline character
                parts = raw_news.split('\n', 1)
        
                # Extract the two parts
                headline = str(cur_num)+'-'+parts[0].strip()
                body = parts[1].strip() if len(parts) > 1 else ''
                
                #headline processing
                while headline in headline_ls:
                    headline=headline+'-'+str(dup_num)
                    dup_num+=1
                headline_ls.append(headline)
                
                if st.button(headline):
                    st.session_state.sel_headline = headline
                #print(headline,sel_headline)
                if sel_headline==headline:
                    body=body.replace('$', '\\\\$')
                    st.markdown(body)
            cur_num+=1