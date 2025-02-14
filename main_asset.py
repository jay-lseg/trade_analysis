# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:49:13 2024

@author: U6077840
"""

import pandas as pd
import datetime
import os
os.chdir(r'C:\Users\U6077840\OneDrive - London Stock Exchange Group\work_files\Value Project\LLM\Customized Report\dev')
from openai import AzureOpenAI
import yaml
from langchain_openai import AzureChatOpenAI
from news_download_asset import NewsAdd
from report_creation_cur import AnalysisProcessor
from pdf_creation_paragraph import PDFReportGenerator
import zip_ref
import utils
import numpy as np
#%%Initialization
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

#%%
#date input
tar_asset=asset_data.query("type=='company'").name.values
start_date='2024-10-01'
end_date='2024-10-31'
timezone='T00:00:00Z'

start_date_news=start_date+timezone
end_date_news=end_date+timezone
print(datetime.datetime.now(),start_date,end_date)

end_date_title=end_date.split('T')[0]
root_path=f'./data/{end_date_title}'

#Download news
news_d=NewsAdd(logger,start_date_news,end_date_news,root_path)
asset_code=asset_data.query("name in @tar_asset")['news_code'].values
asset_code_ls=list(set(asset_code))
download_dic={'company':asset_code_ls} 
folder_dic=news_d.news_download_arch(download_dic)
print(datetime.datetime.now(),'end')

#Daily Report Data Preparation
report_processor=AnalysisProcessor(logger,llm_client,llm,cfg['ds_username'],cfg['ds_password'])
task_ls=[[tar_asset,'currency',currency_data,root_path+'/currency_LLM_news.json',start_date,end_date,False]]
report_result=report_processor.report_arch(task_ls)



final_info=report_processor.pricing_table(company_data,cfg['ds_username'],cfg['ds_password'], end_date_title)
#Daily Report PDF Creation
#input value
pdf_c=PDFReportGenerator(logger,report_result, company_data, final_info,end_date_title)
pdf_c.generate_pdf()
#Output zip file
zip_file=f'{end_date_title}_ref.zip'
_=zip_ref.zip_ref(zip_file,root_path,report_result)
logger.info("ZIP file created successfully.")


