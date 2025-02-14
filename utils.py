# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:11:51 2024

@author: U6077840
"""
import pandas as pd
import logging

def info_aprse(file_path,asset_type):
    data=pd.read_csv(file_path)
    if asset_type=='company':
        code=data.query("type=='company'")['permid'].values
        name=data.query("type=='company'")['name'].values
    elif asset_type=='country':
        code=data.code.values
        name=data.name.values
    return data, code, name

def get_logger(filename):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filename=filename, 
                        filemode='w')
    logger = logging.getLogger()
    return logger
def get_ref_all(llm_client):
    client_obj=llm_client.files.list().data
    file_all=list([[file.id,file.filename] for file in client_obj])
    ref_all=pd.DataFrame(file_all,columns=['id','name'])
    return ref_all

