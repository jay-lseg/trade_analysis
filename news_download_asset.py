# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:00:21 2024

@author: U6077840
"""

import pandas as pd
import requests
import os
import time
import json
import login
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
#%%

class NewsAdd:
    def __init__(self, logger,start_date,end_date, root_path,report_file_path = 'news.json',max_worker_thread=8,urgency=3):
        self.logger=logger
        self.root_path=root_path
        self.data_news_v1_story_main = "https://api.refinitiv.com/data/news/v1/stories/"
        self.urgency=urgency
        self.max_worker_thread=max_worker_thread
        self.report_file_path = report_file_path
        self.count=0
        self.start_date=start_date
        self.end_date=end_date
        #exclude news, filings, DCJ and translation
        self.exclude_code=['NS:GURU','NP:GURU','NP:GUR','NS:DJN','NS:DJCP','M:3H', 'E:1','E:4R','E:60','FH:4','FL:107']
        self.lock = threading.Lock()
    def query_creation(self,asset_type,asset_code,lan_filter,page_limit=100):
        query_line=''
        #create query info
        ##asset code
        asset_mod = list([str(i) for i in asset_code])
        asset_str = ' or '.join(asset_mod)
        query_line=f'({asset_str})'
        #language filter
        if lan_filter:
            query_line=f'{query_line} and {lan_filter}'
        #final
        if asset_type=='company':
            exclude_code_str=' or '.join(self.exclude_code)
            query_line=f'{query_line} and (not ({exclude_code_str}))&relevance=High&sort=newToOld&limit={str(page_limit)}&dateFrom={self.start_date}&dateTo={self.end_date}'
        elif asset_type=='country':
            query_line=f'{query_line} and NS:RTRS and (I:17 or A:2)&relevance=High&sort=newToOld&limit={str(page_limit)}&dateFrom={self.start_date}&dateTo={self.end_date}'
        elif asset_type=='currency':
            query_line=f'{query_line} and NS:RTRS&relevance=High&sort=newToOld&limit={str(page_limit)}&dateFrom={self.start_date}&dateTo={self.end_date}'
        else:
            print('abnormal stop')
            sys.exit()
        self.query_info = 'https://api.refinitiv.com/data/news/v1/headlines?query=' + query_line
        
    def news_download_arch(self,download_dic):
        folder_dic={}
        for type_i,code_ls in download_dic.items():
            self.asset_type=type_i
            #language filter
            if type_i=='country':
                lan_filter='(L:EN)'
            elif type_i=='company':
                lan_filter='(L:EN or L:ZH or L:JA)'
            elif type_i=='currency':
                lan_filter='(L:EN)'
                
            self.save_folder=self.root_path+'/'+type_i
            _=self.create_or_clear_folder(self.save_folder)
            folder_dic[type_i]=self.save_folder
            #create query info
            _=self.query_creation(type_i,code_ls,lan_filter)
            print(self.query_info)
            #get headline
            self.logger.info('Download news job starts')
            _=self.RDP_news_headline_loop()
            print('headline complete',len(self.news_headline))
            #get content
            _=self.get_content()
            #print('content complete')
            #parse news
            _=list([self.news_parse(i) for i in self.news_raw])
            #remove duplicated news
            _=self.clean_news_prep()    
            print(len(self.clean_llm_news))
            self.logger.info('cleand news num:'+str(len(self.clean_llm_news)))
            #save result
            self.report_save()
        return folder_dic 
    def create_or_clear_folder(self,path):
        if os.path.exists(path):
            # If the folder exists, clear its contents
            self.logger.info('folder exist')
        else:
            # Create a new folder
            os.makedirs(path)
    def RDP_news_headline_loop(self):
        query_info=self.query_info
        accessToken = login.getToken()
        resp = requests.request("GET", query_info, headers={"Authorization": "Bearer " + accessToken})
        resp_object = json.loads(resp.text)
        result = resp_object['data']
        # Search multiple pages
        while "next" in resp_object["meta"]:
            accessToken = login.getToken()
            cursor_this = resp_object["meta"]["next"]
            cursor_this = cursor_this.replace("+", "%2B")
            cursor_line = "https://api.refinitiv.com/data/news/v1/headlines?cursor=" + cursor_this
            resp = requests.request("GET", cursor_line, headers={"Authorization": "Bearer " + accessToken})
            resp_object = json.loads(resp.text)
            result = result + resp_object['data']
        self.news_headline=result
        return result

    def download_news(self, chunk):
        #headline_input=headline_raw["storyId"]
        for headline_input in chunk:
            accessToken = login.getToken()
            story_id=headline_input['storyId']
            if headline_input['newsItem']['contentMeta']['urgency']['$'] >= self.urgency:
                
                #topic code filter
                # code_ls=list([i['_qcode'] for i in headline_input['newsItem']['contentMeta']['subject']])
                # overlap = set(code_ls) & set(self.exclude_code)
                # if overlap:
                #     continue
                
                request_this = self.data_news_v1_story_main + story_id
                error_cnt=0
                
                fail=True
                while error_cnt<=3 and fail:
                    resp = requests.request("GET", request_this, headers={"Authorization": "Bearer " + accessToken})
                    news_i = resp.text
                    if 'Request per second limit' in news_i:
                        #print(news_i,error_cnt)
                        time.sleep(5)
                        error_cnt+=1
                    else:
                        fail=False
                #work check & raw news save
                #self.lock.acquire()
                self.news_raw.append(json.loads(news_i))
                #self.lock.release()
    #parse news information
    def news_parse(self, news_i):
        if 'newsItem' in news_i.keys():
            _=self.news_parse_processing(news_i['newsItem'])
        else:
            for news_j in news_i['data']:
                _=self.news_parse_processing(news_j)
                
    def news_parse_processing(self,news_i):
        company_name=[]
        company_permid=[]
        market_codes=[]
        
        headline = news_i['itemMeta']['title'][0]['$']
        # check if content exist
        urgency = news_i['contentMeta']['urgency']['$']
        if urgency == 1:
            content = headline
        else:
            content = news_i['contentSet']['inlineData'][0]['$']
        #combine headline
        content=headline+'  \n '+content
        # subjects
        try:
            subject = news_i['contentMeta']['subject']
        except:
            return None
        lan = news_i['contentMeta']['language'][0]['_tag']
        timestamp = news_i['itemMeta']['versionCreated']['$']
        guid = news_i['_guid']
        # default dupekey
        dedupe_key = None
        # dupid
        try:
            for dup_item in news_i['contentMeta']['contentMetaExtProperty']:
                try:
                    if 'dedupeKey:strict' in dup_item['_qcode']:
                        dedupe_key = dup_item['_qcode']
                        break
                except:
                    pass
        except:
            pass
        news_source = news_i['contentMeta']['contributor'][0]['_qcode']

        metadata = {
            'source': 'news',
            'contributor': news_source,
            'documentarrivedate_unix': int(datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=datetime.timezone.utc).timestamp()),
            'docid': guid,
            'headline': headline,
            'language': lan,
            'company': company_name,
            'company_permid': company_permid,
            'market_codes':market_codes,
            'dedupe_key': dedupe_key,
            'subjects':subject
        }

        if None in metadata.values():
            for key_i, value_i in metadata.items():
                if value_i == None:
                    metadata[key_i] = 'not_available'
        #self.lock.acquire()
        self.news_llm.append([metadata, content])
        #self.lock.release()
        
    #remove duplicated news
    def news_clean(self, meta_df):
        null_index = meta_df[meta_df.dedupe_key.isnull()].index.tolist()

        meta_df['index_value'] = meta_df.index
        meta_df.sort_values(by='documentarrivedate_unix', inplace=True)
        clean_index = meta_df.groupby('dedupe_key').last().index_value.tolist()

        clean_index = set(null_index + clean_index)
        return clean_index
        
        
    def get_content(self,news_chunk=None):
        self.news_raw = []
        self.news_llm = []
        if news_chunk:
            self.news_headline=self.news_headline[news_chunk[0]:news_chunk[1]]

        chunks = [self.news_headline[i::self.max_worker_thread] for i in range(self.max_worker_thread)]
        with ThreadPoolExecutor(max_workers=self.max_worker_thread) as executor:
            futures = [executor.submit(self.download_news, chunk) for chunk in chunks]
            for future in as_completed(futures):
                # all thread finished
                result = future.result()

    def clean_news_prep(self):
        # remove duplicated news
        meta_ls = list([i[0] for i in self.news_llm])
        meta_df = pd.DataFrame(meta_ls)

        selected_index = self.news_clean(meta_df)
        self.clean_llm_news = list([self.news_llm[i] for i in selected_index])
    
    def report_save(self):
        #create file path
        dest_path_file=self.root_path+'/'+self.asset_type+'_raw_news.json'
        with open(dest_path_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.news_raw))

        #Prepare research data for LLM application
        ##create file path
        dest_path_file=self.root_path+'/'+self.asset_type+'_LLM_news.json'
        with open(dest_path_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.clean_llm_news))
        #save indpendent news
        for file_i in self.clean_llm_news:
            docid=file_i[0]['docid']
            content=file_i[1]
            docid_name=docid.replace(':','-').replace('.','---').replace(',','__')
            with open(self.save_folder+'/'+docid_name+'.txt','w',encoding='utf-8')as f:
                f.write(content)
#%%
if __name__ == '__main__':
    tar_currency=['United States Dollar to Euro','CHINESE YUAN TO US Dollar']
    start_date='2024-11-03'
    end_date='2024-11-03'
    
    start_date_news=start_date+'T00:00:00Z'
    end_date_news=end_date+'T01:00:00Z'
    print(datetime.datetime.now(),start_date,end_date)
    
    end_date_title=end_date.split('T')[0]
    root_path=f'./data/{end_date_title}'
    
    #Download news
    news_d=NewsAdd(logger,start_date_news,end_date_news,root_path)
    currency_code=currency_data.query("name in @tar_currency")['news_code'].values
    currency_code_str=','.join(currency_code)
    currency_code_ls=list(set(currency_code_str.split(',')))
    download_dic={'currency':currency_code_ls} 
    folder_dic=news_d.news_download_arch(download_dic)