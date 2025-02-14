# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:39:57 2024

@author: U6077840
"""
import json
import pandas as pd
import DatastreamPy as dsws
import re
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time
import datetime
from scipy.signal import find_peaks

class AnalysisProcessor:
    def __init__(self,mrr_value,logger,llm_client,llm,ds_username,ds_password,language,bullet_num=5,timezone=8,top_n=15,prominance_per=60,distance=2,days_back=2,days_after=1):
        self.llm=llm
        self.logger=logger
        self.llm_client = llm_client
        #adjust 2 multi threading processes
        self.max_worker=6
        self.max_worker_sys=8
        #for MRR
        self.top_n=top_n
        self.diversity_param=mrr_value
        #for embedding
        self.embedding_model='embedding-small'
        #for DSSCAN
        self.duplicate_eps=0.1
        self.cluster_eps=0.2
        self.minpts=2
        self.count=0
        self.lock=threading.Lock()
        self.ds_username=ds_username
        self.ds_password=ds_password
        self.timezone=timezone
        self.prominence_per=prominance_per
        # self.width=width
        # self.threshold=threshold
        self.distance=distance
        self.days_back=days_back
        self.days_after=days_after
        
        self.parse_model='gpt-4o'
        self.bullet_num=bullet_num
        self.language=language
        self.news_num={}

        
    def report_arch(self,task_ls):
        #process report content
        sorted_dic={}
        self.result_dic={}
        for tar_asset,asset_type,asset_data,llm_file,start_date,end_date,cluster_ind,source_language in task_ls:
            self.asset_data=asset_data
            self.cluster_ind=cluster_ind
            self.tar_asset=tar_asset
            self.result_dic[asset_type]={}
            self.source_language=source_language
            _=self.analysis_prep(asset_type,asset_data,llm_file,start_date,end_date)
            
            
            sorted_dic[asset_type]={}
                          
        #sort dictionary
        for key_i,value_j in self.result_dic.items():
            sorted_keys = sorted(self.result_dic[key_i].keys())
            sorted_dic[key_i] = {key: self.result_dic[key_i][key] for key in sorted_keys}
        return sorted_dic
       
    def analysis_prep(self,asset_type,asset_data,llm_file,start_date,end_date,max_token=10000,relevance=2):
        self.result_raw_dic={}            
             
        with open(llm_file,'r',encoding='utf-8') as f:
            llm_data=json.loads(f.read())
        self.llm_data=llm_data   
        #prepare paired value
        asset_data=asset_data.query("name in @self.tar_asset")
        if asset_type=='country':
            asset_data_sel=asset_data.query("type=='country'")
            asset_data_sel['news_code']=asset_data_sel['news_code'].apply(lambda x: x.split(','))
            asset_bulk=list(zip(asset_data_sel['news_code'],asset_data_sel['name']))
        elif asset_type=='company':
            asset_data_sel=asset_data.query("type=='company'")
            asset_data_sel['news_code']=asset_data_sel['news_code'].apply(lambda x: x.split(','))
            asset_bulk=list(zip(asset_data_sel['news_code'],asset_data_sel['name']))  
        elif asset_type=='currency':
            asset_data_sel=asset_data.query("type=='currency'")
            asset_data_sel['news_code']=asset_data_sel['news_code'].apply(lambda x: x.split(','))
            asset_bulk=list(zip(asset_data_sel['news_code'],asset_data_sel['name']))
        else:
            pass
            
        #select news in parellel
        worker_num_news=min(self.max_worker,len(asset_bulk))
        with ThreadPoolExecutor(max_workers=worker_num_news) as executor:
            futures = [executor.submit(self.news_selection_asset, chunk,llm_data,asset_type,relevance,worker_num_news,start_date,end_date) for chunk in asset_bulk]
            for future in as_completed(futures):
                # all thread finished
                result=future.result()
                if result:
                    self.result_raw_dic.update(result)
        #print('ok1',datetime.datetime.now())        
        news_select=self.content_dedupe(self.result_raw_dic)
        #print('ok2',datetime.datetime.now())
        _=self.news_cluster(news_select,asset_type,start_date,end_date)
    #content deduplication
    def content_dedupe(self,result_raw_dic):
        news_select=[]
        if len(result_raw_dic):
            news_temp_ls = list([[key, value['embedding'],value['ref'],value['asset_name']] for key, value in result_raw_dic.items()])
            news_embedding=np.array(([i[1] for i in news_temp_ls]))
            news_group=self.dbscan(news_embedding,self.duplicate_eps,self.minpts)
            #print(news_group)
            if len(news_group)>0:
                for key_i,group_ls in news_group.items():
                #find longest text
                    text_len=0
                    text_sel=0
                    for item_i in group_ls:
                        item_len=len(news_temp_ls[item_i])
                        if item_len>text_len:
                            text_len=item_len
                            text_sel=item_i
                    news_select.append(news_temp_ls[text_sel])
        return news_select
        
        #news clustering
    def news_cluster(self,news_select,asset_type,start_date,end_date):
        if len(news_select)>0:
            #print(news_group)
            company_cluster={}
            if self.cluster_ind:
                #cluster news
                news_embedding=np.array(([i[1] for i in news_select]))
                news_group=self.dbscan(news_embedding,self.cluster_eps,self.minpts)
                for key_i,group_ls in news_group.items():
                    sel_summary=''
                    sel_asset_name=''
                    #single item no need for clustering
                    if len(group_ls)==1:
                        item_i=group_ls[0]
                        sel_summary=[news_select[item_i][0]]
                        sel_asset_name=news_select[item_i][3]
                        ref_ls=[news_select[item_i][2]]
    
                    else:
                        #multiple items, process clustering
                        sub_company_ls=list([news_select[c_i][3] for c_i in group_ls])
                        counter = Counter(sub_company_ls)
                        # Find the item with the highest count
                        most_common_item, count = counter.most_common(1)[0]
                        ref_ls=list([news_select[c_i][2] for c_i in group_ls])
                        content_ls=list([news_select[c_i][0] for c_i in group_ls])
                        sel_asset_name=most_common_item
                        sel_summary=content_ls
                    
                    #temp result
                    if sel_asset_name in company_cluster.keys():
                        pass
                    else:
                        company_cluster[sel_asset_name]={}
                        company_cluster[sel_asset_name]['item']=[]
                        company_cluster[sel_asset_name]['ref']=[]
                    company_cluster[sel_asset_name]['item']+=sel_summary
                    company_cluster[sel_asset_name]['ref']+=ref_ls
            #no clustering
            else:
                for news_i in news_select:
                    sel_summary=[news_i[0]]
                    sel_asset_name=news_i[3]
                    ref_ls=[news_i[2]]
                    
                    #temp result
                    if sel_asset_name in company_cluster.keys():
                        pass
                    else:
                        company_cluster[sel_asset_name]={}
                        company_cluster[sel_asset_name]['item']=[]
                        company_cluster[sel_asset_name]['ref']=[]
                    company_cluster[sel_asset_name]['item']+=sel_summary
                    company_cluster[sel_asset_name]['ref']+=ref_ls                    
                    
            #final result generation
            for asset_i, value_i in company_cluster.items():
                turn_info=None
                # Find the item with the highest count
                ref_ls=value_i['ref']
                content_ls=value_i['item']
                #print(asset_i)
                pricing,price_trend=self.price_trend_fun(asset_i,start_date,end_date)
                #turning point explanation
                if len(pricing)>5:
                    turn_info=self.turning_point(pricing,asset_i,asset_type)
                if asset_type=='currency':
                    base_currency=asset_i.split('TO')[1].strip()
                    quote_currency=asset_i.split('TO')[0].strip()
                    consol_prompt=f'''Below is a list of documents:
                        {content_ls}
                    The price trend for {asset_i} over the time period is: 
                        {price_trend}
                    In the price description, the base currency is {base_currency}. If {asset_i} goes up, {base_currency} will be stronger.
                    As a professional financial analyst, use the information in the above documents to explain the price trend for {asset_i}.
                    The analysis will be in bullet point format with at most {self.bullet_num} points.Each bullet point has 2 components, topic name and details.
                    Do not write generale statement. Each bullet point should include specific event content.
                    No need to give the reference.No need to describe the price trend again in the answer.
                    Make the answer just based on the input documents and information.
                    Reply in language of {self.language}
                   '''
                    consol_prompt2=f'''Below is a list of documents:
                        {content_ls}
                    As a professional financial analyst, use the information in the above documents to predict the future price trend for {asset_i}, up, down or neutral with more than 80% confidence.
                    The base currency is {base_currency}. If {asset_i} goes up, it means {base_currency} will be stronger.
                    You should give the succinct reason to support your prediction in 50 words. The reason should be consistent with your prediction.
                    Generate succinct answer directly. Do not add general statement or description.
                    Reply in language of {self.language}
                   '''
                elif asset_type=='company':
                    consol_prompt=f'''Below is a list of documents:
                        {content_ls}
                    The price trend for {asset_i} over the time period is: 
                        {price_trend}
                    As a professional financial analyst, use the information in the above documents to explain the price trend for {asset_i}.
                    The analysis will be in bullet point format with at most {self.bullet_num} points.Each bullet point has 2 components, topic name and details.
                    Do not write generale statement. Each bullet point should include specific event content.
                    No need to give the reference.No need to describe the price trend again in the answer.
                    Make the answer just based on the input documents and information.
                    Reply in language of {self.language}
                   '''
                    consol_prompt2=f'''Below is a list of documents:
                        {content_ls}
                    As a professional financial analyst, use the information in the above documents to if the future outlook is postive, negative or neutral for {asset_i} with more than 80% confidence.
                    You should give the succinct reason to support your prediction in 50 words.
                    Generate succinct answer directly. Do not add general statement or description.
                    If there is no relevant information, just say no relevant information.
                    Reply in language of {self.language}
                   '''
                   
                error_cnt=0
                protection=True
                while error_cnt<=3 and protection:
                    try:
                        message=self.llm.invoke(consol_prompt).content
                        message2=self.llm.invoke(consol_prompt2).content
                        
                        protection=False
                    except Exception as e:
                        error_cnt+=1
                        time.sleep(60)
                        print('final error',error_cnt,e)
                #parsed_value=self.result_parse_summary(message)
                #parsed_value2=self.result_parse_summary(message2)
                # if len(parsed_value) > 0 and len(parsed_value['summary'])>0:
                #     #validation test
                #     mess_val=self.summary_val(message)
                #     if mess_val:
                #print(self.result_dic)
                if asset_i in self.result_dic[asset_type].keys():
                    pass
                else:
                    self.result_dic[asset_type][asset_i]={}   
                pricing.index=pd.to_datetime(pricing.index)
                pricing.columns=['price']
                self.result_dic[asset_type][asset_i]['summary']={}
                self.result_dic[asset_type][asset_i]['summary']['trend_analysis']=message
                self.result_dic[asset_type][asset_i]['summary']['future_view']=message2
                self.result_dic[asset_type][asset_i]['citations']=ref_ls
                self.result_dic[asset_type][asset_i]['pricing']=pricing
                if turn_info:
                    self.result_dic[asset_type][asset_i]['turn_info']=turn_info

    def turning_point(self,pricing,asset_i,asset_type):
        result={'peak':[],'trough':[]}
        asset_code=self.asset_data.query("name==@asset_i")['news_code'].values[0].split(',')
        #paramter cal
        value_diff=abs(np.diff(pricing.iloc[:,0].values))
        prominence=np.percentile(value_diff,self.prominence_per)
        # Find peaks (local maxima)
        peaks, _ = find_peaks(pricing.iloc[:,0],prominence=prominence,distance=self.distance)
        peaks_ls=list([[pricing.index[index],pricing.iloc[index, 0]] for index in peaks])
        #print('ok3',peaks_ls)
        if asset_type=="currency":
            base_currency=asset_i.split('TO')[1].strip()
            quote_currency=asset_i.split('TO')[0].strip()
            prompt=f"""
            {asset_i} price trend incurred peak point. Based on the input documents, explain why {base_currency} is becoming weaker compared to {quote_currency}.
            Do not write generale statement or just describe market price trend.
            Make the answer just based on the input documents and information.
            Make answer in 10 words.
            Try our best to find relevant information to explain. If there is no relevant information, just reply no.
            Reply in language of {self.language}"""
        elif asset_type=="company":
            prompt=f"""
            {asset_i} price trend incurred peak point. Based on the input documents, explain why the price goes down in 10 words.
            Do not write generale statement or just describe market price trend.
            Make the answer just based on the input documents and information.
            Make answer in 10 words.
            Try our best to find relevant information to explain. If there is no relevant information, just reply no.
            Reply in language of {self.language}"""
        for key_i,price_i in peaks_ls:
            start_date=self.get_previous_date(key_i,self.days_back)
            end_date=self.get_previous_date(key_i,-1*self.days_after)
            explain=self.info_summary(prompt,start_date,end_date,asset_i,asset_code,asset_type)
            if explain:
                result['peak'].append([key_i,price_i,explain])
        # Find troughs (local minima) by inverting the price series
        troughs, _ = find_peaks(-pricing.iloc[:,0],prominence=prominence,distance=self.distance)
        troughs_ls=list([[pricing.index[index],pricing.iloc[index, 0]] for index in troughs])
        #print('ok4',troughs_ls)
        if asset_type=="currency":
            base_currency=asset_i.split('TO')[1].strip()
            quote_currency=asset_i.split('TO')[0].strip()
            prompt=f"""
            {asset_i} price trend incurred peak point. Based on the input documents, explain why {base_currency} is becoming stronger compared to {quote_currency}.
            Do not write generale statement or just describe market price trend.
            Make the answer just based on the input documents and information.
            Make answer in 10 words.
            Try our best to find relevant information to explain. If there is no relevant information, just reply no.
            Reply in language of {self.language}"""
        elif asset_type=="company":
            prompt=f"""
            {asset_i} price trend incurred trough point. Based on the input documents, explain why the price goes up.
            Do not write generale statement or just describe market price trend.
            Make the answer just based on the input documents and information.
            Make answer in 10 words.
            Try our best to find relevant information to explain. If there is no relevant information, just reply no.
            Reply in language of {self.language}"""
        for key_i,price_i in troughs_ls:
            start_date=self.get_previous_date(key_i,self.days_back)
            end_date=self.get_previous_date(key_i,-1*self.days_after)
            explain=self.info_summary(prompt,start_date,end_date,asset_i,asset_code,asset_type)
            if explain:
                result['trough'].append([key_i,price_i,explain])
        return result
        
    def info_summary(self,prompt,start_date,end_date,asset_i,asset_code,asset_type):
        asset_pair=[asset_code,asset_i]
        relevance=2
        worker_num_news=0
        #collect news information
        result_raw_dic=self.news_selection_asset(asset_pair, self.llm_data, asset_type, relevance, worker_num_news, start_date, end_date)
        #content dedupe
        news_select=self.content_dedupe(result_raw_dic)
        
        news_ls=list([i[0] for i in news_select])
        if len(news_ls)>0:
            error_cnt=0
            protection=True
            prompt=f"""Below is a list of documents:
                {news_ls}
                {prompt}"""
            while error_cnt<=3 and protection:
                try:
                    message=self.llm.invoke(prompt).content             
                    protection=False
                except Exception as e:
                    error_cnt+=1
                    time.sleep(60)
                    print('final error',error_cnt,e)
            mess_val=self.summary_val(message)
        
        print(message,mess_val)
        
        if mess_val:
            return message
        
        return None
    def get_previous_date(self,date_str,num_days):
        # Convert the string to a datetime object
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        # Subtract one day
        previous_date_obj = date_obj - datetime.timedelta(days=num_days)
        # Convert the datetime object back to a string
        previous_date_str = previous_date_obj.strftime("%Y-%m-%d")
        return previous_date_str
    def dbscan(self,news_embedding,eps,min_pts):
        #apply DBSCAN
        cosine_distance_matrix=cosine_distances(news_embedding)
        ##protenction for one group--placeholder
        dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_pts)
        clusters = dbscan.fit_predict(cosine_distance_matrix)
        news_group={}
        out_group=max(clusters)+1
        for index_i,group_i in enumerate(clusters):
            if group_i==-1:
                news_group[out_group]=[index_i]
                out_group+=1
            else:
                if group_i in news_group.keys():
                    pass
                else:
                    news_group[group_i]=[]
                news_group[group_i].append(index_i)
        return news_group
    #select news
    def news_selection_asset(self,asset_pair,llm_data,asset_type,relevance,worker_num_news,start_date,end_date):   
        asset_code=asset_pair[0]
        asset_i=asset_pair[1]
        result_raw_dic={}
        
        #news selection. 2 items in return, news content and news id
        #print(asset_code,start_date,end_date,relevance)
        news_select=self.news_selection(llm_data,asset_code,start_date,end_date,relevance=relevance)
        self.news_num[asset_i]=len(news_select)
        print(asset_code,len(news_select))
        asset_news_temp=[]
        #generate summary for each piece of news
        t_result=[]
        if news_select:
            asset_news_temp=news_select
            
            con_len=len(asset_news_temp)
            if con_len>0:
                #deal with target embedding
                if asset_type=='country':
                    tar_prompt=f"""Infomration related to {asset_i}"""
                elif asset_type=='company':
                    tar_prompt=f"""Information related to {asset_i}"""
                elif asset_type=='currency':
                    tar_prompt=f"""Infomration related to {asset_i}"""
                tar_embedding=self.embedding(tar_prompt)
                #content emebedding
                content_text=list([a_i[0] for a_i in asset_news_temp])
                t_result=[]
                
                with ThreadPoolExecutor(max_workers=min(self.max_worker_sys-worker_num_news,con_len)) as executor:
                    futures = [executor.submit(self.embedding, chunk) for chunk in content_text]
                    for future in as_completed(futures):
                        # all thread finished
                        t_result.append(future.result())
                
                #news_temp_ls = list([[key, value['embedding'],value['ref']] for key, value in self.news_temp.items()])
                for i in range(len(t_result)):
                    asset_news_temp[i].append(t_result[i])
                if con_len<self.top_n:
                    select_content_ar=np.array(asset_news_temp,dtype=object)
                else:
                    #use MMR to select top 10
                    content_embedding=list([i[2] for i in asset_news_temp])
                    select_content_index=self.mmr(tar_embedding,content_embedding,self.top_n,self.diversity_param)
                    #print(select_content_index)
                    select_content_ar=np.array(asset_news_temp,dtype=object)[select_content_index]
                #print(list([i[0] for i in select_content_ar]))
                for i in select_content_ar:
                    result_raw_dic[i[0]]={}
                    result_raw_dic[i[0]]={
                        'asset_name':asset_i,
                        'ref':i[1],
                        'embedding':i[2],
                        }
        return result_raw_dic
    def embedding(self,text):
        embedding=self.llm_client.embeddings.create(input=[text],model=self.embedding_model).data[0].embedding
        return embedding
    #generate summary for each piece of news
    def summary_generator_piece(self,news_pair,asset_i,asset_type):
        result_piece=None
        if asset_type=='country':
            input_prompt=f'''Below is a piece of news:
                {news_pair[0]}
            As company strategy analyst, based on the above news, try best to find and summarize important events specifically related to {asset_i} capital market, especially events which can be used to explain the capital markets movement.
            The summary will include at most {self.point_num} different and relevant bullet points. It can include summaries for multiple events. It should follow the format as below, starting with Summary:
            Summary: <meaningful and specific summarized text in bullet point format> 
            Do not include technicall indicator analysis for stock price or stock price movement description in the summary.
            If there is no relevant information or there is just market commentary or description of technical indicators in the input, just say no.
            '''
        elif asset_type=='companmy':
            input_prompt=f'''Below is a piece of news:
                {news_pair[0]}
             As company strategy analyst, based on the above news, try best to find and summarize important events specifically related to {asset_i} business, especially events which have significant impact on the company business operation.
             The summary will be a cohesive paragraph within 100 words. It can include summaries for multiple events. It should follow the format as below, starting with Summary:
             Summary: <meaningful and specific summarized text> 
             Do not include technicall indicator analysis for stock price or stock price movement description in the summary.
             If there is no relevant information or there is just market commentary or description of technical indicators in the input, just say no.
            ''' 
        elif asset_type=='currency':
            input_prompt=f'''Below is a piece of news:
                {news_pair[0]}
             As professional financial analyst, based on the above news, try best to find and summarize important events specifically related to {asset_i}.
             The summary will be a cohesive paragraph within 100 words. It can include summaries for multiple events. It should follow the format as below, starting with Summary:
             Summary: <meaningful and specific summarized text> 
             Do not include technicall indicator analysis for currency price movement description in the summary.
             If there is no relevant information or there is just market commentary or description of technical indicators in the input, just say no.
            ''' 
        else:
            pass
        error_cnt=0
        protection=True
        while error_cnt<=3 and protection:
            try:
                message=self.llm.invoke(input_prompt).content
                protection=False
            except Exception as e:
                self.lock.acquire()
                print('error',error_cnt,e)
                error_cnt+=1
                time.sleep(60)
                self.lock.release()
        #message validation
        #print(message)
        mess_val=self.summary_val(message)
        if mess_val:
            result_piece=[message,news_pair[1]]
        return result_piece
    def mmr(self,tar_embedding,content_embedding,top_n,diversity_param):
        """
        :param doc_embedding: the embedding of the query document
        :param doc_embeddings: the embeddings of the candidate documents
        :param diversity_param: parameter to control the trade-off between relevance and diversity
        :param top_n: number of top documents to return
        """
        
        # Calculate cosine similarity between the query and the candidate documents
        sim_to_query = cosine_similarity([tar_embedding], content_embedding)[0]
        
        self.sim_to_query=sim_to_query
        # Initialize the selected documents list and candidates list
        selected_docs = []
        candidates = list(range(len(content_embedding)))
        #print(sim_to_query)
        for _ in range(top_n):
            mmr_score = []

            for doc_index in candidates:
                sim_to_selected = max([cosine_similarity([content_embedding[doc_index]], [content_embedding[sel_doc]])[0][0] for sel_doc in selected_docs]) if selected_docs else 0
                mmr_score.append((1 - diversity_param) * sim_to_query[doc_index] - diversity_param * sim_to_selected)
            #print(mmr_score)
            # Select the document with the highest MMR score
            selected_index = candidates[np.argmax(mmr_score)]
            selected_docs.append(selected_index)
            candidates.remove(selected_index)
        return selected_docs
    def summary_val(self,summary):
        error_cnt=0
        val_result=None
        while error_cnt<=3 and val_result==None:
            try:
                #print(error_cnt)
                messages = [{"role": "user", "content": summary}]
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "summary_val",
                        "description": "detect if there is meaningful and valid summarization information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "val": {
                                    "type": "boolean",
                                    "description": '''If there is valid summarization information, return True. If the input means there is no relevant information, return False''',
                                    #'enum':language_ar
                                },
        
                            },
                            "required": ['val'],
                        },
                    }
                }
                ]
                tool_choice = {"type": "function", "function": {"name": "summary_val"}}
        
                response = self.llm_client.chat.completions.create(
                    seed=42,
                    temperature=0,
                    model=self.parse_model,
                    messages=messages,
                    tools=tools,
                    #tool_choice=tool_choice
                    # auto is default, but we'll be explicit
                )
                # help(client.chat.completions.create)
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                # Step 2: check if the model wanted to call a function
                if tool_calls:
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    messages.append(response_message)  # extend conversation with assistant's reply
                    # Step 4: send the info for each function call and function response to the model
                    for tool_call in tool_calls:
                        function_args = json.loads(tool_call.function.arguments)
                        val_result=function_args['val']
                        #print('success')
                if val_result==None:
                    error_cnt+=1
                    time.sleep(10)
                    
                if val_result!=None:
                    return val_result

            except:
                error_cnt+=1
                time.sleep(1)
        return False
    def result_parse_summary(self, message_value):
        error_cnt=0
        function_args=None
        while error_cnt<=3 and function_args==None:
            messages = [{"role": "user", "content": message_value}]
            tools = [{
                "type": "function",
                "function": {
                    "name": "view_parse",
                    "description": """The input is summarization of key events in the format of :
                        Summary: <Summarized text> 
                        Extract the summarized text.
                        There summarization should be meaningful and specific information for market events. If there is no relevant information, just ignore it. Do not tag out no relevant information""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            'summary':{
                                'type':'string',
                                'description':'Mearningful summaried text.If theere is no relevant information, just ignore it. Do not include the prefix words,like summary.'
                                }
                        },
                    },
                }
            }]
            tool_choice = {"type": "function", "function": {"name": "view_parse"}}
    
            response = self.llm_client.chat.completions.create(
                seed=42,
                temperature=0,
                model=self.parse_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls:
                messages.append(response_message)
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
            if function_args==None:
                error_cnt+=1
                time.sleep(1)
        return function_args
    def price_trend_fun(self,asset_i,start_date,end_date):
        ds = dsws.Datastream(username=self.ds_username, password=self.ds_password)
        
        asset_code=self.asset_data.query("name==@asset_i")['code'].values[0]
        asset_code=f'<{asset_code}>'
        field=self.asset_data.query("name==@asset_i")['field'].values[0]
        pricing = ds.get_data(tickers=asset_code, fields=[field], start=start_date, end=end_date, freq='D')
        
        pricing.dropna(inplace=True)
        if len(pricing)>0:
            #keep dataframe format consistent
            column_names = list(pricing.columns.names)
            column_names.remove('Instrument')
            pricing.columns = pricing.columns.droplevel(level=column_names)
            
            #key metrics
            ##pricing
            eod = pricing.iloc[-1, :].iloc[0]
            start_p=pricing.iloc[0, :].iloc[0]
            interval_change=eod/start_p-1
            high_index=pricing.iloc[:, 0].idxmax()
            high_val=pricing.loc[high_index].values[0]
            low_index=pricing.iloc[:, 0].idxmin()
            low_val=pricing.loc[low_index].values[0]
            
            trend_content=f"""
            The start closing price in the time period is {str(start_p)}. The end closing prie one is {str(eod)}.
            The highest closing price in the time period is {str(high_val)}. The lowest one is {str(low_val)}. The relative price change in the time period is {interval_change * 100:.2f}%.
            """
        else:
            trend_content='No'
        return pricing, trend_content
    def pricing_table(self, code_df,ds_username, ds_password, end_date_str):
        ds = dsws.Datastream(username=ds_username, password=ds_password)
        grouped = code_df[['code', 'code_type', 'field', 'name', 'type']].groupby(['code_type', 'field'])
        
        year=end_date_str.split('-')[0]
        year_start_date=f'{year}-01-01'
        result = pd.DataFrame()
        name_ls = []
        type_ls = []
        #extract data by different types
        for group_name, group_df in grouped:
            code_type, field = group_name
            codes = group_df['code'].values
            name_ls += list(group_df.name.values)
            type_ls += list(group_df.type.values)
            #prepare code by different type
            ticker_ls = [f"<{i}>" if code_type == 'ric' else i for i in codes]
            tickers_str = ','.join(ticker_ls)
            #get pricing data
            pricing = ds.get_data(tickers=tickers_str, fields=[field], start=year_start_date, end=end_date_str, freq='D')
            #keep dataframe format consistent
            column_names = list(pricing.columns.names)
            column_names.remove('Instrument')
            pricing.columns = pricing.columns.droplevel(level=column_names)
            #merge result
            result=pricing if result.empty else pd.merge(result,pricing,left_index=True,right_index=True)  
           
        #calculate metrics
        eod = result.iloc[-1, :]
        eod.name = 'Closing Price'
        return_1d = eod / result.iloc[-2, :] - 1
        return_1d.name = 'Return(1d)'
        return_1d = return_1d.apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else '-')
        year_first_price=np.array(list([result[asset].dropna().iloc[0] for asset in result.columns.values]))
        return_ytd = eod / year_first_price - 1
        return_ytd.name = 'Return(YTD)'
        return_ytd = return_ytd.apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else '-')
        code_type = pd.Series(type_ls, index=result.columns.values)
        code_type.name = 'Type'
        code_name = pd.Series(name_ls, index=result.columns.values)
        code_name.name = 'Name'
        #combine market data info
        final_info = pd.concat([code_type, code_name, eod, return_1d, return_ytd], axis=1)
        final_info.index.name = 'Code'
        #standardize df output
        final_info.index = final_info.index.map(lambda x: x.replace('<', '').replace('>', ''))
        final_info.Type = final_info.Type.apply(lambda x: x.capitalize())
        final_info.Name = final_info.Name.apply(lambda x: x.upper())
        final_info.fillna('-',inplace=True)
        final_info.sort_values(by='Type',inplace=True)
        return final_info
    
    def news_selection(self,llm_data,asset_code,start_date,end_date,relevance=2):
        #selection process
        selected_news=[]
        #print('asset_code:',asset_code)
        for news_i in llm_data:
            select_content=[]
            select_temp=self.news_filter_process(news_i,relevance,asset_code,start_date,end_date)
            if select_temp:
                select_content.append(select_temp)
            selected_news=selected_news+select_content
        return selected_news
    def news_filter_process(self,news_i,relevance,asset_code,start_date,end_date):
        subjects=news_i[0]['subjects']
        for i in subjects:
            q_code=i.get('_qcode',None)
            if q_code and q_code in asset_code:
                #time filter:
                news_unix=news_i[0]['documentarrivedate_unix']
                start_dt=datetime.datetime.strptime(start_date,'%Y-%m-%d')
                end_dt=datetime.datetime.strptime(end_date,'%Y-%m-%d')
                news_lan=news_i[0]['language']
                #print(news_unix,start_dt.timestamp(),end_dt.timestamp())
                if self.source_language=='all' or self.source_language==news_lan:
                    if news_unix>=start_dt.timestamp() and news_unix<end_dt.timestamp():
                        #relevance filter
                        #print('ok1')
                        if relevance>0:
                            #get the asset relevance in the article
                            related=i.get('related',None)
                            #print('ok2')
                            if related:
                                relevance_i=related[0].get('_qcode',None)
                                if relevance_i:
                                    relevance_str=relevance_i.split(':')[1]
                                    if relevance_str=='high':
                                        relevance_value=2
                                    elif relevance_str=='medium':
                                        relevance_value=1
                                    else:
                                        relevance_value=None
                                    if relevance_value and relevance_value>=relevance:
                                        merged_content=self.target_file_processing(news_i)
                                        return merged_content
                        #no relevance filter
                        else:
                            merged_content=self.target_file_processing(news_i)
                            return merged_content
        return None
    def target_file_processing(self,news_i):

        docid=news_i[0]['docid']
        docid_name=docid.replace(':','-').replace('.','---').replace(',','__')+'.txt'
        
        merged_content=[news_i[1],docid_name]
        return merged_content
#%%
if __name__ == '__main__':
    print('start',datetime.datetime.now())

    tar_currency='US Dollar(USD) TO Euro Dollar(EUR)'
    top_n=15
    mrr_value=0.4
    start_date='2025-01-01'
    end_date='2025-01-17'
    root_path='./data/2025-01-20'
    report_processor=AnalysisProcessor(mrr_value,logger,llm_client,llm,cfg['ds_username'],cfg['ds_password'],'English')
    task_ls=[[tar_currency,'currency',asset_data,root_path+'/currency_LLM_news.json',start_date,end_date,False]]
    report_result=report_processor.report_arch(task_ls)
    print('ends',datetime.datetime.now())
    #x1,x2=report_processor.price_trend_fun('CHINESE YUAN TO US Dollar')
    
    # x1=report_result['currency']['BRL TO USD']['pricing']
    # x1.index
    # x1.index=pd.to_datetime(x1.index)
    # x1.columns=['price']
