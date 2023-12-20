import pandas as pd 
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, make_scorer
from utills.utills_append_1_features import main_utills_1
from utills.utills_append_meta import main_utills_2

with open('utills\columns_data.json', 'r') as fcc_file:
    columns_data = json.load(fcc_file)

with open('model/model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

def dict_to_columns(df, column):
    '''
    Функция преобразования словаря в набор столбиков
    '''
    return df[column].apply(pd.Series)

def categories_to_dict(list_col):
    '''
    Функция преобразования списка в словарь для колонки 'last-visits-in-categories'    
    '''
    if isinstance(list_col, list):
        finish_dict={}
        for row in list_col:
            finish_dict[row['category']]=1
        return finish_dict      
    else:
        return None

def visit_to_dict(list_col):
    '''
    Функция преобразования списка в словарь для колонки 'visits'
    '''
    if isinstance(list_col, list):
        finish_dict={}
        for row in list_col:
            finish_dict[f'visit_{row["site-id"]}']=1
        return finish_dict
    else:
        return None

def order_to_dict(list_col):
    '''
    Функция преобразования списка в словарь для колонки 'orders'
    '''
    if isinstance(list_col, list):
        finish_dict={}
        for row in list_col:
            finish_dict[f'order_{row["site-id"]}']=1
        return finish_dict
    else:
        return None

def meta_to_dict(list_col):
    '''
    Функция преобразования списка в словарь для колонки 'site-meta'    
    '''
    if isinstance(list_col, list):
        finish_dict={}
        for row in list_col:
            if len(row)>1:
                finish_dict[f'meta{row["site-id"]}']=1
            else:
                finish_dict[f'meta{row["site-id"]}']=-1
        return finish_dict      
    else:
        return {}

def create_data(path):
    '''
    Функция которая стартовый json преобразовывает в равернутый dataframe
    '''
    logging.info('Start read json file: ' + str(datetime.now()))
    df = pd.read_json(path, orient='index')
    logging.info('Start create columns: ' + str(datetime.now()))
    new_df = dict_to_columns(df, 'features')

    new_df['orders']=new_df['orders'].apply(order_to_dict)
    new_df['visits']=new_df['visits'].apply(visit_to_dict)
    new_df['last-visits-in-categories']=new_df['last-visits-in-categories'].apply(categories_to_dict)
    new_df['site-meta']=new_df['site-meta'].apply(meta_to_dict)


    new_df_1=dict_to_columns(new_df, 'orders')
    new_df_2=dict_to_columns(new_df, 'visits')
    new_df_3=dict_to_columns(new_df, 'last-visits-in-categories')
    new_df_4=dict_to_columns(new_df, 'site-meta')


    df_finish=pd.concat([new_df_1, new_df_2, new_df_3, new_df_4], axis=1)
    logging.info('End create columns: ' + str(datetime.now()))
    '''
    добавление таргета в датасет
    '''
    if 'target' in list(df.columns):
        y=df.loc[:, ['target']]
        dict_change={'male':1, 'female':0}
        y['target'] = y['target'].replace(dict_change)
        df_finish=pd.concat([y, df_finish], axis=1)
        logging.info('Add target: ' + str(datetime.now()))
    else:
        logging.info('No target: ' + str(datetime.now()))
    '''
    добавление двух фичей (вероятность пола юзера по приобретенным товарам)
    ''' 
    df_1=main_utills_1(new_df)
    df_1.reset_index(drop=True, inplace=True)
    df_finish.reset_index(drop=True, inplace=True)
    df_finish=pd.concat([df_finish, df_1], axis=1)
    logging.info('Append features male/female probability: ' + str(datetime.now()))
    '''
    изменение порядка столбцов под модель
    '''     
    global columns_data 
    col=list(set(columns_data['with_target'])-set(df_finish.columns))
    ind=df_finish.index
    df_finish=pd.concat([df_finish, pd.DataFrame(columns=col, index=ind, dtype=float)], axis=1)
    if 'target' in list(df.columns): 
        df_finish=df_finish[['target']+columns_data['with_target']]
    else:
        df_finish=df_finish[columns_data['with_target']]     
    logging.info('End create dataframe: ' + str(datetime.now()))
    return df_finish

def create_prediction(df):
    '''
    функция для predict, на вход уже подготовленный датасет. prediction.csv файл в корне main.py
    ''' 
    logging.info('Start create predictions: ' + str(datetime.now()))
    global model
    if 'target' in list(df.columns):
        predictions=model.predict(df.iloc[:, 1:])
        prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv', index=False)
        logging.info('accuracy_score:'+ str(accuracy_score(df.iloc[:, 0], predictions)) + str(datetime.now()))
        logging.info('End create predictions: ' + str(datetime.now()))
        
    else:
        predictions=model.predict(df)
        prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv', index=False)
        logging.info('End create predictions without target: ' + str(datetime.now()))

        
        
    
    