import pandas as pd 
import numpy as np
from utills.utills_append_1_features import main_utills_1
from utills.utills_append_meta import main_utills_2

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

def main_utills(path):
    '''
    Функция которая стартовый json преобразовывает в равернутый dataframe
    '''
    df = pd.read_json(path, orient='index')
    
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
    '''
    добавление таргета в датасет
    '''
    y=df.loc[:, ['target']]
    dict_change={'male':1, 'female':0}
    y['target'] = y['target'].replace(dict_change)
    df_finish=pd.concat([y, df_finish], axis=1)
    '''
    добавление двух фичей (вероятность пола юзера по приобретенным товарам)
    ''' 
    
    df_1=main_utills_1(new_df)
    df_1.reset_index(drop=True, inplace=True)
    df_finish.reset_index(drop=True, inplace=True)
    df_finish=pd.concat([df_finish, df_1], axis=1)
    

    
    return df_finish