import pandas as pd 
import json


with open('utills\info_features_category_item_probability.json', 'r') as fcc_file:
    info_1 = json.load(fcc_file)
    

        

def append_male_features(dict_row):
    '''
    Функция подсчета вероятности  мужского пола по покупкам
    return: вероятность
    '''
    global info_1
    sum_male=0
    if len(dict_row)==0:
        return 0
    else:
        for i in dict_row:
            sum_male+=info_1[str(i)][0]*dict_row[i]
        return sum_male/len(dict_row)

def append_female_features(dict_row):
    '''
    Функция подсчета вероятности  женскго пола по покупкам
    return: вероятность
    '''
    global info_1
    sum_female=0
    if len(dict_row)==0:
        return 0
    else:
        for i in dict_row:
            sum_female+=info_1[str(i)][1]*dict_row[i]   
        return sum_female/len(dict_row)

def create_dict_cat_num(list_row):
    '''
    Функция подскчитывает количество категорий товаров по всем покупкам пользователя
    return: dict категория колтчество
    '''
    a={}
    
    if isinstance(list_row, list):
        for line in list_row:
            for line_1 in line['orders']:
                if len(line_1['items'])>0:
                    for line_2 in line_1['items']:
                        if 'general-category-path' in line_2.keys():
                            a={line_3:0 for line_3 in line_2['general-category-path']}
                            for line_3 in line_2['general-category-path']:
                                a[line_3]+=1
        return a
    else:
        return {}
    

def main_utills_1(new_df):
    '''
    Функция которая стартовый json преобразовывает в равернутый dataframe
    '''

    #new_df = df['features'].apply(pd.Series) если начинаем со стартового df

    new_df['orders']=new_df['orders'].apply(create_dict_cat_num)

    new_df=new_df.loc[:, ['orders']]

    new_df['male_features']=new_df['orders'].apply(append_male_features)
    new_df['female_features']=new_df['orders'].apply(append_female_features)

    return new_df[['male_features', 'female_features']]