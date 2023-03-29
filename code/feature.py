import os
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from constant import *


GEN_FEATURE = True

train_labels = pd.read_csv(TRAIN_LABELS)
test_user_df = pd.read_csv(TEST_USERS)
user_info_df = pd.read_csv(USER_INFO)
record_info_df = pd.read_csv(RECORD_INFO)
user_df = pd.concat([train_labels, test_user_df], axis=0)

JYQD = record_info_df['jyqd'].unique()

def cat2cnt(series):
    map_ = series.groupby(series).agg('count').to_dict()
    return series.map(map_)


def cat2tgmean(df, col):
    map_ = df.groupby(col)[TARGET_COL].agg('mean').to_dict()
    return df[col].map(map_)
    

def agg_user_record_features(one_user_feature_df):
    feature_dict = {
        USER_ID_COL: one_user_feature_df[USER_ID_COL].iloc[0],
        TARGET_COL: one_user_feature_df[TARGET_COL].iloc[0],
        'NumRecord': len(one_user_feature_df)}
    
    num_ops = ['min', 'max', 'std', 'mean', 'median']
    # 1. 给多少人转过账
    for col in ['dfzh', 'dfhh']:
        feature_dict['{}_List'.format(col)] = one_user_feature_df[col].agg(list)
        feature_dict['{}_Nunique'.format(col)] = one_user_feature_df[col].nunique()
        feature_dict['{}_NuniqueDivLen'.format(col)] = one_user_feature_df[col].nunique() / len(one_user_feature_df)
    
    # 2. 借贷标志
    feature_dict['jdbj_InCnt'] = one_user_feature_df['jdbj'].sum()
    feature_dict['jdbj_OutCnt'] = len(one_user_feature_df) - one_user_feature_df['jdbj'].sum()
    feature_dict['jdbj_InRatio'] = one_user_feature_df['jdbj'].sum() / len(one_user_feature_df)
    
    # 3. 数值特征
    for col in ['jyje', 'zhye'] + ['{}_cnt'.format(col) for col in RECORD_CAT_COLS]:
        for op in num_ops:
            feature_dict['{}_{}'.format(col, op.capitalize())] = one_user_feature_df[col].agg(op)
        
        feature_dict['{}_Range'.format(col)] = feature_dict['{}_Max'.format(col)] - feature_dict['{}_Min'.format(col)]
        
    # 4. 组合借贷与其他特征
    # 与数值特征
    for op in num_ops:
        in_op_val = one_user_feature_df[one_user_feature_df['jdbj'] == 1]['jyje'].agg(op)
        out_op_val = one_user_feature_df[one_user_feature_df['jdbj'] == 0]['jyje'].agg(op)
        
        feature_dict['InMoney_{}'.format(op.capitalize())] = in_op_val    
        feature_dict['OutMoney_{}'.format(op.capitalize())] = out_op_val
        feature_dict['InMoney_{}Ratio'.format(op.capitalize())] = in_op_val / (in_op_val + out_op_val + EPS)
        
    # 与类别特征          
    feature_dict['jdbj_InUserCnt'] = one_user_feature_df[one_user_feature_df['jdbj'] == 1]['dfzh'].nunique()
    feature_dict['jdbj_OutUserCnt'] = one_user_feature_df[one_user_feature_df['jdbj'] == 0]['dfzh'].nunique()
    feature_dict['jdbj_InUserRatio'] = feature_dict['jdbj_InUserCnt'] / (feature_dict['jdbj_InUserCnt'] + feature_dict['jdbj_OutUserCnt'] + EPS)
    
    # 5. 时间特征
    feature_dict['whole_life_jy_Interval(h)'] =\
        (one_user_feature_df['jyts'].max() - one_user_feature_df['jyts'].min()).total_seconds() / 3600
    feature_dict['jy_Freq(h)'] =\
        feature_dict['whole_life_jy_Interval(h)'] / len(one_user_feature_df)

    for op in num_ops:
        feature_dict['jyje_{}DivFreq'.format(op.capitalize())] =\
            feature_dict['jyje_{}'.format(op.capitalize())] / feature_dict['jy_Freq(h)']
        feature_dict['oneday_jytimes_{}'.format(op.capitalize())] = \
            one_user_feature_df.groupby("jyrq")['jyts'].agg('count').agg(op)
        # V2添加: 参考别人baseline加上timestamp的均值等
        feature_dict['jytimestampval_{}'.format(op.capitalize())] = \
            (pd.to_numeric(one_user_feature_df['jyts']) / 1e9).agg(op)
    
    for op in num_ops:
        feature_dict['jy_interval_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].diff(1).dt.total_seconds().agg(op)
        feature_dict['jy_day_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].dt.day.agg(op)
        feature_dict['jy_weekday_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].dt.dayofweek.agg(op)
        feature_dict['jy_hour_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].dt.hour.agg(op)
        # V3添加: 参考别人baseline的更精细的时间戳处理
        feature_dict['jy_monthstart_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].dt.is_month_start.agg(op)
        feature_dict['jy_monthend_{}'.format(op.capitalize())] =\
            one_user_feature_df['jyts'].dt.is_month_end.agg(op)
        feature_dict['jy_wkend_{}'.format(op.capitalize())] =\
            (one_user_feature_df['jyts'].dt.dayofweek // 6).agg(op)
                
    for ts in ['day', 'weekday', 'hour']:
        # V3添加: 修正之前的错误
        feature_dict['jy_{}_nunique'.format(ts)] =\
            getattr(one_user_feature_df['jyts'].dt, ts).nunique()
            
    feature_dict['jyts_Min'] = one_user_feature_df['jyts'].min()
    feature_dict['jyts_Max'] = one_user_feature_df['jyts'].max()
    
    # 6. 交易渠道、对方行号、对方账号、摘要代号
    ## 对方账号出现的次数的统计值
    for col in ['dfzh', 'dfhh']:
        for op in num_ops:
            feature_dict['{}_GroupCount_{}'.format(col, op.capitalize())] =\
                one_user_feature_df.groupby(col)[col].agg('count').agg(op)
    
    ## 类别型特征的target enc
    for op in num_ops:
        for col in ['zydh', 'jyqd']:
            feature_dict['{}_tgenc_{}'.format(col, op.capitalize())] =\
                one_user_feature_df['{}_tgenc'.format(col)].agg(op)
    
    ## 交易渠道与摘要代号nunique
    feature_dict['jyqd_nunique'] = one_user_feature_df['jyqd'].nunique()
    feature_dict['zydh_nunique'] = one_user_feature_df['zydh'].nunique()
    
    # 7. 对方名称长度统计值
    for op in num_ops:
        feature_dict['dfmccd_{}'.format(op.capitalize())] = one_user_feature_df['dfmccd'].agg(op)
    
    
    return pd.DataFrame([feature_dict])


def delete_list_cols(df):
    new_df = df.copy()
    is_list = new_df.columns.str.endswith('_List')
    list_cols = new_df.columns[is_list].to_list()
    
    new_df = new_df.drop(list_cols, axis=1)
    return new_df


def process_combined_feature(df):
    df['first_jy_to_kh'] = (df['jyts_Min'] - df['khrq']).dt.total_seconds() / 3600
    df['last_jy_to_kh'] = (df['jyts_Max'] - df['khrq']).dt.total_seconds() / 3600
    df['khrq_year'] = df['khrq'].dt.year
    df['khrq_month'] = df['khrq'].dt.month
    df['khrq_diff_nl'] = df['khrq'].dt.year - df['年龄']
        
    return df


def label_enc(series):
    unique = list(series.unique())
    map_ = dict(zip(unique, range(series.nunique())))
    return series.map(map_)


if __name__ == "__main__":
    
    user_record_df = user_df.merge(record_info_df, on='zhdh', how='left')
    user_record_df.head()
    user_record_df['jyts'] = pd.to_datetime(user_record_df['jyrq'] + ' ' + user_record_df['jysj'])
    
    for col in RECORD_CAT_COLS:
        ss = cat2cnt(user_record_df[col])
        user_record_df['{}_cnt'.format(col)] = ss

    for col in ['zydh', 'dfhh', 'jyqd']:
        tg_ss = cat2tgmean(user_record_df, col)
        user_record_df['{}_tgenc'.format(col)] = tg_ss

    
    if not GEN_FEATURE and os.path.exists(SAVE_DIR / "user_record_and_static_info_df.csv"):
        user_record_and_static_info_df = pd.read_csv(SAVE_DIR / "user_record_and_static_info_df.csv")
    else:
        user_agg_reocrd_df = user_record_df.groupby(USER_ID_COL).apply(agg_user_record_features)
        user_agg_reocrd_df.reset_index(drop=True, inplace=True)
        user_agg_reocrd_df = delete_list_cols(user_agg_reocrd_df)
        user_record_and_static_info_df = user_agg_reocrd_df.merge(user_info_df, on='zhdh', how='left')
        user_record_and_static_info_df['khrq'] = pd.to_datetime(user_record_and_static_info_df['khrq'])
        user_record_and_static_info_df = process_combined_feature(user_record_and_static_info_df)
        
        user_record_and_static_info_df.to_csv(SAVE_DIR / "user_record_and_static_info_df.csv", index=False)