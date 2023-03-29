import os
from pathlib import Path


DIR = Path(os.path.dirname(os.path.dirname(__file__)))
print(DIR)
SAVE_DIR = DIR / 'process_data/'
INFER_DIR = DIR / 'inference_data/'
RESULT_DIR = DIR / 'prediction_result/'

TRAIN_LABELS = DIR / 'raw_data/训练集标签.csv'
TEST_USERS = DIR / 'raw_data/test_dataset.csv'
USER_INFO = DIR / 'raw_data/账户静态信息.csv'
RECORD_INFO = DIR / 'raw_data/账户交易信息.csv'

USER_ID_COL = 'zhdh'
TARGET_COL = 'black_flag'
RECORD_CAT_COLS = ['jyqd', 'zydh', 'dfhh', 'dfzh']
UNAVAILABLE_COLS = [TARGET_COL, USER_ID_COL, 'khrq', 'khjgdh', 'jyts_Min', 'jyts_Max']

EPS = 1e-5
SEED = 2023
N_SPLITS = 10

LGB_PARAMS = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'max_depth':7,
    'num_leaves':63,
    'min_child_samples':20,
    'n_estimators': 150,
}

XGB_PARAMS = {
    'n_estimators': 120,
    'max_depth': 7,
    'max_leaves': 63,
    'learning_rate': 0.05,
    'booster':'gbtree',
    'subsample': 0.7,
    'colsample_bytree':0.7,
    'reg_alpha':0.1,
    'reg_lambda':0.01,
}

CGB_PARAMS = {
    'verbose':False,
    'use_best_model':True,
    'learning_rate':0.08,
    'rsm': 0.7,
    'depth': 5,
    'border_count': 32,
    'feature_border_type': 'Median',
}

HGBT_PARAMS = {
    'max_leaf_nodes': 15,
    'min_samples_leaf': 20,
    'max_iter':1000,
    'learning_rate':0.08,
    'early_stopping':True,
    'max_depth':7,
}

GBDT_PARAMS = {
    'n_estimators': 120,
    'min_samples_leaf': 10,
    'max_depth': 7,
    
}