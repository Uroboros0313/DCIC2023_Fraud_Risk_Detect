import time

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

from constant import *
from data_utils import *



def train_oof_stack_hard_models(X, y, final_model='lr', stack_method='auto'):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        print(f"Fold {i + 1}, start time {time.strftime(r'%Y-%m-%d %H:%M:%S')}, training stacking model!")
        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]
        estimators = []
        for j in range(6):
            for mod in ['lgb', 'xgb', 'hgbt', 'cgb', 'gbdt']:
                if mod == 'lgb':
                    params = LGB_PARAMS
                    base_mod_func = LGBMClassifier
                elif mod == 'xgb':
                    params = XGB_PARAMS
                    base_mod_func = XGBClassifier
                elif mod == 'hgbt':
                    params = HGBT_PARAMS
                    base_mod_func = HistGradientBoostingClassifier
                elif mod == 'cgb':
                    params = CGB_PARAMS
                    params.update({'use_best_model': False})
                    base_mod_func = CatBoostClassifier
                elif mod == 'gbdt':
                    params = GBDT_PARAMS
                    base_mod_func = GradientBoostingClassifier
                
                params.update({"random_state": SEED * i + j})
                base_model = base_mod_func(**params)
                estimators.append((f"{mod}_{j}", base_model))
        
        if final_model == 'lr':
            final_estimator = LogisticRegression()
        elif final_model == 'mlp':
            final_estimator = MLPClassifier()
        
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, stack_method=stack_method)
        model.fit(trn_X, trn_y)
        models.append(model)

        preds = model.predict_proba(val_X)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def train_oof_gbdt_models(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        gbdt_params = GBDT_PARAMS
        gbdt_params.update({"random_state": SEED + i})
        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]

        model = GradientBoostingClassifier(**gbdt_params)
        model.fit(trn_X, trn_y)
        models.append(model)

        preds = model.predict_proba(val_X)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def train_oof_cgb_models(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        cgb_params = CGB_PARAMS
        cgb_params.update({'random_state': SEED + i})

        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]

        model = CatBoostClassifier(**cgb_params)
        model.fit(trn_X, trn_y, eval_set=[(val_X, val_y)])
        models.append(model)

        preds = model.predict_proba(val_X)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def train_oof_hgbt_models(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        hgbt_params = HGBT_PARAMS
        hgbt_params.update({'random_state': SEED + i})

        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]

        model = HistGradientBoostingClassifier(**hgbt_params)
        model.fit(trn_X, trn_y)
        models.append(model)

        preds = model.predict_proba(val_X)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def train_oof_xgb_models(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        xgb_params = XGB_PARAMS
        xgb_params.update({'random_state': SEED + i})

        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]

        model = XGBClassifier(**xgb_params)
        model.fit(trn_X, trn_y, eval_set=[(val_X, val_y)])
        models.append(model)

        preds = model.predict_proba(val_X)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def train_oof_lgb_models(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    oof = y.copy()
    models = []

    for i, (trn_idxs, val_idxs) in enumerate(skf.split(X, y)):
        lgb_params = LGB_PARAMS
        lgb_params.update({'random_state': SEED + i})

        trn_X, val_X, trn_y, val_y = X.iloc[trn_idxs], X.iloc[val_idxs], y.iloc[trn_idxs], y.iloc[val_idxs]

        model = LGBMClassifier(**lgb_params)
        model.fit(trn_X, trn_y, eval_set=[(val_X, val_y)])
        models.append(model)

        preds = model.predict_proba(
            val_X, num_iteration=model.best_iteration_)[:, 1]
        oof.iloc[val_idxs] = preds

    return models, oof


def select_best_threshold(oof, y_true):
    result = pd.DataFrame()

    for threshold in np.arange(0.1, 0.8, 0.02):
        preds = (oof > threshold).astype(int)
        f1 = f1_score(y_true, preds)
        pr = precision_score(y_true, preds)
        rc = recall_score(y_true, preds)
        result = result.append(pd.DataFrame({
            'threshold': [threshold],
            'f1': [f1],
            'recall': [rc],
            'precision': [pr]
        }))

    best_threshold = result.sort_values('f1')['threshold'].iloc[-1]

    return best_threshold, result


'''
def predict_and_save_test(models, test_df_id, X, best_threshold, name):
    test_ps = [model.predict_proba(X.values.astype(float))[
        :, 1] for model in models]
    test_preds = (np.mean(test_ps, axis=0) > best_threshold).astype(int)

    test_result = pd.DataFrame({
        USER_ID_COL: test_df_id,
        TARGET_COL: test_preds
    })

    test_result.to_csv(SAVE_DIR / "submit_{}_{}_threshold{:.4f}.csv".
                       format(name, time.strftime("%Y%m%d-%H%M"), best_threshold), index=False)
    return test_result
'''


def predict_one_model_by_threshold(model, X, best_threshold):
    test_p = model.predict_proba(X)[:, 1]
    test_preds = (test_p > best_threshold).astype(int)
    return test_preds