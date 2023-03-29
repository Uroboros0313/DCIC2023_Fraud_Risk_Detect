from model import *
from constant import *
from data_utils import *

MODEL_OPT_DICT = {
    'hgbt': (1, train_oof_hgbt_models),
    'lgb': (1, train_oof_lgb_models),
    'cgb': (1, train_oof_cgb_models),
    'xgb': (1, train_oof_xgb_models),
    'gbdt': (1, train_oof_gbdt_models),
    'stack_hard': (1, train_oof_stack_hard_models) 
}

def train_tree_model(model_name, X, y):
    
    model_opt = MODEL_OPT_DICT.get(model_name, None)[1]
    if model_opt is None:
        raise ValueError("Model does not exist!")
    models, oof = model_opt(X, y)
    return models, oof

if __name__ == "__main__":
    all_info_df = read_all_features()
    train_df = all_info_df[all_info_df[TARGET_COL].notnull()]
    test_df = all_info_df[all_info_df[TARGET_COL].isnull()]

    available_feats = [col for col in train_df.columns if col not in UNAVAILABLE_COLS]
    y = train_df[TARGET_COL].round()
    X = train_df[available_feats].fillna(-10)
    test_df_id = test_df[USER_ID_COL].values
    test_X = test_df[available_feats].fillna(-10)

    test_preds = []
    model_weight_sum = 0
    for model_name in MODEL_OPT_DICT:
        models, oof = train_tree_model(model_name, X, y)
        best_threshold, oof_result = select_best_threshold(oof, y)
        model_weight = MODEL_OPT_DICT.get(model_name, [0, None])[0]
        model_weight_sum += model_weight * len(models)
        test_preds.extend([model_weight * predict_one_model_by_threshold(model, test_X, best_threshold) 
                           for model in models])
        
    test_res = ((np.sum(test_preds, axis=0) / model_weight_sum) > 0.5).astype(int)
    test_result = pd.DataFrame({
            USER_ID_COL: test_df_id,
            TARGET_COL: test_res})
    test_result.to_csv(RESULT_DIR / "result.csv", index=False)