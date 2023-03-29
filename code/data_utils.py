import os

import pandas as pd

from constant import *


def read_all_features():
    if os.path.exists(SAVE_DIR / "user_record_and_static_info_df.csv"):
        all_info_df = pd.read_csv(SAVE_DIR / "user_record_and_static_info_df.csv")
    return all_info_df