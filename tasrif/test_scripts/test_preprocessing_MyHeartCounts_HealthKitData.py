# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os

from tasrif.data_readers.my_heart_counts import HealthKitDataDataset

hkd_file_path = os.environ['MYHEARTCOUNTS_HEALTHKITDATA_PATH']
hkd = HealthKitDataDataset(hkd_file_path)
record, df = next(hkd.processed_df[0])

# %%
hkd.raw_df

# %%
record

# %%
df
