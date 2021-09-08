import os
from tasrif.processing_pipeline import SequenceOperator
from tasrif.data_readers.withings_dataset import WithingsDataset
from tasrif.processing_pipeline.pandas import ConvertToDatetimeOperator, SetIndexOperator, AsTypeOperator

# +
withings_data_filename = os.environ['WITHINGS_PATH']+'weight.csv'
# -

pipeline = SequenceOperator([
    WithingsDataset(withings_data_filename, table_name="Weight"),
    ConvertToDatetimeOperator(feature_names=["Date"], infer_datetime_format=True),
    SetIndexOperator("Date"),
    AsTypeOperator({"Weight (kg)": "float32"})
])

df = pipeline.process()

print(df)