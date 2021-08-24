# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""Example on how to read all data from SIHA
"""
import os

from tasrif.processing_pipeline import (
    ProcessingPipeline,
    ProcessingOperator,
    ComposeOperator,
    NoopOperator,
)

from tasrif.processing_pipeline.map_processing_operator import MapProcessingOperator
from tasrif.data_readers.siha_dataset import SihaDataset
from tasrif.processing_pipeline.custom import JqOperator, CreateFeatureOperator
from tasrif.processing_pipeline.pandas import (
    JsonNormalizeOperator,
    SetIndexOperator,
    ConvertToDatetimeOperator,
    AsTypeOperator,
    DropFeaturesOperator
)

siha_folder_path = os.environ['SIHA_PATH']


class FlattenOperator(MapProcessingOperator):
     def processing_function(self, arr):
        return arr[0]
    
# Rename column names
class RenameOperator(ProcessingOperator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process(self, *data_frames):
        processed = []
        for data_frame in data_frames:
            data_frame = data_frame.rename(**self.kwargs)
            processed.append(data_frame)

        return processed



# %%
# base_datasets = ["EMR", "CGM", "Steps", "Distance", "Calories"]
base_datasets = ProcessingPipeline([
    SihaDataset(folder=siha_folder_path, table_name="EMR"),
    ComposeOperator([
        JqOperator("map({patientID} + .data.emr[])"), # EMR
        JqOperator("map({patientID} + .data.cgm[])"), # CGM
        JqOperator(
            'map({patientID} + .data.activities_tracker_steps[].data."activities-tracker-steps"[0])'
        ), # Steps
        JqOperator(
            'map({patientID} + .data.activities_tracker_distance[].data."activities-tracker-distance"[0])'
        ), # Distance
        JqOperator(
            'map({patientID} + .data.activities_tracker_calories[].data."activities-tracker-calories"[0])'
        ), # Calories
    ]),
    FlattenOperator(),
    JsonNormalizeOperator(),
    RenameOperator(columns={'time': 'dateTime'}, errors='ignore'),
    ConvertToDatetimeOperator(feature_names=["dateTime"], infer_datetime_format=True),
    SetIndexOperator("dateTime"),
    AsTypeOperator({"value": "float32"}, errors='ignore')
])



df = base_datasets.process()
df

# %%
# intraday_datasets = ["HeartRateIntraday", "CaloriesIntraday", "StepsIntraday", "DistanceIntraday"]
intraday_datasets = ProcessingPipeline([
    SihaDataset(folder=siha_folder_path, table_name="HeartRateIntraday"),
    ComposeOperator([
        JqOperator(
            'map({patientID} + .data.activities_heart_intraday[].data as $item  | '
            +
            '$item."activities-heart-intraday".dataset | '
            +
            'map({date: $item."activities-heart"[0].dateTime} + .) | .[])'
        ), # HeartRateIntraday
        JqOperator(
            "map({patientID} + .data.activities_calories_intraday[].data as $item  |"
            +
            ' $item."activities-calories-intraday".dataset | '
            +
            'map({date: $item."activities-calories"[0].dateTime} + .) | .[])'
        ), # CaloriesIntraday
        JqOperator(
            'map({patientID} + .data.activities_steps_intraday[].data as $item  | '
            +
            '$item."activities-steps-intraday".dataset | '
            +
            'map({date: $item."activities-steps"[0].dateTime} + .) | .[])'
        ), # StepsIntraday
        JqOperator(
            "map({patientID} + .data.activities_distance_intraday[].data as $item  |"
            +
            ' $item."activities-distance-intraday".dataset | '
            +
            'map({date: $item."activities-distance"[0].dateTime} + .) | .[])'
        ), # DistanceIntraday
    ]),
    FlattenOperator(),
    JsonNormalizeOperator(),
    CreateFeatureOperator(
        feature_name="dateTime",
        feature_creator=lambda df: df["date"] + "T" + df["time"],
    ),
    DropFeaturesOperator(["date", "time"]),
    ConvertToDatetimeOperator(feature_names=["dateTime"], infer_datetime_format=True),
    SetIndexOperator("dateTime"),
    AsTypeOperator({"value": "float32"}, errors='ignore')
])

df_intra = intraday_datasets.process()
df_intra
