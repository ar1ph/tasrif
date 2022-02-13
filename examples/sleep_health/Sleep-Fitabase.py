# +
import os
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split 
from datetime import timedelta, datetime
from tasrif.processing_pipeline import SequenceOperator
from tasrif.processing_pipeline.pandas import (
    RenameOperator, 
    ConvertToDatetimeOperator, 
    CutOperator, 
    ReadCsvOperator
)
from tasrif.processing_pipeline.custom import (
    AggregateActivityDatesOperator, 
    CreateFeatureOperator,
    FilterOperator,
    MergeFragmentedActivityOperator
)

from tasrif.processing_pipeline.tsfresh import TSFreshFeatureExtractorOperator


df = pd.read_csv("./minuteSleep_merged_2.csv")
df['date'] = pd.to_datetime(df['date'])


# +
def calculate_total_duration(df):
    return (df['Asleep'] + df['Restless'] + df['Awake'])

def calculate_sleep_quality(df):
    total_duration = df['Asleep'] + df['Restless'] + df['Awake']
    if total_duration == 0:
        return 0
    return df['Asleep'] / total_duration

pipeline = SequenceOperator([AggregateActivityDatesOperator(date_feature_name="date",
                                          participant_identifier=['Id', 'logId'],
                                          aggregation_definition={'value': [lambda x: (x[x == 1] == 1).sum(), 
                                                                            lambda x: (x[x == 2] == 2).sum(),
                                                                            lambda x: (x[x == 3] == 3).sum(),
                                                                           ]}),
                                RenameOperator(columns={"value_0": "Asleep",
                                                        "value_1": "Restless",
                                                        "value_2": "Awake"}),
                                CreateFeatureOperator(
                                    feature_name='Bed',
                                    feature_creator=calculate_total_duration 
                                ),                                
                                CreateFeatureOperator(
                                    feature_name='Sleep Efficiency',
                                    feature_creator=calculate_sleep_quality 
                                ),
                                ConvertToDatetimeOperator(
                                    feature_names=['start', 'end']
                                )
                            ])
# -

aggregate = pipeline.process(df)[0]

aggregate

# +
aggregation_definition = {
    'logId': lambda df: df.iloc[0],
    'Asleep': np.sum,
    'Restless': np.sum,
    'Awake': np.sum,
    'Bed': np.sum,
}

operator = MergeFragmentedActivityOperator(
            participant_identifier='Id',
            start_date_feature_name='start',
            end_date_feature_name='end',
            threshold="3 hour",
            aggregation_definition=aggregation_definition)

aggregate = operator.process(aggregate)[0]
# -

aggregate

# +
fig = plt.figure(figsize =(20,8))
plt.scatter(aggregate['Asleep']/60, aggregate['Sleep Efficiency'])

plt.axhline(y=0.95, color='g', linestyle='-')
plt.axhline(y=0.90, color='y', linestyle='-')
plt.axhline(y=0.85, color='r', linestyle='-')

plt.xlabel('Sleep Duration (Hrs)')
plt.ylabel('Sleep Efficiency')
plt.title('Duration V/S Efficiency')
plt.show()

# +
# Add sleep codes
threshold = 10800

pipeline = SequenceOperator([
    CreateFeatureOperator(
        feature_name='main sleep',
        feature_creator=lambda df: (df['end'] - df['start']).total_seconds() > threshold
    ),
    CutOperator( # Categorize continious variabale
        cut_column_name='Sleep Efficiency', 
        bin_column_name='Sleep Category', 
        bins=[0, 0.85, 0.9, 1.01],
        labels=[2, 1, 0],
        right=False
    )
])

aggregate = pipeline.process(aggregate)[0]
# -

aggregate

# +
# Prepare steps and sleep for X, and y
aggregate_ids = aggregate.Id.unique()

pipeline = SequenceOperator([
    ReadCsvOperator('./fifteenMinuteSteps_merged.csv'),
    ConvertToDatetimeOperator(['ActivityMinute']),
])

steps = pipeline.process()[0]
steps['time'] = steps['ActivityMinute'].dt.strftime("%H:%M")

# Filter aggregate set
aggregate = aggregate[aggregate.Id.isin(steps.Id.unique())]

# Need logId in steps, so that X, y samples are equal in number
steps = steps.merge(aggregate, how='inner', on='Id')
steps['between_time'] = (steps['ActivityMinute'] <= steps['end']) & ((steps['ActivityMinute'] >= steps['start']))
steps = steps[steps['between_time'] == True] # This let's us know which logId belongs to the current ActivityMinute
steps = steps[steps['main sleep'] == True]
steps['tsfresh_id'] = steps['Id'].astype(str) + '_' + steps['logId'].astype(str)
# -

# Form y
y = aggregate[aggregate['main sleep'] == True].copy()
y["prediction_yesterday"] = y.groupby("Id")["Sleep Efficiency"].shift(1).fillna(y["Sleep Efficiency"].median())
y = y[['Id', 'logId', 'Bed', 'Sleep Efficiency', 'Sleep Category', 'prediction_yesterday']]
y['tsfresh_id'] = y['Id'].astype(str) + '_' + y['logId'].astype(str)
y = y.set_index('tsfresh_id')
y

# +
# Form X

operator = TSFreshFeatureExtractorOperator(
    seq_id_col="tsfresh_id", 
    date_feature_name='time',
    value_col='Steps',
    labels=y['Sleep Efficiency'],
)

X = operator.process(steps)[0]
