# +
import pandas as pd
from tasrif.processing_pipeline.custom import SlidingWindowOperator

df = pd.DataFrame([
    ["2020-02-16 11:45:00",27,102.5],
    ["2020-02-16 12:00:00",27,68.5],
    ["2020-02-16 12:15:00",27,40.0],
    ["2020-02-16 15:15:00",27,282.5],
    ["2020-02-16 15:30:00",27,275.0],
    ["2020-02-16 15:45:00",27,250.0],
    ["2020-02-16 16:00:00",27,235.0],
    ["2020-02-16 16:15:00",27,206.5],
    ["2020-02-16 16:30:00",27,191.0],
    ["2020-02-16 16:45:00",27,166.5],
    ["2020-02-16 17:00:00",27,171.5],
    ["2020-02-16 17:15:00",27,152.0],
    ["2020-02-16 17:30:00",27,124.0],
    ["2020-02-16 17:45:00",27,106.0],
    ["2020-02-16 18:00:00",27,96.5],
    ["2020-02-16 18:15:00",27,86.5],
    ["2020-02-16 18:30:00",27,78.0],
    ["2020-02-16 18:45:00",27,71.5],
    ["2020-02-16 19:00:00",27,64.5],
    ["2020-02-16 19:15:00",27,51.0],
    ["2020-02-16 19:30:00",27,50.666668],
    ["2020-02-16 19:45:00",27,41.0],
    ["2020-02-16 20:00:00",27,40.0],
    ["2020-02-16 20:15:00",27,40.0],
    ["2020-02-16 20:30:00",27,40.0],
    ["2020-02-16 14:45:00",31,125.0],
    ["2020-02-16 15:00:00",31,140.5],
    ["2020-02-16 15:15:00",31,183.0],
    ["2020-02-16 15:30:00",31,222.0],
    ["2020-02-16 15:45:00",31,234.5],
    ["2020-02-16 16:00:00",31,249.0],
    ["2020-02-16 16:15:00",31,245.5],
    ["2020-02-16 16:30:00",31,236.0],
    ["2020-02-16 16:45:00",31,223.0],
    ["2020-02-16 17:00:00",31,208.0],
    ["2020-02-16 17:15:00",31,194.0],
    ["2020-02-16 17:30:00",31,186.0],
    ["2020-02-16 17:45:00",31,177.0],
    ["2020-02-16 18:00:00",31,171.0],
    ["2020-02-16 18:15:00",31,164.0],
    ["2020-02-16 18:30:00",31,156.0],
    ["2020-02-16 18:45:00",31,157.0],
    ["2020-02-16 19:00:00",31,158.0],
    ["2020-02-16 19:15:00",31,158.5],
    ["2020-02-16 19:30:00",31,150.0],
    ["2020-02-16 19:45:00",31,145.0],
    ["2020-02-16 20:00:00",31,137.0],
    ["2020-02-16 20:15:00",31,141.0],
    ["2020-02-16 20:45:00",31,146.0],
    ["2020-02-16 21:00:00",31,141.0]],
    columns=['dateTime','patientID','CGM'])

df['dateTime'] = pd.to_datetime(df['dateTime'])
df
# +
op = SlidingWindowOperator(winsize="1h15t",
                           time_col="dateTime",
                           label_col="CGM",
                           participant_identifier="patientID")



df_timeseries, df_labels, df_label_time, df_pids = op.process(df)[0]
df_timeseries
