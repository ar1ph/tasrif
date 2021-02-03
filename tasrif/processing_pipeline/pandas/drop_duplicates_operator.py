
"""
Remove duplicate values from one or more dataframes.
"""
from tasrif.processing_pipeline import ProcessingOperator


class DropDuplicatesOperator(ProcessingOperator):
    """
    Remove duplicate rows from one or more dataframes.

    Parameters
    ----------

    Raises
    ------

    Examples
    --------

    >>>import pandas as pd
    >>>import numpy as np
    >>>
    >>>from tasrif.processing_pipeline.pandas import DropDuplicatesOperator
    >>>
    >>>df0 = pd.DataFrame([['tom', 10], ['Alfred', 15], ['Alfred', 18], ['juli', 14]], columns=['name', 'age'])
    >>>df1 = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman', 'Catwoman'],
    >>>                   "toy": [np.nan, 'Batmobile', 'Bullwhip', 'Bullwhip'],
    >>>                   "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT,
    >>>                            pd.NaT]})
    >>>
    >>>operator = DropDuplicatesOperator(subset='name')
    >>>df0, df1 = operator.process(df0, df1)
    >>>
    >>>print(df0)
    >>>print(df1)

    """

    def __init__(self, **kwargs):
        """
        Initializes the operator

        **kwargs:
          key word arguments passed to pandas DataFrame.drop_duplicates method
        """
        self.kwargs = kwargs
        super().__init__()

    def process(self, *data_frames):
        """Process the passed data using the processing configuration specified
        in the constructor

        data_frames:
          Variable number of pandas dataframes to be processed

        Returns
        -------
        data_frames
            Processed data frames
        """

        processed = []
        for dataframe in data_frames:
            dataframe = dataframe.drop_duplicates(**self.kwargs)
            processed.append(dataframe)

        return tuple(processed)
