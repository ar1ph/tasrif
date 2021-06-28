"""
Concatenate multiple dataframes into a single one.
"""
import pandas as pd
from tasrif.processing_pipeline import ProcessingOperator


class ConcatOperator(ProcessingOperator):
    """Concatenate different datasets based on Pandas concat method.

    Parameters
    ----------

    Returns
    -------

    Examples:
    ---------
    >>> import pandas as pd
    >>>
    >>> from tasrif.processing_pipeline.pandas import ConcatOperator
    >>>
    >>> # Full
    >>> df1 = pd.DataFrame({'id': [1, 2, 3], 'cities': ['Rome', 'Barcelona', 'Stockholm']})
    >>> df2 = pd.DataFrame({'id': [4, 5, 6], 'cities': ['Doha', 'Vienna', 'Belo Horizonte']})
    >>>
    >>> concat = ConcatOperator().process(df1, df2)
    >>> concat

        id  cities
    0   1   Rome
    1   2   Barcelona
    2   3   Stockholm
    0   4   Doha
    1   5   Vienna
    2   6   Belo Horizonte

    """

    def __init__(self, **kwargs):
        """Merge different datasets on a common feature defined by ``on``.

        Parameters
        ----------
        data_frames:
          Variable number of pandas dataframes to be processed

        **kwargs:
          key word arguments passed to pandas concat method


        """
        self.kwargs = kwargs
        super().__init__()

    def process(self, *data_frames):
        """Concatenate multiple datasets.

        Returns
        -------
        data_frame
            Concatenated dataframe based on the input data_frames.
        """
        data_frame = pd.concat(list(data_frames), **self.kwargs)
        return [data_frame]
