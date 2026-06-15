import dask.dataframe as dd
from src.filters import DataFilter


def sync_with_wrapper(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.df_wrapper.update_dataframe(self.df)
        return result

    return wrapper


class DataFrameWrapper:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = dd.read_csv(csv_path, assume_missing=True)
        if "header.counter" in self.df.columns:
            self.df = self.df.set_index("header.counter", sorted=True, drop=False)

    def get_dataframe(self):
        return self.df

    def get_csv_path(self):
        return self.csv_path

    def update_dataframe(self, dataframe):
        self.df = dataframe


class DataProcessor:
    def __init__(self, df_wrapper, log_fn=None):
        """Initializes processor with a DataFrameWrapper instance."""
        self.df_wrapper = df_wrapper
        self.df = df_wrapper.get_dataframe()
        self.filter_manager = DataFilter(log_fn=log_fn)  # Handles filter strategies

    def add_filter(self, columns, filter_name, **kwargs):
        """Adds a predefined filter with parameters to be applied later."""
        self.filter_manager.add_filter(columns, filter_name, **kwargs)

    @sync_with_wrapper
    def queue_filters(self):
        """Applies all filters to the DataFrame."""
        self.df = self.filter_manager.queue_filters(self.df)

    @sync_with_wrapper
    def normalize_columns(self, columns):
        """Applies min-max normalization to multiple columns."""
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            min_val = self.df[column].min().compute()
            max_val = self.df[column].max().compute()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)

    @sync_with_wrapper
    def scale_columns(self, columns, factor):
        """
        Multiplies multiple columns by a given factor.
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            self.df[column] = self.df[column] * factor

    @sync_with_wrapper
    def sort_data(self, key, ascending=True):
        """Sorts the dataframe by a specific column."""
        self.df = self.df.sort_values(by=key, ascending=ascending)

    @sync_with_wrapper
    def drop_data(self, columns=None, row_range=None, row_condition=None):
        """
        Drops specified columns and/or rows based on range or condition.

        :param columns: List of column names to drop.
        :param row_range: Tuple (start, end) to drop rows within index range.
        :param row_condition: A lambda condition to apply to rows.
        """
        if columns:
            self.df = self.df.drop(columns=columns, axis=1, errors='ignore')

        if row_range:
            start, end = row_range
            self.df = self.df.loc[(self.df.index < start) | (self.df.index > end)]

        if row_condition:
            self.df = self.df[~self.df.map_partitions(lambda df: df.apply(row_condition, axis=1))]

        self.df = self.df.persist()

    def find_index_where_max(self, column_to_max, condition_column):
        """
        Finds the index and value in column_to_max where condition_column has its maximum.
        Can be used to determine the exact moment of ignition, so it then can be mapped as the time 0
        :param column_to_max: column to return the value from
        :param condition_column: column to check for maximum value
        :return: (index, value)
        """
        max_row = self.df[self.df[condition_column] == self.df[condition_column].max()]
        result = max_row[[column_to_max]].compute()
        index = result.index[0]
        if column_to_max:
            value = result[column_to_max].iloc[0]
            return index, value
        return index

    @sync_with_wrapper
    def offset(self, columns, offset):
        """
        Offset data  by a given float, positive or negative
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            self.df[column] += offset

    @sync_with_wrapper
    def flip_column_sign(self, columns):
        """
        Flips the sign of all values in the specified column(s).
        Positive values become negative and vice versa.
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            self.df[column] = -self.df[column]

    @sync_with_wrapper
    def absolute(self, columns):
        for column in columns:
            self.df[column] = abs(self.df[column])

    @sync_with_wrapper
    def rename_column(self, columns, new_name):
        self.df = self.df.rename(columns={columns: new_name})

    def get_processed_data(self):
        """Returns the processed DataFrame."""
        return self.df

    def save_data(self, path):
        """Save the processed DataFrame."""
        dd.to_csv(self.df, path, single_file=True, index=False)
