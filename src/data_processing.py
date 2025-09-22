import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q
from src.filters import DataFilter

EQUATIONS = {"biliq": "placeholder", "broom_stick": "placeholder"}


def sync_with_wrapper(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.df_wrapper.update_dataframe(self.df)
        return result

    return wrapper


class DataFrameWrapper:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = dd.read_csv(csv_path)

    def get_dataframe(self):
        return self.df

    def get_csv_path(self):
        return self.csv_path

    def update_dataframe(self, dataframe):
        self.df = dataframe

    def get_raw_pandas(self):
        return pd.read_csv(self.csv_path, parse_dates=["header.timestamp_epoch"], index_col="header.counter")


class DataProcessor:
    def __init__(self, df_wrapper):
        """Initializes processor with a DataFrameWrapper instance."""
        self.df_wrapper = df_wrapper
        self.df = df_wrapper.get_dataframe()
        self.filter_manager = DataFilter()  # Handles filter strategies

    def get_filters(self):
        return self.filter_manager.get_filter_queue()

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

    @sync_with_wrapper
    def scale_index_by_equation(self, equation_func, start_idx=None, end_idx=None):
        """
        Scales the DataFrame index based on a time-dependent function.

        :param equation_func: a function that accepts index and returns scale factor
        :param start_idx: optional starting index for applying the scale
        :param end_idx: optional ending index for applying the scale
        """

        def apply_func(partition):
            idx = partition.index
            if start_idx is not None:
                partition = partition.loc[(idx >= start_idx)]
            if end_idx is not None:
                partition = partition.loc[(idx <= end_idx)]
            scale_factors = equation_func(idx)
            partition.index = idx * scale_factors
            return partition

        self.df = self.df.map_partitions(apply_func)

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

    def interpolate_index(self):
        df_pd = self.df_wrapper.get_raw_pandas()
        df_pd = df_pd.copy()
        df_pd["Time"] = (df_pd["header.timestamp_epoch"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("10ms")
        df_pd["Time"] = df_pd["Time"] - df_pd["Time"].iloc[0]
        df_pd.set_index("Time", inplace=True)
        df_pd = df_pd.reindex(range(df_pd.index.min(), df_pd.index.max() + 1), fill_value=pd.NA)
        df_pd.interpolate(inplace=True)
        self.df = dd.from_pandas(df_pd, npartitions=1)

    def compute_orientation(self):
        df_pd = self.df.compute()
        df_pd = df_pd.copy()
        acc = df_pd[["data.telemetry.acc_data.x", "data.telemetry.acc_data.y", "data.telemetry.acc_data.z"]].to_numpy()
        gyro = df_pd[["data.telemetry.quaternion.roll", "data.telemetry.quaternion.pitch",
                      "data.telemetry.quaternion.heading"]].to_numpy()

        acc[:, 0] -= np.mean(acc[:20, 0])
        acc[:, 1] -= np.mean(acc[:20, 2])

        gyro[:, 0] -= np.mean(gyro[:20, 0])
        gyro[:, 1] -= np.mean(gyro[:20, 1])
        gyro[:, 2] -= np.mean(gyro[:20, 2])

        ekf = EKF(
            gyr=gyro,
            acc=acc,
            q0=acc2q(acc[0, :]),
            dt=0.01,
            var_gyr=0.1 ** 2,
            var_acc=0.3 ** 2,
            frame="ENU",
        )

        df_pd["data.telemetry.quaternion.q0_computed"] = ekf.Q[:, 1]
        df_pd["data.telemetry.quaternion.q1_computed"] = ekf.Q[:, 2]
        df_pd["data.telemetry.quaternion.q2_computed"] = ekf.Q[:, 3]
        df_pd["data.telemetry.quaternion.q3_computed"] = ekf.Q[:, 0]

        self.df = dd.from_pandas(df_pd, npartitions=1)

    def compute_position_from_orientation(self):
        df_pd = self.df.compute()
        df_pd = df_pd.copy()
        acc = df_pd[["data.telemetry.acc_data.x", "data.telemetry.acc_data.y", "data.telemetry.acc_data.z"]].to_numpy()
        orient = df_pd[["data.telemetry.quaternion.q0_computed", "data.telemetry.quaternion.q1_computed",
                        "data.telemetry.quaternion.q2_computed",
                        "data.telemetry.quaternion.q3_computed"]].to_numpy()

        acc[:, 0] -= np.mean(acc[:20, 0])
        acc[:, 1] -= np.mean(acc[:20, 2])

        acc_global = np.zeros_like(acc)
        for i in range(len(acc_global)):
            rot = Rotation.from_quat(orient[i, :])
            acc_global[i, :] = rot.apply(acc[i, :])

        x_acc, y_acc, z_acc = acc_global.T
        z_acc_offset = np.mean(z_acc[:20])
        z_acc -= z_acc_offset

        accel_coef = 9.81 / z_acc_offset
        x_acc *= accel_coef
        y_acc *= accel_coef
        z_acc *= accel_coef

        x_vel = np.cumsum(x_acc) * 0.01
        y_vel = np.cumsum(y_acc) * 0.01
        z_vel = np.cumsum(z_acc) * 0.01

        x_pos = np.cumsum(x_vel) * 0.01
        y_pos = np.cumsum(y_vel) * 0.01
        z_pos = np.cumsum(z_vel) * 0.01

        df_pd["Computed_Acc_X"] = x_acc
        df_pd["Computed_Acc_Y"] = y_acc
        df_pd["Computed_Acc_Z"] = z_acc
        df_pd["Computed_Vel_X"] = x_vel
        df_pd["Computed_Vel_Y"] = y_vel
        df_pd["Computed_Vel_Z"] = z_vel
        df_pd["Computed_Pos_X"] = x_pos
        df_pd["Computed_Pos_Y"] = y_pos
        df_pd["Computed_Pos_Z"] = z_pos

        self.df = dd.from_pandas(df_pd, npartitions=1)

    @sync_with_wrapper
    def compute_flight_profile(self):
        columns_to_keep = ["header.timestamp_epoch", "data.telemetry.acc_data.x", "data.telemetry.acc_data.y",
                           "data.telemetry.acc_data.z",
                           "data.telemetry.quaternion.roll", "data.telemetry.quaternion.pitch",
                           "data.telemetry.quaternion.heading",
                           "data.telemetry.quaternion.q0", "data.telemetry.quaternion.q1",
                           "data.telemetry.quaternion.q2",
                           "data.telemetry.quaternion.q3"]
        columns_to_drop = [col for col in self.df.columns if col not in columns_to_keep]
        self.drop_data(columns=columns_to_drop)
        self.interpolate_index()
        self.compute_orientation()
        self.compute_position_from_orientation()

    def get_processed_data(self):
        """Returns the processed DataFrame."""
        return self.df

    def save_data(self, path):
        """Save the processed DataFrame."""
        dd.to_csv(self.df, path, single_file=True)
