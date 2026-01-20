import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfiltfilt
import pywt
import pandas as pd
import dask.dataframe as dd
import re
import hashlib


class DataFilter:
    def __init__(self, log_fn=None):
        self.filters = {}
        self.log_fn = log_fn
        self.strategy_map = {
            "adaptive_kalman": AdaptiveKalman(),
            "low-pass": LowPassFilter(),
            "remove_negatives": RemoveNegativesFilter(),
            "remove_positives": RemovePositivesFilter(),
            "rolling_mean": RollingMeanFilter(),
            "rolling_median": RollingMedianFilter(),
            "threshold": ThresholdFilter(),
            "wavelet_transform": WaveletTransformFilter()
        }

    def _log(self, msg):
        if self.log_fn:
            self.log_fn(msg, "INFO")

    @staticmethod
    def _snr_db(before, after, eps=1e-12):
        x = pd.to_numeric(before, errors="coerce").to_numpy(dtype="float64")
        y = pd.to_numeric(after, errors="coerce").to_numpy(dtype="float64")

        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return float("nan")

        noise = x[m] - y[m]
        sig_var = float(np.var(y[m], ddof=0))
        noi_var = float(np.var(noise, ddof=0))

        if noi_var <= eps:
            return float("inf")
        return 10.0 * np.log10((sig_var + eps) / (noi_var + eps))

    def get_filter_queue(self):
        return self.filters

    def add_filter(self, columns, filter_name, **kwargs):
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            if column not in self.filters:
                self.filters[column] = deque()
            strategy = self.strategy_map.get(filter_name)
            if not strategy:
                raise ValueError(f"Filter '{filter_name}' not recognized.")
            self.filters[column].append((filter_name, strategy, kwargs))

    def queue_filters(self, df):
        # Each filter produces a new column; the next filter uses the previous output as input.
        for source_col, queue in self.filters.items():
            if source_col not in df.columns:
                continue

            base_s = df[source_col].compute()
            current_s = base_s
            current_col = source_col

            for filter_name, strategy, params in queue:
                out_col = self._make_filtered_col_name(current_col, filter_name, params)

                # Ensure uniqueness if the column already exists
                if out_col in df.columns:
                    i = 2
                    alt = f"{out_col}__{i}"
                    while alt in df.columns:
                        i += 1
                        alt = f"{out_col}__{i}"
                    out_col = alt

                out_s = strategy.apply(current_s, **params)

                if not isinstance(out_s, pd.Series):
                    out_s = pd.Series(out_s, index=current_s.index)

                out_s = out_s.rename(out_col)

                out_dd = dd.from_pandas(out_s, npartitions=df.npartitions)
                out_dd = out_dd.repartition(divisions=df.divisions)

                df[out_col] = out_dd

                df[out_col] = dd.from_pandas(out_s, npartitions=df.npartitions)

                current_s = out_s
                current_col = out_col

                snr_after = self._snr_db(base_s, out_s)

                self._log(f"[{source_col}] {filter_name} -> '{out_col}', SNR: {snr_after:.2f} dB")

        self.filters.clear()
        return df

    @staticmethod
    def _format_param_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        try:
            if isinstance(v, np.generic):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    @staticmethod
    def _sanitize_col_token(s):
        s = s.strip()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^0-9a-zA-Z_=\-\.]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    def _make_filtered_col_name(self, base_col: str, filter_name: str, params: dict) -> str:
        if params:
            parts = []
            for k in sorted(params.keys()):
                parts.append(f"{k}={self._format_param_value(params[k])}")
            param_token = "__".join(self._sanitize_col_token(p) for p in parts if p)
            name = f"{base_col}__{filter_name}__{param_token}"
        else:
            name = f"{base_col}__{filter_name}"

        name = self._sanitize_col_token(name)

        # Hard cap length, hash is in place to protect against too long col names for pandas to handle
        if len(name) > 128:
            h = hashlib.md5(name.encode("utf-8")).hexdigest()[:10]
            name = f"{name[:110]}__{h}"
        return name


class FilterStrategy:
    def apply(self, series, **kwargs):
        raise NotImplementedError("Each filter must implement the apply method.")


class RemoveNegativesFilter(FilterStrategy):
    def apply(self, series, **kwargs):
        return series.where(series >= 0, 0)


class RemovePositivesFilter(FilterStrategy):
    def apply(self, series, **kwargs):
        return series.where(series <= 0, 0)


class RollingMeanFilter(FilterStrategy):
    def apply(self, series, window_size=3, **kwargs):
        return series.rolling(window=window_size, min_periods=1).mean()


class RollingMedianFilter(FilterStrategy):
    def apply(self, series, window_size=3, **kwargs):
        return series.rolling(window=window_size, min_periods=1).median()


class ThresholdFilter(FilterStrategy):
    def apply(self, series, threshold=100, **kwargs):
        return series.where(series < threshold, threshold)


class LowPassFilter(FilterStrategy):
    def apply(self, series, order=4, cutoff=None, fs=None, zero_phase=True, **kwargs):
        if fs is None or fs <= 0:
            raise ValueError("Parameter 'fs' (sampling frequency) must be > 0.")
        if cutoff is None or not (0 < float(cutoff) < 0.5 * fs):
            raise ValueError("Parameter 'cutoff' must be within (0, fs/2).")
        try:
            order = int(order)
        except Exception:
            raise ValueError("Parameter 'order' must be an integer.")

        if order < 1:
            raise ValueError("Parameter 'order' must be ≥ 1.")

        s_num = pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')
        x = s_num.to_numpy(dtype="float64")

        nan_mask = np.isnan(x)
        if np.any(nan_mask):
            not_nan = ~nan_mask
            if not_nan.sum() >= 2:
                x[nan_mask] = np.interp(
                    np.flatnonzero(nan_mask),
                    np.flatnonzero(not_nan),
                    x[not_nan]
                )
            x[np.isnan(x)] = 0.0  # fallback

        nyq = 0.5 * fs
        wn = float(cutoff) / nyq
        sos = butter(order, wn, btype="low", analog=False, output="sos")

        try:
            y = sosfiltfilt(sos, x) if zero_phase else sosfilt(sos, x)
        except ValueError:
            y = sosfilt(sos, x)

        y[nan_mask] = np.nan
        return pd.Series(y, index=series.index, name=series.name)


class WaveletTransformFilter(FilterStrategy):
    def apply(self, series, wavelet_name='db4', level=2, threshold_mode='soft', epsilon=1e-12, **kwargs):
        s_num = pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')
        x = s_num.to_numpy(dtype="float64")

        nan_mask = np.isnan(x)
        if np.any(nan_mask):
            not_nan = ~nan_mask
            if not_nan.sum() >= 2:
                x[nan_mask] = np.interp(
                    np.flatnonzero(nan_mask),
                    np.flatnonzero(not_nan),
                    x[not_nan]
                )
            x[np.isnan(x)] = 0.0

        try:
            lvl = int(level)
            coeffs = pywt.wavedec(x, wavelet_name, level=lvl)
            idx = min(lvl, len(coeffs) - 1)
            detail = coeffs[idx]

            sigma_p = np.nanstd(detail)
            n = max(len(x), 2)

            sig_pow = np.nanmean(x * x)
            noise_pow = np.nanmean(detail * detail)
            snr_p = np.sqrt(max(sig_pow / (noise_pow + epsilon), 0.0)) if noise_pow > 0 else 1.0

            omega_base = sigma_p * np.sqrt(2.0 * np.log(n))
            omega_opt = omega_base * snr_p

            def adaptive_thresh(ci, omega):
                ci = np.asarray(ci, dtype=float)
                a = np.abs(ci)
                out = np.empty_like(ci)
                mask_high = a > omega
                out[mask_high] = ci[mask_high]
                m = ~mask_high
                out[m] = ci[m] * (a[m] - omega) / (a[m] + epsilon)
                return out

            if threshold_mode == 'adaptive':
                coeffs_thr = [adaptive_thresh(ci, omega_opt) for ci in coeffs]
            else:
                thr = omega_base
                coeffs_thr = [pywt.threshold(ci, thr, mode=threshold_mode) for ci in coeffs]

            y = pywt.waverec(coeffs_thr, wavelet_name)
            y = y[:len(x)]
            y[nan_mask] = np.nan
        except Exception:
            y = x

        return pd.Series(y, index=series.index, name=series.name)


class AdaptiveKalman(FilterStrategy):
    def apply(self, series, model="cv", fs=None, R=1e-2, Q0=1e-3, tau=3.0,
              q_up=5.0, q_down=0.9, q_min=1e-9, q_max=1e+3,
              x0=None, v0=0.0, P0=None, zero_phase=False, **kwargs):
        if fs is None or fs <= 0:
            raise ValueError("AdaptiveKalman1D: parameter 'fs' must be > 0.")
        dt = 1.0 / float(fs)

        s_num = pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')
        y = s_num.to_numpy(dtype=float)
        n = y.size

        finite = np.isfinite(y)
        if not finite.any():
            return s_num  # awaryjnie: nic się nie da policzyć

        out = np.full(n, np.nan, dtype=float)

        i = 0
        while i < n:
            if not finite[i]:
                i += 1
                continue
            j = i
            while j < n and finite[j]:
                j += 1
            seg = y[i:j]

            match model:
                case "constant":
                    x0_eff = seg[0] if x0 is None else float(x0)
                    X = np.array([x0_eff])
                    P = np.array([[1.0]]) if P0 is None else np.array(P0).reshape(1, 1)
                    F = np.array([[1.0]])
                    H = np.array([[1.0]])
                    Rm = np.array([[float(R)]])
                    I = np.array([[1.0]])
                    q = float(Q0)

                    for k in range(seg.size):
                        Qm = np.array([[q]])
                        X = F @ X
                        P = F @ P @ F.T + Qm
                        S = H @ P @ H.T + Rm
                        S_scalar = float(S)
                        K = (P @ H.T) / S_scalar
                        pred = float(H @ X)
                        innov = seg[k] - pred
                        X = X + K * innov
                        P = (I - K @ H) @ P
                        out[i + k] = X[0]
                        s_var = S_scalar
                        thr = tau * np.sqrt(s_var) if s_var > 0 else np.inf
                        if abs(innov) > thr:
                            q = min(q * q_up, q_max)
                        else:
                            q = max(q * q_down, q_min)

                case "cv":
                    x0_eff = seg[0] if x0 is None else float(x0)
                    X = np.array([x0_eff, float(v0)])
                    P = np.eye(2) if P0 is None else np.array(P0).reshape(2, 2)
                    F = np.array([[1.0, dt],
                                  [0.0, 1.0]])
                    H = np.array([[1.0, 0.0]])
                    Rm = np.array([[float(R)]])
                    I2 = np.eye(2)
                    G = np.array([[0.5 * dt * dt],
                                  [dt]])
                    q = float(Q0)

                    for k in range(seg.size):
                        Qm = q * (G @ G.T)
                        X = F @ X
                        P = F @ P @ F.T + Qm
                        S = H @ P @ H.T + Rm
                        S_scalar = float(S)
                        K = (P @ H.T) / S_scalar
                        pred = float(H @ X)
                        innov = seg[k] - pred
                        X = X + (K.flatten() * innov)
                        P = (I2 - K @ H) @ P
                        out[i + k] = X[0]
                        s_var = S_scalar
                        thr = tau * np.sqrt(s_var) if s_var > 0 else np.inf
                        if abs(innov) > thr:
                            q = min(q * q_up, q_max)
                        else:
                            q = max(q * q_down, q_min)

                case _:
                    raise ValueError("AdaptiveKalman1D: model must be 'constant' or 'cv'.")

            i = j

        return pd.Series(out, index=series.index, name=series.name)
