import matplotlib.pyplot as plot
import os
import matplotlib.ticker as ticker
import numpy as np
from scipy import signal
import matplotlib as mpl

mpl.use('QtAgg')

mpl.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


class Plotter:
    def __init__(self, config_dict=None, dataframe_map=None, plots_folder_path="plots"):
        self.config = config_dict
        if config_dict and dataframe_map is not None:
            self.dataframes = dataframe_map  # mapping: {"db1": DataFrameWrapper, ...}
            self.plot_name = config_dict["plot_settings"].get("title", "Plot")
            self.plot_type = config_dict["plot_settings"].get("type", "line")
            self.precise_grid = config_dict["plot_settings"].get("precise_grid", False)
            self.convert_epoch = config_dict["plot_settings"].get("convert_epoch", None)
            self.offset = config_dict["plot_settings"].get("offset", 0)
            self.secondary_db_offset = config_dict["plot_settings"].get("secondary_db_offset", 0)
            self.axis_labels = {
                "x": config_dict["plot_settings"].get("x_axis_label", "X-Axis"),
                "y1": config_dict["plot_settings"].get("y_axis_labels", {}).get("y1", "Y1-Axis"),
                "y2": config_dict["plot_settings"].get("y_axis_labels", {}).get("y2", "Y2-Axis")
            }

            self.horizontal_lines = config_dict["plot_settings"].get("horizontal_lines", {})
            self.vertical_lines = config_dict["plot_settings"].get("vertical_lines", {})
        self.plots_folder_path = plots_folder_path
        os.makedirs(self.plots_folder_path, exist_ok=True)

    def prepare_plot_data(self):
        """
        Prepare plot data by computing all Dask operations.
        This can run in a worker thread to avoid blocking the main thread.
        Returns a dictionary with all pre-computed numpy arrays.
        """
        plot_data = {}

        for db_key, db_config in self.config["databases"].items():
            df_wrapper = self.dataframes[db_key]
            df = df_wrapper.get_dataframe()
            plot_data[db_key] = {}

            for channel, ch_conf in db_config["channels"].items():
                x_column = ch_conf.get("x_column", "index")
                y_axis = ch_conf.get("y_axis", "y1")
                label = ch_conf.get("label", channel)
                color = ch_conf.get("color", None)
                alpha = ch_conf.get("alpha", 1.0)
                size = ch_conf.get("size", 1)

                effective_offset = self.offset
                if db_key == "db2" and self.convert_epoch != "none":
                    effective_offset += self.secondary_db_offset

                # Compute x values
                if x_column in df.columns:
                    match self.convert_epoch:
                        case "seconds":
                            x_values = df[x_column].compute()
                            x_values = (x_values - x_values.min() + effective_offset) / 1000
                        case "miliseconds":
                            x_values = df[x_column].compute()
                            x_values = x_values - x_values.min() + effective_offset
                        case _:
                            x_values = df[x_column].compute() + effective_offset
                else:
                    x_values = df.index.compute() + effective_offset

                # Compute y values
                if channel in df.columns:
                    y_values = df[channel].compute()
                else:
                    y_values = None

                # Store computed data
                plot_data[db_key][channel] = {
                    "x": x_values.to_numpy() if hasattr(x_values, 'to_numpy') else np.array(x_values),
                    "y": y_values.to_numpy() if hasattr(y_values, 'to_numpy') and y_values is not None else (
                        np.array(y_values) if y_values is not None else None),
                    "label": label,
                    "color": color,
                    "alpha": alpha,
                    "size": size,
                    "y_axis": y_axis
                }

        return plot_data

    def plot_from_data(self, plot_data):
        """
        Render plot from pre-computed data. This runs on the main thread.
        Call prepare_plot_data() in a worker thread, then call this on the main thread.
        """
        fig, ax1 = plot.subplots(figsize=(10, 7))
        ax2 = None

        # Check if any channel is assigned to y2
        has_y2 = any(
            data.get("y_axis") == "y2"
            for db_config in plot_data.values()
            for data in db_config.values()
        )
        if has_y2:
            ax2 = ax1.twinx()

        legend_handles = []

        # Plot all channels using pre-computed data
        for db_key, channels_data in plot_data.items():
            for channel, ch_data in channels_data.items():
                x_values = ch_data["x"]
                y_values = ch_data["y"]
                y_axis = ch_data["y_axis"]
                label = ch_data["label"]
                color = ch_data["color"]
                alpha = ch_data["alpha"]
                size = ch_data["size"]

                if y_values is None:
                    continue

                if y_axis == "y1":
                    if self.plot_type == "line":
                        handle, = ax1.plot(x_values, y_values, label=label, color=color, alpha=alpha)
                    else:
                        handle = ax1.scatter(x_values, y_values, s=size, label=label, color=color, alpha=alpha)
                elif ax2:
                    if self.plot_type == "line":
                        handle, = ax2.plot(x_values, y_values, label=label, color=color, alpha=alpha)
                    else:
                        handle = ax2.scatter(x_values, y_values, s=size, label=label, color=color, alpha=alpha)
                legend_handles.append(handle)

        # Axis labels
        ax1.set_xlabel(self.axis_labels.get("x", "X-Axis"))
        ax1.set_ylabel(self.axis_labels.get("y1", "Y1-Axis"))
        if ax2:
            ax2.set_ylabel(self.axis_labels.get("y2", "Y2-Axis"))

        # Draw horizontal lines
        for line in self.horizontal_lines.values():
            y = line.get("place")
            label = line.get("label")
            color = line.get("color", "black")
            axis = line.get("axis", "y1")
            if axis == "y1":
                handle = ax1.axhline(y=y, color=color, linestyle='--', label=label)
            else:
                handle = ax2.axhline(y=y, color=color, linestyle='--', label=label)
            legend_handles.append(handle)

        # Draw vertical lines
        for line in self.vertical_lines.values():
            x = line.get("place")
            label = line.get("label")
            color = line.get("color", "black")
            axis = line.get("axis", "y1")
            if axis == "y1":
                handle = ax1.axvline(x=x, color=color, linestyle='--', label=label)
            else:
                handle = ax2.axvline(x=x, color=color, linestyle='--', label=label)
            legend_handles.append(handle)

        # Grid setup
        if self.precise_grid:
            ax1.xaxis.set_major_locator(ticker.AutoLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.yaxis.set_major_locator(ticker.AutoLocator())
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ax2:
            if self.precise_grid:
                ax2.yaxis.set_major_locator(ticker.AutoLocator())
                ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Legend and layout
        plot.subplots_adjust(bottom=0.2)
        plot.legend(handles=legend_handles, bbox_to_anchor=(0.5, 0.02), loc="lower center",
                    bbox_transform=fig.transFigure, fancybox=True, shadow=True, ncol=3)

        plot.title(self.plot_name)
        self.save_plot(self.plot_name)

        # This blocks on the main thread, allowing interactive use of the plot
        plot.show()

    @staticmethod
    def _compute_fs(x):
        """
        Tries to make out the fs parameter  out of the x-axis
        :param x:
        :return:
        """
        if x is None or len(x) < 3:
            return 1.0
        dx = np.diff(x)
        dx = dx[np.isfinite(dx)]
        if dx.size == 0:
            return 1.0
        med = float(np.median(dx))
        if med <= 0:
            return 1.0
        if med > 5:
            med = med / 1000.0
        return 1.0 / med

    def _series_from_cfg(self, db_key, channel, x_column=None):
        df = self.dataframes[db_key].get_dataframe()
        y = df[channel].compute().to_numpy()
        x = None
        if x_column and x_column in df.columns:
            x = df[x_column].compute().to_numpy()
        return x, y

    def compute_fft(self, db_key, channel, fs=None, nfft=None, window="hann", detrend=False, db_scale=True, max_freq=None,
                    x_column=None, title=None):
        """Compute FFT data (can run in worker thread). Returns dict with FFT arrays."""
        x, y = self._series_from_cfg(db_key, channel, x_column)

        if fs is None:
            fs = self._compute_fs(x)
        if detrend:
            y = signal.detrend(y)
        if window:
            try:
                win = signal.get_window(window, len(y))
                y *= win
            except Exception:
                pass

        n = len(y) if nfft is None else int(nfft)
        y_fft = np.fft.rfft(y, n=n)
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        Pxx = (1 / (fs * n)) * np.abs(y_fft) ** 2

        if db_scale:
            eps = 1e-12
            Pxx = 10 * np.log10(Pxx + eps)

        if max_freq is not None:
            mask = f <= float(max_freq)
            f = f[mask]
            Pxx = Pxx[mask]

        return {
            "f": f,
            "Pxx": Pxx,
            "channel": channel,
            "title": title,
            "db_scale": db_scale
        }

    def compute_spectrogram(self, db_key, channel, fs=None, nperseg=256, noverlap=None, window="hann", mode="psd",
                            db_scale=True, cmap="viridis", x_column=None, title=None):
        """Compute spectrogram data (can run in worker thread). Returns dict with spectrogram arrays."""
        x, y = self._series_from_cfg(db_key, channel, x_column)

        # Validate and clean data
        if y is None or len(y) == 0:
            raise ValueError(f"No data found for channel {channel}")

        # Remove NaN/Inf values
        valid_mask = np.isfinite(y)
        if not np.any(valid_mask):
            raise ValueError(f"All data values are NaN or Inf for channel {channel}")
        y = y[valid_mask]
        if x is not None:
            x = x[valid_mask]

        # Compute fs if not provided
        if fs is None:
            fs = self._compute_fs(x)
        if fs is None or fs <= 0:
            fs = 1.0

        # Validate nperseg
        nperseg = int(nperseg)
        if nperseg <= 0:
            nperseg = 256
        if nperseg > len(y):
            # If nperseg is too large, reduce it
            nperseg = len(y) // 4 if len(y) > 4 else len(y)

        # Validate and set noverlap
        if noverlap is None:
            noverlap = int(0.5 * nperseg)
        else:
            noverlap = int(noverlap)

        # noverlap must be strictly less than nperseg
        if noverlap >= nperseg:
            noverlap = int(0.5 * nperseg)
        if noverlap < 0:
            noverlap = 0

        try:
            f, t, Sxx = signal.spectrogram(
                y,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                mode=mode,
                detrend=False,
                scaling="density"
            )
        except Exception as e:
            raise ValueError(f"Spectrogram computation failed: {e}")

        # Ensure Sxx is valid before processing
        if np.all(Sxx == 0):
            raise ValueError("Spectrogram produced all-zero result. Check input data.")

        if db_scale:
            # Handle potential negative or zero values before log
            Sxx = np.maximum(Sxx, 1e-12)  # Clamp to avoid log errors
            Sxx = 10 * np.log10(Sxx)

        return {
            "f": f,
            "t": t,
            "Sxx": Sxx,
            "channel": channel,
            "title": title,
            "db_scale": db_scale,
            "cmap": cmap,
            "mode": mode
        }

    def plot_spectrogram_data(self, spec_data):
        """Plot pre-computed spectrogram data (must run on main thread)."""
        f = spec_data["f"]
        t = spec_data["t"]
        Sxx = spec_data["Sxx"]
        channel = spec_data["channel"]
        title = spec_data["title"]
        db_scale = spec_data["db_scale"]
        cmap = spec_data["cmap"]
        mode = spec_data["mode"]

        # Create figure
        fig, ax = plot.subplots(figsize=(10, 6))
        m = ax.pcolormesh(t, f, Sxx, shading="auto", cmap=cmap)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        cbar = plot.colorbar(m, ax=ax)
        cbar.set_label("Power [dB]" if db_scale else ("PSD" if mode == "psd" else "Magnitude"))
        ax.set_title(title or f"Spectrogram: {channel}")

        # Only set ylim if f has valid values
        if len(f) > 0 and f.max() > 0:
            ax.set_ylim(0, f.max())

        self.save_plot(title or f"Spectrogram_{channel}")
        plot.show()

    def save_plot(self, filename):
        path = os.path.join(self.plots_folder_path, f"{filename}.png")
        plot.savefig(path)
