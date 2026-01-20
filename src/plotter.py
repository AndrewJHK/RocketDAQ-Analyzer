import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import os
import matplotlib.ticker as ticker
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"""
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{polski}
"""
mpl.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


rc('text', usetex=True)


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
        self.anim = None
        os.makedirs(self.plots_folder_path, exist_ok=True)

    def plot(self):
        fig, ax1 = plot.subplots(figsize=(10, 7))
        ax2 = None

        # Check if any channel is assigned to y2
        has_y2 = any(
            ch.get("y_axis") == "y2"
            for db in self.config["databases"].values()
            for ch in db["channels"].values()
        )
        if has_y2:
            ax2 = ax1.twinx()

        legend_handles = []

        for db_key, db_config in self.config["databases"].items():
            df_wrapper = self.dataframes[db_key]
            df = df_wrapper.get_dataframe()

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

                if channel in df.columns:
                    y_values = df[channel].compute()

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
        plot.show()

    def flight_plot_orientation(self, save, orientation):

        t = np.arange(0, len(orientation)) * 0.01

        fig_gyro, ax_gyro = plot.subplots()
        ax_gyro.plot(t, orientation[:, 0], label="x")
        ax_gyro.plot(t, orientation[:, 1], label="y")
        ax_gyro.plot(t, orientation[:, 2], label="z")
        ax_gyro.set_title("Rocket Rotation")
        ax_gyro.set_xlabel("Time [s]")
        ax_gyro.set_ylabel("Angular velocity [rad/s]")
        ax_gyro.grid()
        ax_gyro.legend()

        fig_orient = plot.figure()
        ax_orient = fig_orient.add_subplot(projection="3d")

        self.anim = FuncAnimation(
            fig_orient,
            func=self.animate_orientation,
            fargs=(ax_orient, orientation),
            frames=len(orientation),
            interval=10,
            repeat_delay=1000,
        )

        if save:
            fig_gyro.savefig("plots/gyro.png")
            self.anim.save("plots/orientation.mp4", writer="ffmpeg")

        plot.show()

    @staticmethod
    def flight_plot_velocity(save, data):
        t = np.arange(0, len(data[:, 0])) * 0.01

        fig_alt, ax_alt = plot.subplots()
        ax_alt.plot(t, data[:, 8])
        ax_alt.set_xlabel("Time [s]")
        ax_alt.set_ylabel("Altitude [m]")
        ax_alt.set_title("Altitude")
        ax_alt.grid()

        fig_vel, ax_vel = plot.subplots()
        ax_vel.plot(t, data[:, 5])
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Velocity [m/s]")
        ax_vel.set_title("Velocity")
        ax_vel.grid()

        fig2 = plot.figure()
        ax2 = fig2.add_subplot(projection="3d")
        ax2.plot(data[:, 6], data[:, 7], data[:, 8])
        ax2.plot(data[:, 6], data[:, 7], np.zeros_like(data[:, 8]), linestyle="--")
        ax2.plot(np.zeros_like(data[:, 6]), data[:, 7], data[:, 8], linestyle="--")
        ax2.plot(data[:, 6], np.zeros_like(data[:, 7]), data[:, 8], linestyle="--")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Position")
        ax2.grid()

        fig_acc, ax_acc = plot.subplots()
        ax_acc.plot(t, data[:, 0], label="x acceleration")
        ax_acc.plot(t, data[:, 1], label="y acceleration")
        ax_acc.set_xlabel("Time [s]")
        ax_acc.set_ylabel("Acceleration [m/s^2]")
        ax_acc.grid()
        ax_acc.legend()

        if save:
            fig_alt.savefig("plots/altitude.png")
            fig_vel.savefig("plots/velocity.png")
            fig2.savefig("plots/position.png")
            fig_acc.savefig("plots/altitude.png")

        plot.show()

    @staticmethod
    def animate_orientation(i, ax, quaternions):
        ax.clear()
        ax.quiver(0, 0, 0, 1, 0, 0, color="r", alpha=0.5, linestyle="--", normalize=True)
        ax.quiver(0, 0, 0, 0, 1, 0, color="g", alpha=0.5, linestyle="--", normalize=True)
        ax.quiver(0, 0, 0, 0, 0, 1, color="b", alpha=0.5, linestyle="--", normalize=True)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Orientation")

        rot = Rotation.from_quat(quaternions[i, :])
        x_after_rot = rot.apply(np.array([1, 0, 0]))
        y_after_rot = rot.apply(np.array([0, 1, 0]))
        z_after_rot = rot.apply(np.array([0, 0, 1]))

        ax.quiver(
            0,
            0,
            0,
            x_after_rot[0],
            x_after_rot[1],
            x_after_rot[2],
            color="r",
            normalize=True,
        )
        ax.quiver(
            0,
            0,
            0,
            y_after_rot[0],
            y_after_rot[1],
            y_after_rot[2],
            color="g",
            normalize=True,
        )
        ax.quiver(
            0,
            0,
            0,
            z_after_rot[0],
            z_after_rot[1],
            z_after_rot[2],
            color="b",
            normalize=True,
        )

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

    def plot_fft(self, db_key, channel, fs=None, nfft=None, window="hann", detrend=False, db_scale=True, max_freq=None,
                 x_column=None, title=None):
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

        fig, ax = plot.subplots(figsize=(10, 6))
        ax.plot(f, Pxx)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(
            r'Power spectral density [$\frac{dB}{Hz}$]' if db_scale else r'Power spectral density [$\frac{V^2}{Hz}$]')
        ax.grid(True)
        plot.title(title or f"FFT: {channel}")

        self.save_plot(title or f"FFT_{channel}")
        plot.show()

    def plot_spectrogram(self, db_key, channel, fs=None, nperseg=256, noverlap=None, window="hann", mode="psd",
                         db_scale=True, cmap="viridis", x_column=None, title=None):
        x, y = self._series_from_cfg(db_key, channel, x_column)

        if fs is None:
            fs = self._compute_fs(x)
        if fs is None:
            fs = 1.0
        if noverlap is None:
            noverlap = int(0.5 * nperseg)

        f, t, Sxx = signal.spectrogram(
            y,
            fs=fs,
            window=window,
            nperseg=int(nperseg),
            noverlap=int(noverlap),
            mode=mode,
            detrend=False,
            scaling="density"
        )

        if db_scale:
            Sxx = 10 * np.log10(Sxx + 1e-12)

        fig, ax = plot.subplots(figsize=(10, 6))
        m = ax.pcolormesh(t, f, Sxx, shading="auto", cmap=cmap)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        cbar = plot.colorbar(m, ax=ax)
        cbar.set_label("Power [dB]" if db_scale else ("PSD" if mode == "psd" else "Magnitude"))
        ax.set_title(title or f"Spectrogram: {channel}")
        ax.set_ylim(0, f.max())

        self.save_plot(title or f"Spectrogram_{channel}")
        plot.show()

    def save_plot(self, filename):
        path = os.path.join(self.plots_folder_path, f"{filename}.png")
        plot.savefig(path)
