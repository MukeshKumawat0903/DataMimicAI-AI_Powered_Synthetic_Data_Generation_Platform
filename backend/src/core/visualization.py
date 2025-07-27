# src/core/visualization.py

import pandas as pd
from typing import List, Optional, Dict
from sdv.evaluation.single_table import get_column_plot, get_column_pair_plot
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats

pio.templates.default = "plotly_white"


class DataVisualizer:
    """
    A class for visualizing and comparing real vs. synthetic tabular data.

    Provides summary statistics, distribution plots, drift detection, and correlation heatmaps.
    """

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict):
        """
        Initialize the DataVisualizer.

        Args:
            real_data (pd.DataFrame): The real/original data.
            synthetic_data (pd.DataFrame): The synthetic/generated data.
            metadata (dict): Table metadata (SDV format).
        """
        self.real_data = self.clean_dataframe_for_visualization(real_data)
        self.synthetic_data = self.clean_dataframe_for_visualization(synthetic_data)
        self.metadata = metadata

    @staticmethod
    def clean_dataframe_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all object columns to strings, to avoid Arrow/Streamlit issues.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return df

    @staticmethod
    def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with only numeric columns (for corr and heatmaps).

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Numeric-only DataFrame.
        """
        return df.select_dtypes(include="number")

    def distribution_plot(
        self,
        column: str,
        plot_type: str = 'Histogram',
        overlay: bool = True
    ) -> str:
        """
        Plot and compare the distribution of a column in real vs. synthetic data.

        Args:
            column (str): The column to plot.
            plot_type (str): 'Histogram', 'KDE', or 'Boxplot'.
            overlay (bool): Overlay real and synthetic (default: True).

        Returns:
            str: Plotly HTML string.
        """
        real = pd.to_numeric(self.real_data[column], errors='coerce').dropna()
        synth = pd.to_numeric(self.synthetic_data[column], errors='coerce').dropna()

        if plot_type.lower() in ['boxplot', 'box']:
            df_real = self.real_data[[column]].assign(Source="Real")
            df_synth = self.synthetic_data[[column]].assign(Source="Synthetic")
            df = pd.concat([df_real, df_synth], ignore_index=True)
            fig = px.box(df, x="Source", y=column, color="Source", title=f"Boxplot: {column}")
        elif plot_type.lower() == "histogram":
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real, name="Real", opacity=0.6))
            fig.add_trace(go.Histogram(x=synth, name="Synthetic", opacity=0.6))
            fig.update_layout(barmode='overlay', title=f"Histogram: {column}")
        elif plot_type.lower() == "kde":
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real, name="Real", opacity=0.6, histnorm='probability density'))
            fig.add_trace(go.Histogram(x=synth, name="Synthetic", opacity=0.6, histnorm='probability density'))
            fig.update_layout(barmode='overlay', title=f"KDE (Density): {column}")
        else:
            fig = get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=column,
                metadata=self.metadata
            )
        return fig.to_html(full_html=False)

    def pair_plot(
        self,
        columns: List[str],
        overlay: bool = True
    ) -> str:
        """
        Plot a pairwise comparison for specified columns.

        Args:
            columns (List[str]): Columns to compare.
            overlay (bool): Overlay real and synthetic (default: True).

        Returns:
            str: Plotly HTML string.
        """
        fig = get_column_pair_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            column_names=columns,
            metadata=self.metadata
        )
        return fig.to_html(full_html=False)

    def real_vs_synth_summary(
        self,
        columns: List[str]
    ) -> str:
        """
        Generate a summary table and distribution plots for real vs. synthetic data.

        Args:
            columns (List[str]): Columns to summarize.

        Returns:
            str: HTML with table and plots.
        """
        html = (
            "<h4>Real vs. Synthetic Comparison</h4>"
            "<table border=1 style='border-collapse:collapse;'>"
            "<tr><th>Column</th><th>Mean (Real)</th><th>Mean (Synthetic)</th>"
            "<th>Std (Real)</th><th>Std (Synthetic)</th><th>KS-Stat</th></tr>"
        )

        for col in columns:
            real = pd.to_numeric(self.real_data[col], errors='coerce').dropna()
            synth = pd.to_numeric(self.synthetic_data[col], errors='coerce').dropna()
            if len(real) > 0 and len(synth) > 0:
                ks_stat = stats.ks_2samp(real, synth).statistic
                mean_real = real.mean()
                mean_synth = synth.mean()
                std_real = real.std()
                std_synth = synth.std()
            else:
                ks_stat = float('nan')
                mean_real = mean_synth = std_real = std_synth = float('nan')
            html += (
                f"<tr>"
                f"<td>{col}</td>"
                f"<td>{mean_real:.4f}</td>"
                f"<td>{mean_synth:.4f}</td>"
                f"<td>{std_real:.4f}</td>"
                f"<td>{std_synth:.4f}</td>"
                f"<td>{ks_stat:.4f}</td>"
                f"</tr>"
            )
        html += "</table><br>"

        # Add distribution plots for each column
        for col in columns:
            fig = get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=col,
                metadata=self.metadata
            )
            html += f"<h5>Distribution: {col}</h5>" + fig.to_html(full_html=False)
        return html

    def drift_detection(
        self,
        columns: List[str]
    ) -> str:
        """
        Perform drift detection using KS-statistic and visualize for each column.

        Args:
            columns (List[str]): Columns to check for drift.

        Returns:
            str: HTML summary with drift plots.
        """
        html = "<h4>Drift Detection</h4>"

        def is_plottable_column(metadata, column):
            try:
                col_meta = metadata['fields'][column]
                return col_meta.get('sdtype', None) not in ['id', 'identifier']
            except Exception:
                return True

        for col in columns:
            if not is_plottable_column(self.metadata, col):
                html += f"<b>{col}</b>: Not a plottable column (ID type).<br>"
                continue

            real = pd.to_numeric(self.real_data[col], errors='coerce').dropna()
            synth = pd.to_numeric(self.synthetic_data[col], errors='coerce').dropna()
            if len(real) > 0 and len(synth) > 0:
                ks_stat, ks_p = stats.ks_2samp(real, synth)
                drift_flag = ks_stat > 0.2
                drift_html = f"<span style='color:red;'>(Drift)</span>" if drift_flag else ""
                html += (
                    f"<b>{col}</b>: "
                    f"KS-stat = {ks_stat:.4f}, p-value = {ks_p:.4f} {drift_html}<br>"
                )
            else:
                html += f"<b>{col}</b>: Not enough data for drift analysis.<br>"
            fig = get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=col,
                metadata=self.metadata
            )
            html += fig.to_html(full_html=False)
        return html

    def correlation_heatmap(self, show_type: str = "Compare Both") -> str:
        """
        Visualize correlation matrices for real, synthetic, or both datasets.

        Args:
            show_type (str): "Real Data", "Synthetic Data", or "Compare Both".

        Returns:
            str: HTML string with correlation heatmaps.
        """
        html = ""
        if show_type == "Real Data" or show_type == "Compare Both":
            df_real_corr = self.numeric_only(self.real_data).corr()
            fig_real = px.imshow(
                df_real_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Real Data Correlation"
            )
            html += "<h5>Real Data Correlation</h5>" + fig_real.to_html(full_html=False)

        if show_type == "Synthetic Data" or show_type == "Compare Both":
            df_synth_corr = self.numeric_only(self.synthetic_data).corr()
            fig_synth = px.imshow(
                df_synth_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Synthetic Data Correlation"
            )
            html += "<h5>Synthetic Data Correlation</h5>" + fig_synth.to_html(full_html=False)
            
        if show_type == "Compare Both":
            real_corr = self.numeric_only(self.real_data).corr()
            synth_corr = self.numeric_only(self.synthetic_data).corr()
            common_cols = real_corr.columns.intersection(synth_corr.columns)
            diff = (real_corr.loc[common_cols, common_cols] - synth_corr.loc[common_cols, common_cols]).abs()
            fig_diff = px.imshow(
                diff,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Reds",
                title="Correlation Difference (Real vs. Synthetic)"
            )
            html += "<h5>Correlation Difference (Real vs. Synthetic)</h5>" + fig_diff.to_html(full_html=False)
        return html

            