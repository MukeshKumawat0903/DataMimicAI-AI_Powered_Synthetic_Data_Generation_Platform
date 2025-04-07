from sdv.evaluation.single_table import get_column_plot, get_column_pair_plot
import plotly.io as pio

pio.templates.default = "plotly_white"

# def generate_visualizations(real_data, synthetic_data, metadata):
#     """Return visualization HTML"""
#     return {
#         'column_plot': get_column_plot(
#             real_data=real_data,
#             synthetic_data=synthetic_data,
#             # column_name=metadata.columns[0],  # First column
#             column_name='capacity_mw',
#             metadata=metadata
#         ).to_html(),
#         'pair_plot': get_column_pair_plot(
#             real_data=real_data,
#             synthetic_data=synthetic_data,
#             column_names=['capacity_mw', 'primary_fuel'],
#             metadata=metadata
#         ).to_html()
#     }

# Update your visualization generation function
def generate_visualizations(real_data, synthetic_data, metadata, column_name=None, pair_columns=None):
    """Return visualization HTML with configurable columns"""
    # Set defaults if not provided
    if not column_name:
        column_name = real_data.columns[0]  # First column as default
    
    if not pair_columns:
        pair_columns = real_data.columns[:2].tolist()  # First two columns
    
    return {
        'column_plot': get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name=column_name,
            metadata=metadata
        ).to_html(),
        'pair_plot': get_column_pair_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_names=pair_columns,
            metadata=metadata
        ).to_html()
    }
