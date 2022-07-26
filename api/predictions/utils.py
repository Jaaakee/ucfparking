"""Utils function for training/prediction."""
import numpy as np


def processing_data(time_series_dates, time_series_spaces_available):
    """Change data to correct format + define input sequence."""
    garage_Libra_time_series_dates_processed = np.array(time_series_dates)
    garage_Libra_time_series_spaces_available_processed = np.array(time_series_spaces_available)
    garage_Libra_time_series_spaces_available_processed = (
        garage_Libra_time_series_spaces_available_processed.reshape(-1, 1)
    )

    return (
        garage_Libra_time_series_dates_processed,
        garage_Libra_time_series_spaces_available_processed,
    )
