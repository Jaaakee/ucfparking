"""Predict the next 3 days of available spaces for all garages."""
from datetime import timedelta

import joblib
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from numpy import array

from api.predictions.config import (
    garage_A_total_capacity,
    garage_B_total_capacity,
    garage_C_total_capacity,
    garage_D_total_capacity,
    garage_H_total_capacity,
    garage_I_total_capacity,
    garage_Libra_total_capacity,
    lists_garages_to_train,
    n_features,
    n_steps_in,
    n_steps_out,
    number_of_hours_to_predict,
)
from api.predictions.utils import processing_data
from api.predictions.visualize_garages_data import (
    get_garages_data_for_predictions,
    visualize_and_process_garage,
)


def inverse_transform(forecasts, scaler):
    """Inverse transform with the scaler to come back to normal size."""
    inverted = []
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]

        # store
        inverted.append(inv_scale)
    return inverted


def predict_next_three_days(model, scaler, data):
    """Predict the next three days."""
    predictions = []
    for index in range(number_of_hours_to_predict // n_steps_out):
        if index == 0:
            next_hours_processed = scaler.transform(data[-n_steps_in:].reshape(-1, 1))
        else:
            len_predictions = len(predictions)
            if len_predictions >= n_steps_in:
                next_hours_processed = scaler.transform(
                    np.array(predictions[-n_steps_in:]).reshape(-1, 1)
                )
            else:
                needed_input_data_points = n_steps_in - len_predictions
                input_data_part_list = list(data[-needed_input_data_points:].squeeze())
                predictions_part_list = predictions
                input_data_part_list.extend(predictions_part_list)
                next_hours_processed = scaler.transform(
                    np.array(input_data_part_list).reshape(-1, 1)
                )

        new_next_hours_processed = next_hours_processed.reshape(
            (next_hours_processed.shape[0], next_hours_processed.shape[1], n_features)
        )

        result = model.predict(new_next_hours_processed.reshape(1, n_steps_in, 1))
        prediction_list = list(inverse_transform(result, scaler))[0]
        predictions = [y for x in [predictions, prediction_list] for y in x]

    return predictions


def main(prediction_showing):
    """Get data from db and process Libra garage data."""
    (
        garage_A_time_series_dates,
        garage_B_time_series_dates,
        garage_C_time_series_dates,
        garage_D_time_series_dates,
        garage_H_time_series_dates,
        garage_I_time_series_dates,
        garage_Libra_time_series_dates,
        garage_A_time_series_spaces_available,
        garage_B_time_series_spaces_available,
        garage_C_time_series_spaces_available,
        garage_D_time_series_spaces_available,
        garage_H_time_series_spaces_available,
        garage_I_time_series_spaces_available,
        garage_Libra_time_series_spaces_available,
    ) = get_garages_data_for_predictions()

    next_day_predictions = {}
    main_garage_dictionary = {
        "A": {
            "capacity": garage_A_total_capacity,
            "time_series_dates": garage_A_time_series_dates,
            "time_series_spaces_available": garage_A_time_series_spaces_available,
        },
        "B": {
            "capacity": garage_B_total_capacity,
            "time_series_dates": garage_B_time_series_dates,
            "time_series_spaces_available": garage_B_time_series_spaces_available,
        },
        "C": {
            "capacity": garage_C_total_capacity,
            "time_series_dates": garage_C_time_series_dates,
            "time_series_spaces_available": garage_C_time_series_spaces_available,
        },
        "D": {
            "capacity": garage_D_total_capacity,
            "time_series_dates": garage_D_time_series_dates,
            "time_series_spaces_available": garage_D_time_series_spaces_available,
        },
        "H": {
            "capacity": garage_H_total_capacity,
            "time_series_dates": garage_H_time_series_dates,
            "time_series_spaces_available": garage_H_time_series_spaces_available,
        },
        "I": {
            "capacity": garage_I_total_capacity,
            "time_series_dates": garage_I_time_series_dates,
            "time_series_spaces_available": garage_I_time_series_spaces_available,
        },
        "Libra": {
            "capacity": garage_Libra_total_capacity,
            "time_series_dates": garage_Libra_time_series_dates,
            "time_series_spaces_available": garage_Libra_time_series_spaces_available,
        },
    }

    for garage in lists_garages_to_train:
        (
            garage_time_series_dates_processed,
            garage_time_series_spaces_available_processed,
        ) = visualize_and_process_garage(
            main_garage_dictionary[garage]["time_series_dates"],
            main_garage_dictionary[garage]["time_series_spaces_available"],
            False,
            garage,
            main_garage_dictionary[garage]["capacity"],
        )

        # Process data in the correct format for training
        (
            garage_time_series_dates_processed,
            garage_time_series_spaces_available_processed,
        ) = processing_data(
            garage_time_series_dates_processed,
            garage_time_series_spaces_available_processed,
        )

        # Load model for predictions
        model = load_model(f"api/predictions/output_dir_models/{garage}_model.h5")
        scaler = joblib.load(f"api/predictions/output_dir_models/{garage}_min_max_scaler.h5")

        predictions = predict_next_three_days(
            model, scaler, garage_time_series_spaces_available_processed
        )

        reformatted_predictions = predictions
        for index, prediction_point in enumerate(reformatted_predictions):

            if reformatted_predictions[index] > main_garage_dictionary[garage]["capacity"]:
                reformatted_predictions[index] = main_garage_dictionary[garage]["capacity"]
            reformatted_predictions[index] = int(reformatted_predictions[index])

        times_corresponding_to_predictions = []
        for index in range(number_of_hours_to_predict):
            if index == 0:
                given_time = garage_time_series_dates_processed[-1]
            else:
                given_time = times_corresponding_to_predictions[-1]
            final_time = given_time + timedelta(hours=1)
            times_corresponding_to_predictions.append(final_time)
            if str(final_time) not in next_day_predictions:
                next_day_predictions[str(final_time)] = {}
            next_day_predictions[str(final_time)][garage] = reformatted_predictions[index]

        if prediction_showing:
            plt.plot(
                garage_time_series_dates_processed,
                garage_time_series_spaces_available_processed,
                "-r",
            )
            plt.plot(times_corresponding_to_predictions, reformatted_predictions, "-b")
            plt.xlabel("Time")
            plt.ylabel("Available parking spaces")
            plt.legend(["Known", "Predicted"])
            plt.show()

    return next_day_predictions


if __name__ == "__main__":
    next_day_predictions = main(prediction_showing=True)
