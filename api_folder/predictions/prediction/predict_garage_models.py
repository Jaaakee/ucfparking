"""Predict the next 3 days of available spaces for all garages."""
import sys
from datetime import timedelta

import joblib
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from numpy import array

from api_folder.predictions.config import (
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
    prediction_showing,
)
from api_folder.predictions.utils import processing_data
from api_folder.predictions.visualize_garages_data import (
    get_garages_data_for_predictions,
    visualize_and_process_garage,
)

sys.path.append("./")

# def forecast_lstm(model, X, n_batch):
#     """Make one forecast with an LSTM."""
#     # Reshape input pattern to [samples, timesteps, features]
#     X = X.reshape(1, len(X), 1)
#     # make forecast
#     forecast = model.predict(X, batch_size=n_batch)
#     # convert to array
#     return [x for x in forecast[0, :]]
#
#
# def make_forecasts(model, input_dataset, output_dataset, n_batch):
#     """Make forecasts."""
#     forecasts = []
#     for i in range(len(input_dataset)):
#         X, _ = (
#             input_dataset[i, :],
#             output_dataset[i, :],
#         )
#
#         # make forecast
#         forecast = forecast_lstm(model, X, n_batch)
#         # store the forecast
#         forecasts.append(forecast)
#     return forecasts


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


# def evaluate_forecasts(forecasts, input_dataset, n_seq, scaler):
#     """Evaluate the RMSE for each forecast time step."""
#     for i in range(n_seq):
#         actual = [input_dataset[i]]  # [row[i] for row in input_dataset]
#         actual = inverse_transform(actual, scaler)
#         # print('actual',actual)
#         predicted = [forecasts[i]]
#         # print('predicted', predicted)
#         rmse = math.sqrt(mean_squared_error(actual, predicted))
#         print("t+%d RMSE: %f" % ((i + 1), rmse))


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

        print("Used for prediction at index", index, new_next_hours_processed)
        result = model.predict(new_next_hours_processed.reshape(1, n_steps_in, 1))
        prediction_list = list(inverse_transform(result, scaler))[0]
        predictions = [y for x in [predictions, prediction_list] for y in x]
        print(index, predictions)

    return predictions


def main():
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
        model = load_model(f"../output_dir_models/{garage}_model.h5")
        model.summary()
        scaler = joblib.load(f"../output_dir_models/{garage}_min_max_scaler.h5")

        predictions = predict_next_three_days(
            model, scaler, garage_time_series_spaces_available_processed
        )

        reformatted_predictions = predictions
        for index, prediction_point in enumerate(reformatted_predictions):
            if reformatted_predictions[index] > main_garage_dictionary[garage]["capacity"]:
                reformatted_predictions[index] = main_garage_dictionary[garage]["capacity"]

        print(f"{number_of_hours_to_predict} prediction data values", reformatted_predictions)
        times_corresponding_to_predictions = []

        for index in range(number_of_hours_to_predict):
            if index == 0:
                given_time = garage_time_series_dates_processed[-1]
            else:
                given_time = times_corresponding_to_predictions[-1]
            final_time = given_time + timedelta(hours=1)
            times_corresponding_to_predictions.append(final_time)

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


if __name__ == "__main__":
    main()
