"""Train a LSTM model on all garage datasets."""
import joblib
from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

from api_folder.predictions.config import (
    batch_size,
    garage_A_total_capacity,
    garage_B_total_capacity,
    garage_C_total_capacity,
    garage_D_total_capacity,
    garage_H_total_capacity,
    garage_I_total_capacity,
    garage_Libra_total_capacity,
    learning_rate,
    lists_garages_to_train,
    loss,
    n_features,
    n_steps_in,
    n_steps_out,
    nber_epochs,
    validation_size,
)
from api_folder.predictions.utils import processing_data
from api_folder.predictions.visualize_garages_data import (
    get_garages_data_for_predictions,
    visualize_and_process_garage,
)


def split_sequence(sequence, n_steps_in, n_steps_out):
    """Create the dataset with n_steps_in values as input data and n_steps_out values as output for the prediction."""
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def define_model():
    """Define the LSTM model."""
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(
                250, activation="relu", return_sequences=True, input_shape=(n_steps_in, n_features)
            )
        )
    )
    model.add(LSTM(100, activation="relu"))
    model.add(Dense(n_steps_out))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def main():
    """Train the LSTM model on the garages data."""
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

    for garage in lists_garages_to_train:  # Loop over each garage to train them all.
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

        #   Process data in the correct format for training
        (
            garage_time_series_dates_processed,
            garage_time_series_spaces_available_processed,
        ) = processing_data(
            garage_time_series_dates_processed,
            garage_time_series_spaces_available_processed,
        )

        # Define normalization scaler for training, fit on the data and save it for prediction later.
        sc = MinMaxScaler(feature_range=(0, 1))
        training_data = sc.fit_transform(garage_time_series_spaces_available_processed)

        # split into samples according to config parameters
        X, y = split_sequence(training_data, n_steps_in, n_steps_out)

        # Reshape data for training LSTM model
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        print("Dataset Input shape after reshape", X.shape)

        # Define model, train it on the data and save it for prediction later.
        compiled_model = define_model()
        compiled_model.fit(
            X,
            y,
            validation_split=validation_size,
            epochs=nber_epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
        )

        # Save the model only if performance is better than the previous one.
        # TODO ADD PERFORMANCE CHECK WITH JSON FILE to make sure nan values are not there in the model data ; CAN ALSO CHECK PERFORAMNCE WITH JSON FILE INSIDE GITHUB
        # TODO CREATE ENDPOINT FOR PREDICTION DATA (MODELS ARE READY AND RETRAINED)
        compiled_model.save(f"../output_dir_models/{garage}_model.h5")
        scaler_filename = f"../output_dir_models/{garage}_min_max_scaler.h5"
        joblib.dump(sc, scaler_filename)

        print(f"Model and scaler for garage {garage} have been saved successfully.")


if __name__ == "__main__":
    main()
