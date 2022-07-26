"""Train a LSTM model on the Libra dataset."""
import joblib
import tensorflow
from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from numpy import array
from sklearn.preprocessing import MinMaxScaler

from api.predictions.config import (
    batch_size,
    garage_Libra_total_capacity,
    n_features,
    n_steps_in,
    n_steps_out,
    nber_epochs,
)
from api.predictions.utils import processing_data
from api.predictions.visualize_garages_data import (
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
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def main():
    """Train the LSTM model on the Libra dataset."""
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
    (
        garage_Libra_time_series_dates_processed,
        garage_Libra_time_series_spaces_available_processed,
    ) = visualize_and_process_garage(
        garage_Libra_time_series_dates,
        garage_Libra_time_series_spaces_available,
        False,
        "Libra",
        garage_Libra_total_capacity,
    )

    #   Process data in the correct format for training
    (
        garage_Libra_time_series_dates_processed,
        garage_Libra_time_series_spaces_available_processed,
    ) = processing_data(
        garage_Libra_time_series_dates_processed,
        garage_Libra_time_series_spaces_available_processed,
    )

    # Define normalization scaler for training, fit on the data and save it for prediction later.
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data = sc.fit_transform(garage_Libra_time_series_spaces_available_processed)
    scaler_filename = "../output_dir_models/Libra_min_max_scaler.h5"
    joblib.dump(sc, scaler_filename)

    # split into samples according to config parameters
    X, y = split_sequence(training_data, n_steps_in, n_steps_out)

    # Reshape data for training LSTM model
    # train_size = int(len(y) * training_size)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    print("Dataset Input shape after reshape", X.shape)

    # # Define training and test data
    # train_X = np.array(X[0:train_size])
    # test_X = np.array(X[train_size : len(X)])
    # train_y = np.array(y[0:train_size])
    # test_y = np.array(y[train_size : len(X)])
    # print("train and test input shapes", train_X.shape, test_X.shape)
    # print("train and test output shapes", train_y.shape, test_y.shape)

    # Define model, train it on the data and save it for prediction later.
    compiled_model = define_model()
    compiled_model.fit(
        X,
        y,
        validation_split=0.1,
        epochs=nber_epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
    )
    # train_y = train_y.squeeze()
    # test_y = test_y.squeeze()
    # print("train and test output shapes after squeezing", train_y.shape, test_y.shape)
    compiled_model.save("../output_dir_models/Libra_model.h5")

    assert training_data[1000] == sc.transform(
        garage_Libra_time_series_spaces_available_processed[1000].reshape(-1, 1)
    )

    print("Model saved")


if __name__ == "__main__":
    main()
