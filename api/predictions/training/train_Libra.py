from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense
from keras.models import load_model
import tensorflow
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

#Create the dataset with n_steps_in values as input data and n_steps_out values as output for the prediction !
from api.predictions.visualize_garages_data import visualize_and_process_garage, get_garages_data_for_predictions

garage_Libra_total_capacity = 1007
Training = True

def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
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



#Get data from db and process Libra garage data
garage_A_time_series_dates, garage_B_time_series_dates, garage_C_time_series_dates, garage_D_time_series_dates, garage_H_time_series_dates, garage_I_time_series_dates, garage_Libra_time_series_dates, garage_A_time_series_spaces_available, garage_B_time_series_spaces_available, garage_C_time_series_spaces_available, garage_D_time_series_spaces_available, garage_H_time_series_spaces_available, garage_I_time_series_spaces_available, garage_Libra_time_series_spaces_available = get_garages_data_for_predictions()
garage_Libra_time_series_dates_processed, garage_Libra_time_series_spaces_available_processed = visualize_and_process_garage(
    garage_Libra_time_series_dates, garage_Libra_time_series_spaces_available, False, 'Libra',
    garage_Libra_total_capacity)



# change data to correct format + define input sequence

garage_Libra_time_series_dates_processed = np.array(garage_Libra_time_series_dates_processed)
garage_Libra_time_series_spaces_available_processed = np.array(garage_Libra_time_series_spaces_available_processed)

garage_Libra_time_series_spaces_available_processed = garage_Libra_time_series_spaces_available_processed.reshape(-1, 1)



sc = MinMaxScaler(feature_range=(0, 1))
print(garage_Libra_time_series_spaces_available_processed)
training_data = sc.fit_transform(garage_Libra_time_series_spaces_available_processed)

# choose a number of time steps
n_steps_in, n_steps_out = 6, 6
# split into samples
X, y = split_sequence(training_data, n_steps_in, n_steps_out)


train_size = int(len(y) * 0.90)
test_size = len(y) - train_size
n_features = 1

print('Dataset Input shape', X.shape)

X = X.reshape((X.shape[0], X.shape[1], n_features))

print('Dataset Input shape', X.shape)


train_X = np.array(X[0:train_size])
test_X = np.array(X[train_size:len(X)])

train_y = np.array(y[0:train_size])
test_y = np.array(y[train_size:len(X)])


print('train and test input shapes', train_X.shape, test_X.shape)
print('train and test output shapes', train_y.shape, test_y.shape)


#Define model
model = Sequential()
model.add(Bidirectional(LSTM(250, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features))))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.05)
model.compile(optimizer=optimizer,loss='mse')

if Training:
	model.fit(train_X, train_y, epochs=15, batch_size=32, verbose=1, shuffle=True)

train_y = train_y.squeeze()
test_y = test_y.squeeze()
print('train and test output shapes after squeezing', train_y.shape, test_y.shape)



# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, len(X), 1 )
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]



def make_forecasts(model, input_dataset, output_dataset , n_batch):
	forecasts = list()
	for i in range(len(input_dataset)):
		X, y = input_dataset[i,:], output_dataset[i,:],

		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts


# make forecasts on the entire test set.
#forecasts = make_forecasts(model, test_X, test_y,  1)
#print('Scaled forecasts',len(forecasts))
#print('Scaled test_y',len(test_y))


def inverse_transform( forecasts, scaler):
	inverted = list()
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
#
#forecasts = inverse_transform( forecasts, sc)
#print('Not Scaled forecasts',len(forecasts))
#

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(forecasts, input_dataset, n_seq):

	for i in range(n_seq):
		actual = [input_dataset[i]] #[row[i] for row in input_dataset]
		actual = inverse_transform( actual, sc)
		#print('actual',actual)
		predicted = [forecasts[i]]
		#print('predicted', predicted)
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))


#evaluate_forecasts(forecasts, test_y, n_steps_out)
if Training:
	model.save('../output_dir_models/Libra_model.h5')
#PREDICT
model_loaded = load_model('../output_dir_models/Libra_model.h5')
model_loaded.summary()


assert training_data[1000]== sc.transform(garage_Libra_time_series_spaces_available_processed[1000].reshape(-1, 1))

def predict_next_three_days():
	predictions = []
	for index in range(4*3):
		if index == 0:
			next_six_hours_processed = sc.transform(garage_Libra_time_series_spaces_available_processed[-6:].reshape(-1, 1))
		else:
			next_six_hours_processed = sc.transform(np.array(predictions[-6:])[0].reshape(-1, 1))
			print('ici', next_six_hours_processed)


		new_next_six_hours_processed = next_six_hours_processed.reshape((next_six_hours_processed.shape[0], next_six_hours_processed.shape[1], n_features))
		print(new_next_six_hours_processed.shape)
		result = model_loaded.predict(new_next_six_hours_processed.reshape(1,6,1))

		print('PREDICTION ', inverse_transform( result, sc))
		predictions.append(list(inverse_transform( result, sc)))

	return predictions


predictions = predict_next_three_days()
print(predictions)