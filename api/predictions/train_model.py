from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Flatten, ConvLSTM2D, Bidirectional, RepeatVector, TimeDistributed
from keras.layers import Dense
import tensorflow
from visualize_garages_data import get_garages_data_for_predictions, visualize_and_process_garage_A, visualize_and_process_garage_B, visualize_and_process_garage_Libra
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error


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
garage_A_time_series_dates_processed, garage_A_time_series_spaces_available_processed = visualize_and_process_garage_Libra(garage_Libra_time_series_dates,garage_Libra_time_series_spaces_available, False )


#Reverse the list to have data in correct date times
garage_A_time_series_dates_processed = list(reversed(garage_A_time_series_dates_processed))
garage_A_time_series_spaces_available_processed = list(reversed(garage_A_time_series_spaces_available_processed))








# change data to correct format + define input sequence

garage_A_time_series_dates_processed = np.array(garage_A_time_series_dates_processed)
garage_A_time_series_spaces_available_processed = np.array(garage_A_time_series_spaces_available_processed)

garage_A_time_series_spaces_available_processed = garage_A_time_series_spaces_available_processed.reshape(-1, 1)



sc = MinMaxScaler(feature_range=(0, 1))
training_data = sc.fit_transform(garage_A_time_series_spaces_available_processed)

# choose a number of time steps
n_steps_in, n_steps_out = 6, 6
# split into samples
X, y = split_sequence(training_data, n_steps_in, n_steps_out)



train_size = int(len(y) * 0.90)
test_size = len(y) - train_size
n_features = 1


X = X.reshape((X.shape[0], X.shape[1], n_features))

print('Dataset shape', X.shape)


train_X = np.array(X[0:train_size])
test_X = np.array(X[train_size:len(X)])

train_y = np.array(y[0:train_size])
test_y = np.array(y[train_size:len(X)])


print('train and test shapes', train_X.shape, test_X.shape)


#Define model

model = Sequential()
model.add(Bidirectional(LSTM(25, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features))))
model.add(LSTM(10, activation='relu'))
model.add(Dense(n_steps_out))
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer,loss='mse')
model.fit(train_X, train_y, epochs=25, batch_size=64, verbose=1, shuffle=True)
# for i in range(2):
# 	model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
# 	model.reset_states()




train_y = train_y.squeeze()
test_y = test_y.squeeze()

# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)
print(trainPredict.shape)# invert predictions
trainPredict = sc.inverse_transform(trainPredict)
trainY = sc.inverse_transform(train_y)
testPredict = sc.inverse_transform(testPredict)
testY = sc.inverse_transform(test_y)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
training_data = X.squeeze()
trainPredictPlot = np.empty_like(training_data)
trainPredictPlot[:, :] = np.nan
print(trainPredictPlot.shape)
trainPredictPlot[6:len(trainPredict)+6, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(training_data)
testPredictPlot[:, :] = np.nan
print(testPredictPlot.shape)
print(testPredict.shape)
print(len(trainPredict)+(6*2)+1,len(training_data)-1, len(training_data)-1  - len(trainPredict)+(6*2)+1 )
testPredictPlot[len(trainPredict)-1:len(training_data)-1, :] = testPredict
# plot baseline and predictions
plt.plot(sc.inverse_transform(training_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


#######################################################
# code to forecast : should be same as before for testing




# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, len(X), 1 )
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]



def make_forecasts(model, n_batch):
	forecasts = list()
	for i in range(len(test_X)):
		X, y = test_X[i,:], test_y[i,:],
		#print(X)
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts


# make forecasts on the entire test set.
forecasts = make_forecasts(model, 1)
print('forecasts',forecasts)


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

forecasts = inverse_transform( forecasts, sc)
print('forecasts',forecasts)


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(forecasts, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test_X]
		actual = inverse_transform( actual, sc)
		predicted = [forecast[i] for forecast in forecasts]
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

results = evaluate_forecasts(forecasts, 6)
print(results)

# # plot the forecasts in the context of the original dataset
# def plot_forecasts(series, forecasts, n_test):
# 	# plot the entire dataset in blue
# 	plt.plot(series.values)
# 	# plot the forecasts in red
# 	for i in range(len(forecasts)):
# 		off_s = len(series) - n_test + i - 1
# 		off_e = off_s + len(forecasts[i]) + 1
# 		xaxis = [x for x in range(off_s, off_e)]
# 		yaxis = [series.values[off_s]] + forecasts[i]
# 		plt.plot(xaxis, yaxis, color='red')
# 	# show the plot
# 	plt.show()





