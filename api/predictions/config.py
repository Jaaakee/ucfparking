"""Configuration file setting the total capacity of each garage."""
garage_A_total_capacity = 1623
garage_B_total_capacity = 1259
garage_C_total_capacity = 1852
garage_D_total_capacity = 1241
garage_H_total_capacity = 1284
garage_I_total_capacity = 1231
garage_Libra_total_capacity = 1007

n_features = 1
validation_size = 0.2

n_steps_in = 24
n_steps_out = 24
nber_epochs = 1
batch_size = 32
learning_rate = 0.01  # 0.05
loss = "mse"
lists_garages_to_train = ["A"]  # , "B", "C", "D", "H", "I", "Libra"]
number_of_hours_to_predict = 24
prediction_showing = True
