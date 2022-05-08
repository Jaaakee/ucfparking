#Visualize Available Space of Garage A.
from utils.get_env_var import *
import mysql.connector
from utils.query_database import *
import matplotlib.pyplot as plt
import numpy as np
import datetime


#TODO Script that fills missing data with average between bounded dates.

my_database = mysql.connector.connect(
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASS'],
    database=os.environ['DB_NAME']
)

my_cursor = my_database.cursor()
object_query = DataQueryExtractor(my_database, my_cursor)
content = object_query.get_all_data_for_predictions()
my_cursor.close()
my_database.close()




garage_A_total_capacity = 1623
garage_B_total_capacity = 1259
garage_C_total_capacity = 1852
garage_D_total_capacity = 1241
garage_H_total_capacity = 1284
garage_I_total_capacity = 1231
garage_Libra_total_capacity = 1007



garage_A_time_series_spaces_available = []
garage_A_time_series_dates = []

garage_B_time_series_spaces_available = []
garage_B_time_series_dates = []


garage_C_time_series_spaces_available = []
garage_C_time_series_dates = []


garage_D_time_series_spaces_available = []
garage_D_time_series_dates = []


garage_I_time_series_spaces_available = []
garage_I_time_series_dates = []


garage_H_time_series_spaces_available = []
garage_H_time_series_dates = []


garage_Libra_time_series_spaces_available = []
garage_Libra_time_series_dates = []
for row in content:
    garage_A_time_series_dates.append(row[0])
    garage_B_time_series_dates.append(row[0])
    garage_C_time_series_dates.append(row[0])
    garage_D_time_series_dates.append(row[0])
    garage_H_time_series_dates.append(row[0])
    garage_I_time_series_dates.append(row[0])
    garage_Libra_time_series_dates.append(row[0])

    garage_A_time_series_spaces_available.append(int(float(row[1])))
    garage_B_time_series_spaces_available.append(int(float(row[4])))
    garage_C_time_series_spaces_available.append(int(float(row[7])))
    garage_D_time_series_spaces_available.append(int(float(row[10])))
    garage_H_time_series_spaces_available.append(int(float(row[13])))
    garage_I_time_series_spaces_available.append(int(float(row[16])))
    garage_Libra_time_series_spaces_available.append(int(float(row[19])))



garage_A_time_series_spaces_available = np.array(garage_A_time_series_spaces_available)
garage_A_time_series_spaces_available_processed = np.where(garage_A_time_series_spaces_available>garage_A_total_capacity,garage_A_total_capacity,garage_A_time_series_spaces_available)
index_to_begin_garage_A = garage_A_time_series_dates.index(datetime.datetime(2021, 8, 13, 14, 0))
garage_A_time_series_dates_processed = garage_A_time_series_dates[:index_to_begin_garage_A]
garage_A_time_series_spaces_available_processed = garage_A_time_series_spaces_available_processed[:index_to_begin_garage_A]



plt.plot(garage_A_time_series_dates,garage_A_time_series_spaces_available)
plt.plot(garage_A_time_series_dates_processed,garage_A_time_series_spaces_available_processed)
plt.axhline(y=garage_A_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage A : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageA.png')
plt.show()


garage_B_time_series_spaces_available = np.array(garage_B_time_series_spaces_available)
garage_B_time_series_spaces_available_processed = np.where(garage_B_time_series_spaces_available>garage_B_total_capacity,garage_B_total_capacity,garage_B_time_series_spaces_available)
garage_B_time_series_dates_processed = garage_B_time_series_dates


plt.plot(garage_B_time_series_dates,garage_B_time_series_spaces_available)
plt.plot(garage_B_time_series_dates_processed,garage_B_time_series_spaces_available_processed)
plt.axhline(y=garage_B_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage B : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageB.png')
plt.show()

garage_C_time_series_spaces_available = np.array(garage_C_time_series_spaces_available)
garage_C_time_series_spaces_available_processed = np.where(garage_C_time_series_spaces_available>garage_C_total_capacity,garage_C_total_capacity,garage_C_time_series_spaces_available)
garage_C_time_series_dates_processed = garage_C_time_series_dates


plt.plot(garage_C_time_series_dates,garage_C_time_series_spaces_available)
plt.plot(garage_C_time_series_dates_processed,garage_C_time_series_spaces_available_processed)
plt.axhline(y=garage_C_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage C : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageC.png')
plt.show()

garage_D_time_series_spaces_available = np.array(garage_D_time_series_spaces_available)
garage_D_time_series_spaces_available_processed = np.where(garage_D_time_series_spaces_available>garage_D_total_capacity,garage_D_total_capacity,garage_D_time_series_spaces_available)
garage_D_time_series_dates_processed = garage_D_time_series_dates


plt.plot(garage_D_time_series_dates,garage_D_time_series_spaces_available)
plt.plot(garage_D_time_series_dates_processed,garage_D_time_series_spaces_available_processed)
plt.axhline(y=garage_D_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage D : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageD.png')
plt.show()


garage_H_time_series_spaces_available = np.array(garage_H_time_series_spaces_available)
garage_H_time_series_spaces_available_processed = np.where(garage_H_time_series_spaces_available>garage_H_total_capacity,garage_H_total_capacity,garage_H_time_series_spaces_available)
garage_H_time_series_dates_processed = garage_H_time_series_dates

plt.plot(garage_H_time_series_dates,garage_H_time_series_spaces_available)
plt.plot(garage_H_time_series_dates_processed,garage_H_time_series_spaces_available_processed)
plt.axhline(y=garage_H_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage H : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageH.png')
plt.show()


garage_I_time_series_spaces_available = np.array(garage_I_time_series_spaces_available)
garage_I_time_series_spaces_available_processed = np.where(garage_I_time_series_spaces_available>garage_I_total_capacity,garage_I_total_capacity,garage_I_time_series_spaces_available)
garage_I_time_series_dates_processed = garage_I_time_series_dates

plt.plot(garage_I_time_series_dates,garage_I_time_series_spaces_available)
plt.plot(garage_I_time_series_dates_processed,garage_I_time_series_spaces_available_processed)
plt.axhline(y=garage_I_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage I : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageI.png')
plt.show()

garage_Libra_time_series_spaces_available = np.array(garage_Libra_time_series_spaces_available)
garage_Libra_time_series_spaces_available_processed = np.where(garage_Libra_time_series_spaces_available>garage_Libra_total_capacity,garage_Libra_total_capacity,garage_Libra_time_series_spaces_available)
garage_Libra_time_series_dates_processed = garage_Libra_time_series_dates

plt.plot(garage_Libra_time_series_dates,garage_Libra_time_series_spaces_available)
plt.plot(garage_Libra_time_series_dates_processed,garage_Libra_time_series_spaces_available_processed)
plt.axhline(y=garage_Libra_total_capacity, color='r', linestyle='-')
plt.xlabel('Date (Time series)')
plt.ylabel('Spaces Available')
plt.title('Garage Libra : Availability of spaces in function of the time')
plt.legend(['Raw','Processed'])
plt.savefig('plots/garageLibra.png')
plt.show()





# train_output = garage_Libra_time_series_spaces_available_processed[0:len(garage_Libra_time_series_spaces_available_processed) - 1000]
# test_output = garage_Libra_time_series_spaces_available_processed[:1000]
# train_date = garage_Libra_time_series_dates_processed[0:len(garage_Libra_time_series_spaces_available_processed) - 1000]
# test_date = garage_Libra_time_series_dates_processed[:1000]
#
#
# plt.plot(train_date, train_output, color = "black")
# plt.plot(test_date, test_output, color = "red")
# plt.ylabel('Spaces Availability Libra')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.title("Train/Test split for Space Availibility Libra Data")
# plt.show()

