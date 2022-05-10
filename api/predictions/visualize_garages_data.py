from utils.get_env_var import *
import mysql.connector
from utils.query_database import *
import matplotlib.pyplot as plt
import numpy as np
import datetime



def get_garages_data_for_predictions():

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

    return garage_A_time_series_dates,garage_B_time_series_dates,garage_C_time_series_dates,garage_D_time_series_dates,garage_H_time_series_dates,garage_I_time_series_dates,garage_Libra_time_series_dates, garage_A_time_series_spaces_available , garage_B_time_series_spaces_available , garage_C_time_series_spaces_available , garage_D_time_series_spaces_available , garage_H_time_series_spaces_available , garage_I_time_series_spaces_available ,garage_Libra_time_series_spaces_available



#Variables known in advance
garage_A_total_capacity = 1623
garage_B_total_capacity = 1259
garage_C_total_capacity = 1852
garage_D_total_capacity = 1241
garage_H_total_capacity = 1284
garage_I_total_capacity = 1231
garage_Libra_total_capacity = 1007



def visualize_and_process_garage_A(garage_A_time_series_dates,garage_A_time_series_spaces_available, showing ):

    garage_A_time_series_spaces_available = np.array(garage_A_time_series_spaces_available)
    garage_A_time_series_spaces_available_processed = np.where(garage_A_time_series_spaces_available>garage_A_total_capacity,garage_A_total_capacity,garage_A_time_series_spaces_available)
    index_to_begin_garage_A = garage_A_time_series_dates.index(datetime.datetime(2021, 8, 13, 14, 0))
    garage_A_time_series_dates_processed = garage_A_time_series_dates[:index_to_begin_garage_A]
    garage_A_time_series_spaces_available_processed = garage_A_time_series_spaces_available_processed[:index_to_begin_garage_A]


    if showing:
        plt.plot(garage_A_time_series_dates,garage_A_time_series_spaces_available)
        plt.plot(garage_A_time_series_dates_processed,garage_A_time_series_spaces_available_processed)
        plt.axhline(y=garage_A_total_capacity, color='r', linestyle='--')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage A : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageA.png')
        plt.show()

    return garage_A_time_series_dates_processed, garage_A_time_series_spaces_available_processed




def visualize_and_process_garage_B(garage_B_time_series_dates,garage_B_time_series_spaces_available, showing ):

    garage_B_time_series_spaces_available = np.array(garage_B_time_series_spaces_available)
    garage_B_time_series_spaces_available_processed = np.where(garage_B_time_series_spaces_available>garage_B_total_capacity,garage_B_total_capacity,garage_B_time_series_spaces_available)
    garage_B_time_series_dates_processed = garage_B_time_series_dates

    if showing:
        plt.plot(garage_B_time_series_dates,garage_B_time_series_spaces_available)
        plt.plot(garage_B_time_series_dates_processed,garage_B_time_series_spaces_available_processed)
        plt.axhline(y=garage_B_total_capacity, color='r', linestyle='--')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage B : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageB.png')
        plt.show()

    return garage_B_time_series_dates_processed, garage_B_time_series_spaces_available_processed


def visualize_and_process_garage_C(garage_C_time_series_dates,garage_C_time_series_spaces_available, showing ):

    garage_C_time_series_spaces_available = np.array(garage_C_time_series_spaces_available)
    garage_C_time_series_spaces_available_processed = np.where(garage_C_time_series_spaces_available>garage_C_total_capacity,garage_C_total_capacity,garage_C_time_series_spaces_available)
    garage_C_time_series_dates_processed = garage_C_time_series_dates

    if showing:
        plt.plot(garage_C_time_series_dates,garage_C_time_series_spaces_available)
        plt.plot(garage_C_time_series_dates_processed,garage_C_time_series_spaces_available_processed)
        plt.axhline(y=garage_C_total_capacity, color='r', linestyle='-')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage C : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageC.png')
        plt.show()

    return garage_C_time_series_dates_processed, garage_C_time_series_spaces_available_processed



def visualize_and_process_garage_D(garage_D_time_series_dates,garage_D_time_series_spaces_available, showing ):

    garage_D_time_series_spaces_available = np.array(garage_D_time_series_spaces_available)
    garage_D_time_series_spaces_available_processed = np.where(garage_D_time_series_spaces_available>garage_D_total_capacity,garage_D_total_capacity,garage_D_time_series_spaces_available)
    garage_D_time_series_dates_processed = garage_D_time_series_dates

    if showing:
        plt.plot(garage_D_time_series_dates,garage_D_time_series_spaces_available)
        plt.plot(garage_D_time_series_dates_processed,garage_D_time_series_spaces_available_processed)
        plt.axhline(y=garage_D_total_capacity, color='r', linestyle='-')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage D : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageD.png')
        plt.show()

    return garage_D_time_series_dates_processed, garage_D_time_series_spaces_available_processed



def visualize_and_process_garage_H(garage_H_time_series_dates,garage_H_time_series_spaces_available, showing ):

    garage_H_time_series_spaces_available = np.array(garage_H_time_series_spaces_available)
    garage_H_time_series_spaces_available_processed = np.where(garage_H_time_series_spaces_available>garage_H_total_capacity,garage_H_total_capacity,garage_H_time_series_spaces_available)
    garage_H_time_series_dates_processed = garage_H_time_series_dates

    if showing:
        plt.plot(garage_H_time_series_dates,garage_H_time_series_spaces_available)
        plt.plot(garage_H_time_series_dates_processed,garage_H_time_series_spaces_available_processed)
        plt.axhline(y=garage_H_total_capacity, color='r', linestyle='-')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage H : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageH.png')
        plt.show()

    return garage_H_time_series_dates_processed, garage_H_time_series_spaces_available_processed




def visualize_and_process_garage_I(garage_I_time_series_dates,garage_I_time_series_spaces_available, showing ):

    garage_I_time_series_spaces_available = np.array(garage_I_time_series_spaces_available)
    garage_I_time_series_spaces_available_processed = np.where(garage_I_time_series_spaces_available>garage_I_total_capacity,garage_I_total_capacity,garage_I_time_series_spaces_available)
    garage_I_time_series_dates_processed = garage_I_time_series_dates


    if showing:
        plt.plot(garage_I_time_series_dates,garage_I_time_series_spaces_available)
        plt.plot(garage_I_time_series_dates_processed,garage_I_time_series_spaces_available_processed)
        plt.axhline(y=garage_I_total_capacity, color='r', linestyle='-')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage I : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageI.png')
        plt.show()


    return garage_I_time_series_dates_processed, garage_I_time_series_spaces_available_processed



def visualize_and_process_garage_Libra(garage_Libra_time_series_dates,garage_Libra_time_series_spaces_available, showing ):

    garage_Libra_time_series_spaces_available = np.array(garage_Libra_time_series_spaces_available)
    garage_Libra_time_series_spaces_available_processed = np.where(garage_Libra_time_series_spaces_available>garage_Libra_total_capacity,garage_Libra_total_capacity,garage_Libra_time_series_spaces_available)
    garage_Libra_time_series_dates_processed = garage_Libra_time_series_dates


    if showing:
        plt.plot(garage_Libra_time_series_dates,garage_Libra_time_series_spaces_available)
        plt.plot(garage_Libra_time_series_dates_processed,garage_Libra_time_series_spaces_available_processed)
        plt.axhline(y=garage_Libra_total_capacity, color='r', linestyle='-')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title('Garage Libra : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig('plots/garageLibra.png')
        plt.show()

    return garage_Libra_time_series_dates_processed, garage_Libra_time_series_spaces_available_processed





