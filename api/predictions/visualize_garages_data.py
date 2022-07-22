from utils.get_env_var import *
import mysql.connector
from utils.query_database import *
import matplotlib.pyplot as plt
import numpy as np
import datetime




#Total capacity of each garage known in advance.
garage_A_total_capacity = 1623
garage_B_total_capacity = 1259
garage_C_total_capacity = 1852
garage_D_total_capacity = 1241
garage_H_total_capacity = 1284
garage_I_total_capacity = 1231
garage_Libra_total_capacity = 1007


#Get all garages data (date + space availables)
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



#Process data and visualize it if asked.
def visualize_and_process_garage(date_data,spaces_available_data, showing, which_garage, total_capacity ):
    # Reverse the lists to have data in correct order (from 2021 to 2022 and not the opposite)
    date_data = list(reversed(date_data))
    spaces_available_data = list(reversed(spaces_available_data))

    spaces_available_data = np.array(spaces_available_data) #transform to numpy array
    spaces_available_data_processed = np.where(spaces_available_data>total_capacity,total_capacity,spaces_available_data) #Clean glitches of the API (values bigger than total capacity)
    date_data_processed = date_data

    if (which_garage == 'A'): #Remove period where garage A was closed
        index_to_begin_garage_A = date_data.index(datetime.datetime(2021, 8, 13, 14, 0))
        date_data_processed = date_data[index_to_begin_garage_A:]
        spaces_available_data_processed = spaces_available_data_processed[index_to_begin_garage_A:]


    if showing:
        plt.plot(date_data,spaces_available_data)
        plt.plot(date_data_processed,spaces_available_data_processed)
        plt.axhline(y=total_capacity, color='r', linestyle='--')
        plt.xlabel('Date (Time series)')
        plt.ylabel('Spaces Available')
        plt.title(f'Garage {which_garage} : Availability of spaces in function of the time')
        plt.legend(['Raw','Processed'])
        plt.savefig(f'plots/garage{which_garage}.png')
        plt.show()

    return date_data_processed, spaces_available_data_processed



if __name__ == "__main__":
    garage_A_time_series_dates, garage_B_time_series_dates, garage_C_time_series_dates, garage_D_time_series_dates, garage_H_time_series_dates, garage_I_time_series_dates, garage_Libra_time_series_dates, garage_A_time_series_spaces_available, garage_B_time_series_spaces_available, garage_C_time_series_spaces_available, garage_D_time_series_spaces_available, garage_H_time_series_spaces_available, garage_I_time_series_spaces_available, garage_Libra_time_series_spaces_available = get_garages_data_for_predictions()


    garage_A_time_series_dates_processed, garage_A_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_A_time_series_dates, garage_A_time_series_spaces_available, True, 'A', garage_A_total_capacity)


    garage_B_time_series_dates_processed, garage_B_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_B_time_series_dates, garage_B_time_series_spaces_available, True, 'B', garage_B_total_capacity)


    garage_C_time_series_dates_processed, garage_C_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_C_time_series_dates, garage_C_time_series_spaces_available, True, 'C', garage_C_total_capacity)
    #

    garage_D_time_series_dates_processed, garage_D_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_D_time_series_dates, garage_D_time_series_spaces_available, True, 'D', garage_D_total_capacity)


    garage_H_time_series_dates_processed, garage_H_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_H_time_series_dates, garage_H_time_series_spaces_available, True, 'H', garage_H_total_capacity)


    garage_I_time_series_dates_processed, garage_I_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_I_time_series_dates, garage_I_time_series_spaces_available, True, 'I', garage_I_total_capacity)


    garage_Libra_time_series_dates_processed, garage_Libra_time_series_spaces_available_processed = visualize_and_process_garage(
        garage_Libra_time_series_dates, garage_Libra_time_series_spaces_available, True, 'Libra', garage_Libra_total_capacity)


