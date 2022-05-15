from utils.get_env_var import *
import mysql.connector
from utils.query_database import *
import datetime


#TODO Add script that fills missing days data from the API to have correct time series.

my_database = mysql.connector.connect(
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASS'],
    database=os.environ['DB_NAME']
)

my_cursor = my_database.cursor()
object_query = DataQueryExtractor(my_database, my_cursor)

data_missing = True

while data_missing:
    content = object_query.get_all_data()
    data_list= []
    for i in range(len(content['data'])):
        data_list.append(content['data'][i])


    for index in range(len(data_list)):

        if (index != len(data_list) - 1 and datetime.datetime.strptime(data_list[index + 1]['date_and_time'], '%Y-%m-%d %H:%M:%S')  + datetime.timedelta(hours=1) != datetime.datetime.strptime(data_list[index]['date_and_time'], '%Y-%m-%d %H:%M:%S')):
            # dates_list[index] miss 1 hour before that ==> add it
            print(datetime.datetime.strptime(data_list[index + 1]['date_and_time'],
                                             '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=1),
                  datetime.datetime.strptime(data_list[index]['date_and_time'], '%Y-%m-%d %H:%M:%S'))
            dict_data = data_list[index]
            refined_date = datetime.datetime.strptime(dict_data['date_and_time'],
                                                      '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=1)
            appendarray = [str(refined_date)]

            for garage_name, values in dict_data['garages'].items():
                appendarray.append(values["spaces_left"])
                appendarray.append(values["max_spaces"])
                appendarray.append(values["percent_full"])
            print('Data row entered', appendarray, index)
            # # # Insert data into the database table
            sql = "INSERT INTO parking_data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            my_cursor.execute(sql, appendarray)
            my_database.commit()
            break

        if (index == len(data_list) - 1):
            data_missing = False
            break


my_cursor.close()
my_database.close()








