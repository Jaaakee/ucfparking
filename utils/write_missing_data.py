#2022-04-25 05:00:00 to 2022-04-29 20:00:00 (timezones are UTC)
import os
import mysql.connector
import json
from utils.get_env_var import *
import datetime

path_2022_04_25 = 'missing_days/2022-04-25.json'
path_2022_04_26 = 'missing_days/2022-04-26.json'
path_2022_04_27 = 'missing_days/2022-04-27.json'
path_2022_04_28 = 'missing_days/2022-04-28.json'
path_2022_04_29 = 'missing_days/2022-04-29.json'

db = mysql.connector.connect(
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASS'],
    database=os.environ['DB_NAME']
)

cursor = db.cursor()


data_2022_04_25 = json.load(open(path_2022_04_25))
data_2022_04_26 = json.load(open(path_2022_04_26))
data_2022_04_27 = json.load(open(path_2022_04_27))
data_2022_04_28 = json.load(open(path_2022_04_28))
data_2022_04_29 = json.load(open(path_2022_04_29))

for missing_date in [data_2022_04_25,data_2022_04_26,data_2022_04_27,data_2022_04_28,data_2022_04_29]:
    for date_time_result in missing_date['data']:
        transformed_date_string = date_time_result['date'].split('T')[0] + ' ' + date_time_result['date'].split('T')[1].split(':')[0] + ':00:00'
        object_date = datetime.datetime.strptime(transformed_date_string, '%Y-%m-%d %H:%M:%S')
        if (object_date > datetime.datetime.strptime('2022-04-25 04:00:00', '%Y-%m-%d %H:%M:%S') and object_date < datetime.datetime.strptime('2022-04-29 21:00:00', '%Y-%m-%d %H:%M:%S')):
            appendarray = [ str(object_date)]

            for obj in date_time_result['garages']:
                appendarray.append(obj["spaces_left"])
                appendarray.append(obj["max_spaces"])
                appendarray.append(obj["percent_full"])


            # Insert data into the database table
            sql = "INSERT INTO parking_data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql, appendarray)
            db.commit()

cursor.close()
db.close()
