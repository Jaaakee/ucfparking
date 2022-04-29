import os

import mysql.connector
from fastapi import FastAPI, Response

from query_database import *
from get_env_variables import *

app = FastAPI()

my_database = mysql.connector.connect(
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASS'],
    database=os.environ['DB_NAME']
)

my_cursor = my_database.cursor()


# all data based on parameters to return as json file.
@app.get("/all")
async def get_alldb_data():
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_all_data()
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# all today data  to return as json file.
@app.get("/today")
async def get_todaydb_data():
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_today_data()
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")



# Weekly data to return as json file.
@app.get("/")
async def get_last_data():
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_last_data()
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Weekly data to return as json file.
@app.get("/week")
async def get_weekly_data():
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_last_week_data()
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# yearly data based on parameter to return as json file.
@app.get("/{year}")
async def get_year_data(year):
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_yearly_data(year)
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# yearly monthly data based on parameters to return as json file.
@app.get("/{year}/{month}")
async def get_year_month_data(year, month):
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_yearly_monthly_data(year, month)
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# yearly monthly daily data based on parameters to return as json file.
@app.get("/{year}/{month}/{day}")
async def get_year_month_day_data(year, month, day):
    my_database = connect_to_db()
    my_cursor = my_database.cursor()
    Object_Query = DataQueryExtractor(my_database, my_cursor)
    content = Object_Query.get_yearly_monthly_daily_data(year, month, day)
    my_cursor.close()
    my_database.close()
    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")
