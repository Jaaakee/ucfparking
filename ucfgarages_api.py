import json
from get_env_var import *
import mysql.connector
from fastapi import FastAPI, Response

from query_database import *

app = FastAPI()


def get_data(data, year=None, month=None, day=None):
    my_database = mysql.connector.connect(
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'],
        database=os.environ['DB_NAME']
    )

    my_cursor = my_database.cursor()
    object_query = DataQueryExtractor(my_database, my_cursor)

    content = None
    if data == 'last':
        content = object_query.get_last_data()
    elif data == 'all':
        content = object_query.get_all_data()
    elif data == 'today':
        content = object_query.get_today_data()
    elif data == 'week':
        content = object_query.get_last_week_data()
    elif data == 'year':
        content = object_query.get_yearly_data(year)
    elif data == 'year-month':
        content = object_query.get_yearly_monthly_data(year, month)
    elif data == 'year-month-day':
        content = object_query.get_yearly_monthly_daily_data(year, month, day)

    my_cursor.close()
    my_database.close()

    if content:
        return content


# Get the last data row from the database.
@app.get("/")
async def get_last_data():
    content = get_data('last')

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get all of the data from the database.
@app.get("/all")
async def get_alldb_data():
    content = get_data('all')

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get today's data from the database.
@app.get("/today")
async def get_todaydb_data():
    content = get_data('today')

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get this week's data from the database.
@app.get("/week")
async def get_weekly_data():
    content = get_data('week')

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get the specified year's data from the database.
@app.get("/{year}")
async def get_year_data(year):
    content = get_data('year', year)

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get the specified year-month's data from the database.
@app.get("/{year}/{month}")
async def get_year_month_data(year, month):
    content = get_data('year-month', year, month)

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")


# Get the specified year-month-day's data from the database.
@app.get("/{year}/{month}/{day}")
async def get_year_month_day_data(year, month, day):
    content = get_data('year-month-day', year, month, day)

    return Response(content=json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=3,
        separators=(", ", ": "),
    ).encode("utf-8"), status_code=200, media_type="application/json")
