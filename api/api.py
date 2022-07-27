"""FastAPI api for interacting with the database."""
import json

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from utils.query_database import Database

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_data(data, year=None, month=None, day=None):
    """Get data from the database according to the user's request."""
    my_db = Database()
    object_query = my_db.setup_query_extractor()

    content = None
    if data == "last":
        content = object_query.get_last_data()
    elif data == "all":
        content = object_query.get_all_data()
    elif data == "today":
        content = object_query.get_today_data()
    elif data == "week":
        content = object_query.get_last_week_data()
    elif data == "year":
        content = object_query.get_yearly_data(year)
    elif data == "year-month":
        content = object_query.get_yearly_monthly_data(year, month)
    elif data == "year-month-day":
        content = object_query.get_yearly_monthly_daily_data(year, month, day)
    elif data == "stats":
        content = object_query.stats()
    elif data == "lastday":
        content = object_query.get_lastday()
    elif data == "lastmonth":
        content = object_query.get_lastmonth()
    elif data == "lastyear":
        content = object_query.get_lastyear()

    my_db.close_connection()

    if content:
        return content


@app.get("/")
async def get_last_data(cache=True):
    """Get the last data row from the database."""
    if not cache:
        content = get_data("last")
    else:
        content = json.load(open("./api/content/last.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/stats")
async def get_stats(cache=True):
    """Get the stats from the database : total amount of rows and last data entry."""
    if not cache:
        content = get_data("stats")
    else:
        content = json.load(open("./api/content/stats.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/lastday")
async def get_lastday(cache=True):
    """Get the last day data from the database."""
    if not cache:
        content = get_data("lastday")
    else:
        content = json.load(open("./api/content/lastday.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/lastmonth")
async def get_lastmonth(cache=True):
    """Get the last month data from the database."""
    if not cache:
        content = get_data("lastmonth")
    else:
        content = json.load(open("./api/content/lastmonth.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/lastyear")
async def get_lastyear(cache=True):
    """Get the last year data from the database."""
    if not cache:
        content = get_data("lastyear")
    else:
        content = json.load(open("./api/content/lastyear.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/all")
async def get_alldb_data(cache=True):
    """Get all the data from the database."""
    if not cache:
        content = get_data("all")
    else:
        content = json.load(open("./api/content/all.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/today")
async def get_todaydb_data(cache=True):
    """Get the data from the database for the current day."""
    if not cache:
        content = get_data("today")
    else:
        content = json.load(open("./api/content/today.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/week")
async def get_weekly_data(cache=True):
    """Get the data from the database for the last week."""
    if not cache:
        content = get_data("week")
    else:
        content = json.load(open("./api/content/week.json"))

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/{year}")
async def get_year_data(year):
    """Get the specified year's data from the database."""
    content = get_data("year", year)

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/{year}/{month}")
async def get_year_month_data(year, month):
    """Get the specified year-month's data from the database."""
    content = get_data("year-month", year, month)

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


@app.get("/{year}/{month}/{day}")
async def get_year_month_day_data(year, month, day):
    """Get the specified year-month-day's data from the database."""
    content = get_data("year-month-day", year, month, day)

    return Response(
        content=json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=3,
            separators=(", ", ": "),
        ).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )
