"""Script that defines all functions to query data from the database."""
import datetime
import os

import mysql.connector
from utils.get_env_var import *  # noqa F403


def row_formatting(data, cnt):
    """
    Format the data to be inserted into the database to create a new row.

    :param data: The data to be formatted.
    :param cnt: number of data points in the database
    :return: formatted data and the cnt value
    """
    date_of_row = data[0]

    garage_row_queried = {
        "date_and_time": str(date_of_row),
        "garages": {},
    }

    name_dict = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
        "4": "H",
        "5": "I",
        "6": "Libra",
    }
    cnt += 1

    for garage_number in range(len(name_dict.keys())):
        spaces_filled = int(
            float(int(data[3 * garage_number + 2])) - int(float(data[3 * garage_number + 1]))
        )
        if spaces_filled < 0:
            spaces_filled = 0

        formatted_dict_for_row_garage_number = {
            "max_spaces": int(float(data[3 * garage_number + 2])),
            "percent_full": int(float(data[3 * garage_number + 3])),
            "spaces_filled": spaces_filled,
            "spaces_left": int(float(data[3 * garage_number + 1])),
        }

        garage_row_queried["garages"][  # type: ignore
            name_dict[str(garage_number)]
        ] = formatted_dict_for_row_garage_number

    return garage_row_queried, cnt


class Database:
    """Define the database interface."""

    def __init__(self):
        """Initialize the database."""
        self.database = mysql.connector.connect(
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            database=os.environ["DB_NAME"],
        )
        self.cursor = self.database.cursor()

    def setup_query_extractor(self):
        """Set the query extractor."""
        return DataQueryExtractor(self.database, self.cursor)

    def update_database(self, query, data):
        """Update the database."""
        self.cursor.execute(query, data)
        self.database.commit()

    def close_connection(self):
        """Close the connection to the database."""
        self.database.close()
        self.cursor.close()


class DataQueryExtractor:
    """Define functions to query data from the database."""

    def __init__(self, my_database, my_cursor):
        """Initialize the class."""
        self.my_database = my_database
        self.my_cursor = my_cursor

    def execute_query(self, query, *args):
        """Execute a query on the database."""
        self.my_cursor.execute(query, *args)
        return self.my_cursor.fetchall()

    def stats(self):
        """Get the stats of the database using SQL."""
        query_result = self.execute_query("SELECT MAX(date_and_time) FROM parking_data")
        for date in query_result:
            max_date = date[0]

        query_result = self.execute_query("SELECT COUNT(*) FROM parking_data")
        for count in query_result:
            count = count[0]

        stats_data = {
            "total_data_rows": count,
            "last_date_and_time": str(max_date),
        }

        return stats_data

    def get_lastday(self):
        """Get the last day of data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data ORDER BY date_and_time DESC LIMIT 24"
        )

        lastday_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            lastday_data_queried["data"].append(garage_row_queried)

        lastday_data_queried["count"] = cnt

        return lastday_data_queried

    def get_lastmonth(self):
        """Get the last month of data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data ORDER BY date_and_time DESC LIMIT 720"
        )

        lastmonth_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            lastmonth_data_queried["data"].append(garage_row_queried)

        lastmonth_data_queried["count"] = cnt

        return lastmonth_data_queried

    def get_lastyear(self):
        """Get the last year of data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data ORDER BY date_and_time DESC LIMIT 8760"
        )

        lastyear_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            lastyear_data_queried["data"].append(garage_row_queried)

        lastyear_data_queried["count"] = cnt

        return lastyear_data_queried

    def get_last_week_data(self):
        """Get the last week of data from the database using SQL."""
        week_ago_date = datetime.datetime.now() - datetime.timedelta(days=7)
        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE date_and_time > %s ORDER BY date_and_time DESC",
            [week_ago_date],
        )

        weekly_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            weekly_data_queried["data"].append(garage_row_queried)

        weekly_data_queried["count"] = cnt

        return weekly_data_queried

    def get_last_data(self):
        """Get the last data from the database using SQL."""
        query_result = self.execute_query("SELECT MAX(date_and_time) FROM parking_data")
        for date in query_result:
            max_date = date[0]

        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE date_and_time = %s", [max_date]
        )

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)

        return garage_row_queried

    def get_yearly_data(self, year):
        """Get the year data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s ORDER BY date_and_time DESC",
            [year],
        )

        yearly_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_data_queried["data"].append(garage_row_queried)

        yearly_data_queried["count"] = cnt

        return yearly_data_queried

    def get_yearly_monthly_data(self, year, month):
        """Get the yearly and monthly data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s ORDER BY date_and_time DESC",
            [year, month],
        )

        yearly_monthly_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_monthly_data_queried["data"].append(garage_row_queried)

        yearly_monthly_data_queried["count"] = cnt

        return yearly_monthly_data_queried

    def get_yearly_monthly_daily_data(self, year, month, the_day):
        """Get the yearly, monthly and daily data from the database using SQL."""
        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s ORDER BY date_and_time DESC",
            [year, month, the_day],
        )

        yearly_monthly_daily_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_monthly_daily_data_queried["data"].append(garage_row_queried)

        yearly_monthly_daily_data_queried["count"] = cnt

        return yearly_monthly_daily_data_queried

    def get_all_data(self):
        """Get all the data from the database using SQL."""
        query_result = self.execute_query("SELECT * FROM parking_data ORDER BY date_and_time DESC")

        all_data_queried = {
            "count": 0,
            "data": [0] * len(query_result),
        }

        cnt = 0

        for index, data in enumerate(query_result):
            garage_row_queried, cnt = row_formatting(data, cnt)
            all_data_queried["data"][index] = garage_row_queried

        all_data_queried["count"] = cnt

        return all_data_queried

    def get_today_data(self):
        """Get the today data from the database using SQL."""
        today_date = datetime.datetime.now()
        today_date = str(today_date)
        year = today_date.split("-")[0]
        month = today_date.split("-")[1]
        day = today_date.split("-")[2]
        query_result = self.execute_query(
            "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s ORDER BY date_and_time DESC",
            [year, month, day],
        )

        all_today_data_queried = {
            "count": 0,
            "data": [],
        }

        cnt = 0
        for data in query_result:
            garage_row_queried, cnt = row_formatting(data, cnt)
            all_today_data_queried["data"].append(garage_row_queried)

        all_today_data_queried["count"] = cnt

        return all_today_data_queried

    def get_all_data_for_predictions(self):
        """Get all the data from the database using SQL for prediction purposes."""
        query_result = self.execute_query("SELECT * FROM parking_data ORDER BY date_and_time DESC")

        return query_result
