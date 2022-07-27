"""Script that fills missing dates from the API."""
import datetime

from utils.query_database import Database


def fill_missing_dates_main():
    """Fill missing dates from the API."""
    # Setup the database and query object.
    my_db = Database()
    object_query = my_db.setup_query_extractor()

    data_missing = True

    while data_missing:
        # Retrieve all the data from the database in data_list.
        content = object_query.get_all_data()
        data_list = []
        for i in range(len(content["data"])):
            data_list.append(content["data"][i])

        # Check if there are any missing dates.
        for index in range(len(data_list)):

            if index != len(data_list) - 1 and datetime.datetime.strptime(
                data_list[index + 1]["date_and_time"], "%Y-%m-%d %H:%M:%S"
            ) + datetime.timedelta(hours=1) != datetime.datetime.strptime(
                data_list[index]["date_and_time"], "%Y-%m-%d %H:%M:%S"
            ):
                dict_data = data_list[index]
                refined_date = datetime.datetime.strptime(
                    dict_data["date_and_time"], "%Y-%m-%d %H:%M:%S"
                ) - datetime.timedelta(hours=1)
                appendarray = [str(refined_date)]

                for garage_name, values in dict_data["garages"].items():
                    appendarray.append(values["spaces_left"])
                    appendarray.append(values["max_spaces"])
                    appendarray.append(values["percent_full"])

                # print('Data row entered', appendarray, index)
                sql = "INSERT INTO parking_data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                my_db.update_database(sql, appendarray)
                break

            if index == len(data_list) - 1:
                data_missing = False
                break

    my_db.close_connection()


if __name__ == "__main__":
    fill_missing_dates_main()
