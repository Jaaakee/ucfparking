import datetime


def row_formatting(data, cnt):
    date_of_row = data[0]

    garage_row_queried = {
        'date_and_time': str(date_of_row),
        "garages": {},
    }

    name_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'H', '5': 'I', '6': 'Libra'}
    cnt += 1

    for garage_number in range(7):
        formatted_dict_for_row_garage_number = {'total_capacity': int(float(data[3 * garage_number + 2])),
                                                'percent_full': int(float(data[3 * garage_number + 3])), 'spaces_filled': int(float(
                int(data[3 * garage_number + 2])) - int(float(data[3 * garage_number + 1]))),
                                                'spaces_available': int(float(data[3 * garage_number + 1]))}

        garage_row_queried['garages'][name_dict[str(garage_number)]] = (formatted_dict_for_row_garage_number)

    return garage_row_queried, cnt


class DataQueryExtractor:
    def __init__(self, my_database, my_cursor):
        self.my_database = my_database
        self.my_cursor = my_cursor

    def get_last_week_data(self):
        week_ago_date = datetime.datetime.now() - datetime.timedelta(days=7)
        query = "SELECT * FROM parking_data WHERE date_and_time > %s ORDER BY date_and_time DESC"
        self.my_cursor.execute(query, [week_ago_date])

        weekly_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)
            weekly_data_queried['data'].append(garage_row_queried)

        weekly_data_queried['count'] = cnt

        return weekly_data_queried

    def get_last_data(self):
        query = "SELECT MAX(date_and_time) FROM parking_data"
        self.my_cursor.execute(query)
        for date in self.my_cursor:
            max_date = date[0]

        query = "SELECT * FROM parking_data WHERE date_and_time = %s"
        self.my_cursor.execute(query, [max_date])

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)

        return garage_row_queried

    def get_yearly_data(self, year):
        query = "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s ORDER BY date_and_time DESC"
        self.my_cursor.execute(query, [year])

        yearly_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_data_queried['data'].append(garage_row_queried)

        yearly_data_queried['count'] = cnt

        return yearly_data_queried

    def get_yearly_monthly_data(self, year, month):
        query = "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s ORDER BY date_and_time DESC"
        self.my_cursor.execute(query, [year, month])

        yearly_monthly_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_monthly_data_queried['data'].append(garage_row_queried)

        yearly_monthly_data_queried['count'] = cnt

        return yearly_monthly_data_queried

    def get_yearly_monthly_daily_data(self, year, month, the_day):
        query = "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s ORDER BY date_and_time DESC"
        self.my_cursor.execute(query, [year, month, the_day])

        yearly_monthly_daily_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)
            yearly_monthly_daily_data_queried['data'].append(garage_row_queried)

        yearly_monthly_daily_data_queried['count'] = cnt

        return yearly_monthly_daily_data_queried

    def get_all_data(self):
        query = "SELECT * FROM parking_data ORDER BY date_and_time DESC"
        self.my_cursor.execute(query)
        result = self.my_cursor.fetchall()
        all_data_queried = {
            'count': 0,
            'data': [0] * len(result),
        }

        cnt = 0

        for index, data in enumerate(result):
            garage_row_queried, cnt = row_formatting(data, cnt)
            all_data_queried['data'][index] = garage_row_queried

        all_data_queried['count'] = cnt

        return all_data_queried

    def get_today_data(self):
        today_date = datetime.datetime.now()
        today_date = str(today_date)
        year = today_date.split('-')[0]
        month = today_date.split('-')[1]
        day = today_date.split('-')[2]
        query = "SELECT * FROM parking_data WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s ORDER BY date_and_time DESC"
        self.my_cursor.execute(query, [year, month, day])

        all_today_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:
            garage_row_queried, cnt = row_formatting(data, cnt)
            all_today_data_queried['data'].append(garage_row_queried)

        all_today_data_queried['count'] = cnt

        return all_today_data_queried
