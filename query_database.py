import datetime
import json



class DataQueryExtractor():
    def __init__(self,my_database,my_cursor):
        self.my_database = my_database
        self.my_cursor = my_cursor


    def get_last_week_data(self):

        query = ("SELECT MAX(date_and_time) FROM parking_data ")
        self.my_cursor.execute(query)
        for date in self.my_cursor:
            max_date = date[0]
        week_ago_date = datetime.datetime.now() - datetime.timedelta(days=7)



        query = ("SELECT * FROM parking_data "
                 "WHERE date_and_time > %s")
        self.my_cursor.execute(query, [week_ago_date] )


        weekly_data_queried = {
            'count': 0,
            'data' : [],
        }

        cnt = 0
        for data in self.my_cursor: #each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]


            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A", 'percent_full': garage_a_percent_full, 'spaces_filled': str(int(garage_a_total_capacity) - int(garage_a_spaces_available)), 'spaces_left': garage_a_spaces_available }

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity, 'name': "Garage Libra",
                                               'percent_full': libra_garage_percent_full,
                                               'spaces_filled': str(int(libra_garage_total_capacity) - int(libra_garage_spaces_available)) ,
                                               'spaces_left': libra_garage_spaces_available}


            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)


            weekly_data_queried['data'].append(garage_row_queried)

        weekly_data_queried['count'] = cnt

        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/weekly.json', 'w') as f:
            json.dump(weekly_data_queried, f, indent=3)

        return weekly_data_queried




    def get_last_data(self):

        query = ("SELECT MAX(date_and_time) FROM parking_data ")
        self.my_cursor.execute(query)
        for date in self.my_cursor:
            max_date = date[0]




        query = ("SELECT * FROM parking_data "
                 "WHERE date_and_time = %s")
        self.my_cursor.execute(query , [max_date])




        cnt = 0
        for data in self.my_cursor: #each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]


            garage_row_queried = {
                "garages": {},
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'percent_full': garage_a_percent_full, 'spaces_filled': str(int(garage_a_total_capacity) - int(garage_a_spaces_available)), 'spaces_left': garage_a_spaces_available }

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B =  {'max_spaces': garage_b_total_capacity,
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C =   {'max_spaces': garage_c_total_capacity,
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D =   {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H =  {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I =  {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA ={'max_spaces': libra_garage_total_capacity, 'name': "Garage Libra",
                                               'percent_full': libra_garage_percent_full,
                                               'spaces_filled': str(int(libra_garage_total_capacity) - int(libra_garage_spaces_available)) ,
                                               'spaces_left': libra_garage_spaces_available}


            garage_row_queried['garages']['A'] = (formatted_dict_for_row_garage_A)
            garage_row_queried['garages']['B'] = (formatted_dict_for_row_garage_B)
            garage_row_queried['garages']['C'] = (formatted_dict_for_row_garage_C)
            garage_row_queried['garages']['D'] = (formatted_dict_for_row_garage_D)
            garage_row_queried['garages']['H'] = (formatted_dict_for_row_garage_H)
            garage_row_queried['garages']['I'] =  (formatted_dict_for_row_garage_I)
            garage_row_queried['garages']['Libra'] = (formatted_dict_for_row_garage_LIBRA)





        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/last.json', 'w') as f:
            json.dump(garage_row_queried, f, indent=3)

        return garage_row_queried






    def get_yearly_data(self, year):

        query = ("SELECT * FROM parking_data "
                 "WHERE EXTRACT(YEAR FROM date_and_time) = %s")
        self.my_cursor.execute(query, [year])

        yearly_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:  # each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]

            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A",
                                               'percent_full': garage_a_percent_full, 'spaces_filled': str(
                    int(garage_a_total_capacity) - int(garage_a_spaces_available)),
                                               'spaces_left': garage_a_spaces_available}

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity,
                                                   'name': "Garage Libra",
                                                   'percent_full': libra_garage_percent_full,
                                                   'spaces_filled': str(int(libra_garage_total_capacity) - int(
                                                       libra_garage_spaces_available)),
                                                   'spaces_left': libra_garage_spaces_available}

            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)

            yearly_data_queried['data'].append(garage_row_queried)

        yearly_data_queried['count'] = cnt

        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/yearly.json', 'w') as f: #ß
            json.dump(yearly_data_queried, f, indent=3)

        return yearly_data_queried




    def get_yearly_monthly_data(self, year, month):

        query = ("SELECT * FROM parking_data "
                 "WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s")
        self.my_cursor.execute(query, [year,month])

        yearly_monthly_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:  # each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]

            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A",
                                               'percent_full': garage_a_percent_full, 'spaces_filled': str(
                    int(garage_a_total_capacity) - int(garage_a_spaces_available)),
                                               'spaces_left': garage_a_spaces_available}

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity,
                                                   'name': "Garage Libra",
                                                   'percent_full': libra_garage_percent_full,
                                                   'spaces_filled': str(int(libra_garage_total_capacity) - int(
                                                       libra_garage_spaces_available)),
                                                   'spaces_left': libra_garage_spaces_available}

            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)

            yearly_monthly_data_queried['data'].append(garage_row_queried)

        yearly_monthly_data_queried['count'] = cnt

        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/yearly_monthly.json', 'w') as f:
            json.dump(yearly_monthly_data_queried, f, indent=3)

        return yearly_monthly_data_queried




    def get_yearly_monthly_daily_data(self, year, month, the_day):

        query = ("SELECT * FROM parking_data "
                 "WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s")
        self.my_cursor.execute(query, [year,month,the_day])

        yearly_monthly_daily_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:  # each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]

            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A",
                                               'percent_full': garage_a_percent_full, 'spaces_filled': str(
                    int(garage_a_total_capacity) - int(garage_a_spaces_available)),
                                               'spaces_left': garage_a_spaces_available}

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity,
                                                   'name': "Garage Libra",
                                                   'percent_full': libra_garage_percent_full,
                                                   'spaces_filled': str(int(libra_garage_total_capacity) - int(
                                                       libra_garage_spaces_available)),
                                                   'spaces_left': libra_garage_spaces_available}

            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)

            yearly_monthly_daily_data_queried['data'].append(garage_row_queried)

        yearly_monthly_daily_data_queried['count'] = cnt
        #
        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/yearly_monthly_daily.json', 'w') as f:
            json.dump(yearly_monthly_daily_data_queried, f, indent=3)

        return yearly_monthly_daily_data_queried




    def get_all_data(self):

        query = ("SELECT * FROM parking_data ")
        self.my_cursor.execute(query)

        all_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:  # each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]

            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A",
                                               'percent_full': garage_a_percent_full, 'spaces_filled': str(
                    int(garage_a_total_capacity) - int(garage_a_spaces_available)),
                                               'spaces_left': garage_a_spaces_available}

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity,
                                                   'name': "Garage Libra",
                                                   'percent_full': libra_garage_percent_full,
                                                   'spaces_filled': str(int(libra_garage_total_capacity) - int(
                                                       libra_garage_spaces_available)),
                                                   'spaces_left': libra_garage_spaces_available}


            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)

            all_data_queried['data'].append(garage_row_queried)

        all_data_queried['count'] = cnt

        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/all.json', 'w') as f:
            json.dump(all_data_queried, f, indent=3)


        return all_data_queried





    def get_today_data(self):
        today_date = datetime.datetime.now()
        today_date = str(today_date)
        year = today_date.split('-')[0]
        month = today_date.split('-')[1]
        day = today_date.split('-')[2]
        query = ("SELECT * FROM parking_data "
                 "WHERE EXTRACT(YEAR FROM date_and_time) = %s AND EXTRACT(MONTH FROM date_and_time) = %s AND EXTRACT(DAY FROM date_and_time) = %s")
        self.my_cursor.execute(query, [year,month,day])

        all_today_data_queried = {
            'count': 0,
            'data': [],
        }

        cnt = 0
        for data in self.my_cursor:  # each data row is a row from the db respecting the query
            date_of_row = data[0]
            day = (str(date_of_row).split('-')[2]).split(' ')[0]
            month = (str(date_of_row).split('-')[1])
            year = (str(date_of_row).split('-')[0])
            week_number = date_of_row.isocalendar()[1]

            garage_row_queried = {
                'date': str(date_of_row),
                'day': day,
                "garages": [],
                "month": month,
                "week": week_number,
                "year": year
            }

            cnt += 1

            garage_a_spaces_available = data[1]
            garage_a_total_capacity = data[2]
            garage_a_percent_full = data[3]
            formatted_dict_for_row_garage_A = {'max_spaces': garage_a_total_capacity, 'name': "Garage A",
                                               'percent_full': garage_a_percent_full, 'spaces_filled': str(
                    int(garage_a_total_capacity) - int(garage_a_spaces_available)),
                                               'spaces_left': garage_a_spaces_available}

            garage_b_spaces_available = data[4]
            garage_b_total_capacity = data[5]
            garage_b_percent_full = data[6]
            formatted_dict_for_row_garage_B = {'max_spaces': garage_b_total_capacity, 'name': "Garage B",
                                               'percent_full': garage_b_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_b_total_capacity) - int(garage_b_spaces_available)),
                                               'spaces_left': garage_b_spaces_available}

            garage_c_spaces_available = data[7]
            garage_c_total_capacity = data[8]
            garage_c_percent_full = data[9]
            formatted_dict_for_row_garage_C = {'max_spaces': garage_c_total_capacity, 'name': "Garage C",
                                               'percent_full': garage_c_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_c_total_capacity) - int(garage_c_spaces_available)),
                                               'spaces_left': garage_c_spaces_available}

            garage_d_spaces_available = data[10]
            garage_d_total_capacity = data[11]
            garage_d_percent_full = data[12]
            formatted_dict_for_row_garage_D = {'max_spaces': garage_d_total_capacity, 'name': "Garage D",
                                               'percent_full': garage_d_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_d_total_capacity) - int(garage_d_spaces_available)),
                                               'spaces_left': garage_d_spaces_available}

            garage_h_spaces_available = data[13]
            garage_h_total_capacity = data[14]
            garage_h_percent_full = data[15]
            formatted_dict_for_row_garage_H = {'max_spaces': garage_a_total_capacity, 'name': "Garage H",
                                               'percent_full': garage_h_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_h_total_capacity) - int(garage_h_spaces_available)),
                                               'spaces_left': garage_h_spaces_available}

            garage_i_spaces_available = data[16]
            garage_i_total_capacity = data[17]
            garage_i_percent_full = data[18]
            formatted_dict_for_row_garage_I = {'max_spaces': garage_i_total_capacity, 'name': "Garage I",
                                               'percent_full': garage_i_percent_full,
                                               'spaces_filled': str(
                                                   int(garage_i_total_capacity) - int(garage_i_spaces_available)),
                                               'spaces_left': garage_i_spaces_available}

            libra_garage_spaces_available = data[19]
            libra_garage_total_capacity = data[20]
            libra_garage_percent_full = data[21]
            formatted_dict_for_row_garage_LIBRA = {'max_spaces': libra_garage_total_capacity,
                                                   'name': "Garage Libra",
                                                   'percent_full': libra_garage_percent_full,
                                                   'spaces_filled': str(int(libra_garage_total_capacity) - int(
                                                       libra_garage_spaces_available)),
                                                   'spaces_left': libra_garage_spaces_available}


            garage_row_queried['garages'].append(formatted_dict_for_row_garage_A)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_B)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_C)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_D)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_H)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_I)
            garage_row_queried['garages'].append(formatted_dict_for_row_garage_LIBRA)

            all_today_data_queried['data'].append(garage_row_queried)

        all_today_data_queried['count'] = cnt

        # self.my_cursor.close()
        # self.my_database.close()

        with open('Output/today.json', 'w') as f:
            json.dump(all_today_data_queried, f, indent=3)


        return all_today_data_queried