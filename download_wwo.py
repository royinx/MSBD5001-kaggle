from wwo_hist import retrieve_hist_data

## based on one day
FREQUENCY = 24
START_DATE = '01-Jan-2017'
END_DATE = '31-DEC-2018'
## set the API_KEY
API_KEY = '2326f955e1854e48b2205957200512'
LOCATION_LIST = ['hongkong']

# stroed in .csv format
hist_weather_data = retrieve_hist_data(API_KEY,
                                        LOCATION_LIST,
                                        START_DATE,
                                        END_DATE,
                                        FREQUENCY,
                                        location_label = False,
                                        export_csv = True,
                                        store_df = True)

# cp /usr/local/lib/python3.6/dist-packages/wwo_hist/__init__.py .