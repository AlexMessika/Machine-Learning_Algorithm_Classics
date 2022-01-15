import datatable as dt # pip install datatble

%%time

tps_dt_october = dt.fread("data/train.csv").to_pandas()