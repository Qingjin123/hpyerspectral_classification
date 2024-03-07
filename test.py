from config.yamls import read_yaml
from data.load_data import load_data

yy = read_yaml()

data_name = 'PaviaU'

data, label = load_data(yy, data_name)