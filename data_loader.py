import pandas as pd


def data_loading(path):
    data = pd.read_csv(path)
    return data


def labels_loading():
    labels = ['STEPING', 'STANDING', 'LYING', 'FEEDING']
    return labels
