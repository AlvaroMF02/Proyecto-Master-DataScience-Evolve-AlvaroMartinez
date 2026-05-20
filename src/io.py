import pandas as pd

# #################################################
# Carga y guardado de datos
# #################################################

def load_data(path):
    return pd.read_csv(path)


def save_dataframe(df, path):
    df.to_csv(path, index=False)