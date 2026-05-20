import pandas as pd

# #################################################
# Limpieza de datos
# #################################################

# Dejar las columnas sin espacios y en minúscula
def clean_columns(df):
    # Quitar espacios en blanco
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

# Quitar columnas que no aportan mucho
def drop_unused_columns(df):
    df = df.drop(
        columns=[
            "rownumber",
            "customerid",
            "surname"
        ]
    )
    df = df.drop(columns=["complain"])
    return df

# Encoding de variables categóricas
def encode_categorical_variables(df):
    
    # Gender encoding
    df["gender"] = df["gender"].map({
        "Male": 0,
        "Female": 1
    })

    # Dummies
    df = pd.get_dummies(
        df,
        columns=[
            "geography",
            "card_type",
            "age_group"
        ],
        drop_first=True
    )

    return df


# Convertir columnas booleanas a enteros
def convert_booleans_to_int(df):
    bool_cols = df.select_dtypes(
        include="bool"
    ).columns
    df[bool_cols] = (
        df[bool_cols]
        .astype(int)
    )
    return df