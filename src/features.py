import pandas as pd

# #################################################
# Creación de nuevas características a partir de las existentes
# #################################################

def create_features(df):
    # Clientes que han estado mas de 5 años
    df["loyal_customer"] = (
        df["Tenure"] >= 5
    ).astype(int)

    # División por grupos de edad
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[18,30,40,50,60,100],
        labels=["18-30","31-40","41-50","51-60","60+"]
    )

    return df