from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# #################################################
# Preparacion de los datos para el modelado
# #################################################

# Dividir train y test
def split_data(df):

    X = df.drop("exited", axis=1)
    y = df["exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# Escalar datos numéricos
def scale_data(X_train, X_test):

    numerical_cols = [
        "creditscore",
        "age",
        "tenure",
        "balance",
        "numofproducts",
        "estimatedsalary",
        "satisfaction_score",
        "point_earned"
    ]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = (
        scaler.fit_transform(
            X_train[numerical_cols]
        )
    )

    X_test_scaled[numerical_cols] = (
        scaler.transform(
            X_test[numerical_cols]
        )
    )

    return X_train_scaled, X_test_scaled