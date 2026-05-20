import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# #################################################
# Eda | Visualización de algunos gráficos
# #################################################

BANK_BLUE = "#0A2540"
BANK_RED = "#8B1E3F"

# Usuarios que se fueron o quedaron
def plot_churn_distribution(df):
    
    plt.figure(figsize=(8,5))
    
    sns.countplot(
        data=df, 
        x="Exited",
        hue="Exited",
        palette=[BANK_BLUE, BANK_RED],
        legend=False
    )

    plt.xticks(
        [0, 1],
        ["Se Quedó", "Se Fue"]
    )
    plt.title("Distribución de Clientes que se Quedaron o se Fueron", fontsize=14, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Clientes")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/Clientes_Churn.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Ver si los clientes activos se van o no
def plot_active_customers_vs_churn(df):
    
    plt.figure(figsize=(8,5))
    
    sns.countplot(
        data=df,
        x="IsActiveMember",
        hue="Exited",
        palette=[BANK_BLUE, BANK_RED],
        legend=False
    )

    plt.xticks([0, 1], ["Inactivo", "Activo"])

    plt.legend(
        title="Churn",
        labels=["Se Quedó", "Se Fue"]
    )

    plt.title("Clientes Activos vs Churn", fontsize=14, fontweight="bold")
    
    plt.tight_layout()

    plt.savefig(
        "reports/figures/Churn_Active_Customers.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close() 

# Matriz de correlación para las variables numéricas
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=np.number)
    
    plt.figure(figsize=(12,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="Blues"
    )

    plt.title("Matriz de Correlación", fontsize=14, fontweight="bold")
    
    plt.tight_layout()

    plt.savefig(
        "reports/figures/Matriz_de_Correlacion.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
