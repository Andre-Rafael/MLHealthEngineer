from logging import info

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

def separe_columns(df):
    # Selecionar apenas as colunas relevantes para o modelo
    features = df[['Daily_Screen_Time', 'Stress_Level', 'Anxiety', 'Physical_Activity', 'Occupation', 'Depression']]
    
    return features

def convert_categorical_to_numeric(df):
    # Converter a coluna Occupation e Physical_Activity para numérica usando one-hot encoding
    ohe = OneHotEncoder(sparse_output=False)
    df[ohe.get_feature_names_out()] = ohe.fit_transform(df[['Occupation', 'Physical_Activity']])
    df.drop(columns=['Occupation', 'Physical_Activity'], inplace=True)
    return df

def set_scaling(df):
    # Aplicar standard scaling às colunas numéricas
    scaler = StandardScaler().set_output(transform="pandas")
    df_scaled = scaler.fit_transform(df.drop('Depression', axis=1))
    df_scaled['Depression'] = df['Depression']
    return df_scaled



if __name__ == "__main__":

    df = pd.read_csv(
        "C:\\Users\\Andre Rafael\\Downloads\\mental_health_project\\data\\raw\\mental_health.csv"
    )

    info("Cleaning and feature engineering data")
    for step_func in [separe_columns, convert_categorical_to_numeric, set_scaling]:
        df = step_func(df)

    df.to_csv(
        'C:\\Users\\Andre Rafael\\Downloads\\mental_health_project\\data\\processed\\features.csv',
        index=False
    )