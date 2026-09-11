from logging import info

import pandas as pd
from sklearn.svm import SVC

from train import train_and_log

info("Iniciando treinamento do modelo baseline")
data = pd.read_csv('data\\processed\\features.csv')


result = train_and_log(
    df=data,
    target_col='Depression',
    model_name='Baseline SVM',
    model_class=SVC,  # Substitua pela classe do modelo que você usará
    model_params={
        "kernel": 'poly'
    },  # Substitua pelos hiperparâmetros do modelo
)

print(f"Modelo treinado e logado com run_id: {result}")