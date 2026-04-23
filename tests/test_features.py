"""Testes de feature engineering — schema contracts."""
import pandera as pa
from pandera import Column, DataFrameSchema

from src.features.feature_engineering import separe_columns, convert_categorical_to_numeric


FEATURE_SCHEMA = DataFrameSchema({
    'Daily_Screen_Time': Column(float, pa.Check.gt(0)),
    'Stress_Level': Column(int, pa.Check.gt(0)),
    'Anxiety': Column(int, pa.Check.isin([0, 1])),
    'Depression': Column(int, pa.Check.isin([0, 1])),
    'Occupation_A': Column(float, pa.Check.isin([0, 1])),
    'Occupation_B': Column(float, pa.Check.isin([0, 1])),
    'Occupation_C': Column(float, pa.Check.isin([0, 1])),
    'Physical_Activity_Low': Column(float, pa.Check.isin([0, 1])),
    'Physical_Activity_Medium': Column(float, pa.Check.isin([0, 1])),
    'Physical_Activity_High': Column(float, pa.Check.isin([0, 1])),
})


def test_schema_contract(sample_data):
    """Features de saída devem respeitar o contrato de schema."""
    result = separe_columns(sample_data)
    result = convert_categorical_to_numeric(result)
    FEATURE_SCHEMA.validate(result)


def test_no_nulls(sample_data):
    """Nenhuma feature pode ter null após transformação."""
    result = separe_columns(sample_data)
    result = convert_categorical_to_numeric(result)
    assert result.isnull().sum().sum() == 0


def test_row_count_preserved(sample_data):
    """Número de registros deve ser preservado."""
    result = separe_columns(sample_data)
    result = convert_categorical_to_numeric(result)
    assert len(result) == len(sample_data)