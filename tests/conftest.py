"""Fixtures compartilhados para testes."""
import pandas as pd
from random import randint, random
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Dados sintéticos para testes (nunca dados reais)."""
    return pd.DataFrame({
        'Daily_Screen_Time': [random() for _ in range(8)],
        'Stress_Level': [randint(1, 10) for _ in range(8)],
        'Anxiety': [randint(0, 1) for _ in range(8)],
        'Physical_Activity': ["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium"],
        'Occupation': ["A", "B", "A", "C", "B", "A", "C", "B"],
        'Depression': [randint(0, 1) for _ in range(8)],
    })