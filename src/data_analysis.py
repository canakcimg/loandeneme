"""
Veri analizi için modül.
"""
from typing import Dict, List

import pandas as pd
from loguru import logger


def get_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Eksik değerlerin yüzdesini hesaplar.

    Args:
        df (pd.DataFrame): İncelenecek veri seti

    Returns:
        Dict[str, float]: Sütun bazında eksik değer yüzdeleri
    """
    try:
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        return missing_percentages.to_dict()
    except Exception as e:
        logger.error(f"Eksik değer analizi hatası: {str(e)}")
        raise


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Sayısal sütunları döndürür.

    Args:
        df (pd.DataFrame): İncelenecek veri seti

    Returns:
        List[str]: Sayısal sütun isimleri
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Kategorik sütunları döndürür.

    Args:
        df (pd.DataFrame): İncelenecek veri seti

    Returns:
        List[str]: Kategorik sütun isimleri
    """
    return df.select_dtypes(include=['object']).columns.tolist() 