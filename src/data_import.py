"""
Veri setlerinin import edilmesi için modül.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train ve test veri setlerini yükler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train ve test dataframe'leri
    """
    try:
        data_dir = Path("data")
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        logger.info("Veri setleri başarıyla yüklendi")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}")
        raise


def combine_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:

    """
    Train ve test verilerini birleştirir.

    Args:
        train_df (pd.DataFrame): Eğitim veri seti
        test_df (pd.DataFrame): Test veri seti

    Returns:
        pd.DataFrame: Birleştirilmiş veri seti
    """
    try:
        combined_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        logger.info("Veri setleri başarıyla birleştirildi")
        return combined_data
    except Exception as e:
        logger.error(f"Veri birleştirme hatası: {str(e)}")
        raise 