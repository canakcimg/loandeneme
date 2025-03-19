"""
Veri hazırlama işlemleri için modül.
"""
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik değerleri doldurur.

    Args:
        df (pd.DataFrame): İşlenecek veri seti

    Returns:
        pd.DataFrame: Eksik değerleri doldurulmuş veri seti
    """
    try:
        # Sayısal değerler için median
        numerical_imputer = SimpleImputer(strategy='median')
        # Kategorik değerler için mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        df_processed = df.copy()
        
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        df_processed[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
        df_processed[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
        
        return df_processed
    except Exception as e:
        logger.error(f"Eksik değer doldurma hatası: {str(e)}")
        raise


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik değişkenleri encode eder.

    Args:
        df (pd.DataFrame): İşlenecek veri seti

    Returns:
        pd.DataFrame: Encode edilmiş veri seti
    """
    try:
        df_encoded = df.copy()
        label_encoders = {}
        
        for column in df.select_dtypes(include=['object']).columns:
            if column != 'Loan_Status':  # Target değişkeni hariç
                label_encoders[column] = LabelEncoder()
                df_encoded[column] = label_encoders[column].fit_transform(df[column])
        
        return df_encoded
    except Exception as e:
        logger.error(f"Encoding hatası: {str(e)}")
        raise
    
     