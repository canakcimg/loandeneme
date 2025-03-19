"""
Veri görselleştirme işlemleri için modül.
"""
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str]) -> None:
    """
    Sayısal değişkenlerin dağılımlarını çizer.

    Args:
        df (pd.DataFrame): Veri seti
        numerical_cols (List[str]): Sayısal sütun isimleri
    """
    try:
        for col in numerical_cols:
            plt.figure(figsize=(10, 5))
            df[col].plot(kind='hist', bins=20, title=col)
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.show()
        logger.info("Sayısal değişken dağılımları çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str]) -> None:
    """
    Kategorik değişkenlerin dağılımlarını çizer.

    Args:
        df (pd.DataFrame): Veri seti
        categorical_cols (List[str]): Kategorik sütun isimleri
    """
    try:
        for col in categorical_cols:
            plt.figure(figsize=(10, 5))
            df.groupby(col).size().plot(
                kind='barh',
                color=sns.palettes.mpl_palette('Dark2')
            )
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.title(f'{col} Distribution')
            plt.show()
        logger.info("Kategorik değişken dağılımları çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: List[str]) -> None:
    """
    Sayısal değişkenler için korelasyon matrisini çizer.

    Args:
        df (pd.DataFrame): Veri seti
        numerical_cols (List[str]): Sayısal sütun isimleri
    """
    try:
        plt.figure(figsize=(20, 5))
        sns.heatmap(df[numerical_cols].corr(), annot=True)
        plt.title('Correlation Matrix')
        plt.show()
        logger.info("Korelasyon matrisi çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def plot_scatter_plots(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None
) -> None:
    """
    İki değişken arasındaki ilişkiyi scatter plot ile görselleştirir.

    Args:
        df (pd.DataFrame): Veri seti
        x_col (str): X ekseni değişkeni
        y_col (str): Y ekseni değişkeni
        title (str, optional): Grafik başlığı
    """
    try:
        plt.figure(figsize=(10, 5))
        df.plot(kind='scatter', x=x_col, y=y_col, s=32, alpha=.8)
        plt.gca().spines[['top', 'right']].set_visible(False)
        if title:
            plt.title(title)
        plt.show()
        logger.info(f"{x_col} vs {y_col} scatter plot çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def plot_heatmap_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None
) -> None:
    """
    İki kategorik değişken arasındaki ilişkiyi heatmap ile görselleştirir.

    Args:
        df (pd.DataFrame): Veri seti
        x_col (str): X ekseni değişkeni
        y_col (str): Y ekseni değişkeni
        title (str, optional): Grafik başlığı
    """
    try:
        plt.subplots(figsize=(8, 8))
        df_2dhist = pd.DataFrame({
            x_label: grp[y_col].value_counts()
            for x_label, grp in df.groupby(x_col)
        })
        sns.heatmap(df_2dhist, cmap='viridis')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if title:
            plt.title(title)
        plt.show()
        logger.info(f"{x_col} vs {y_col} heatmap çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def plot_line_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = None
) -> None:
    """
    Gruplandırılmış seri verilerini çizgi grafik olarak görselleştirir.

    Args:
        df (pd.DataFrame): Veri seti
        x_col (str): X ekseni değişkeni
        y_col (str): Y ekseni değişkeni
        group_col (str): Gruplandırma değişkeni
        title (str, optional): Grafik başlığı
    """
    try:
        def _plot_series(series, series_name, series_index=0):
            palette = list(sns.palettes.mpl_palette('Dark2'))
            xs = series[x_col]
            ys = series[y_col]
            plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

        fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
        df_sorted = df.sort_values(x_col, ascending=True)
        
        for i, (series_name, series) in enumerate(df_sorted.groupby(group_col)):
            _plot_series(series, series_name, i)
            fig.legend(title=group_col, bbox_to_anchor=(1, 1), loc='upper left')
        
        sns.despine(fig=fig, ax=ax)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if title:
            plt.title(title)
        plt.show()
        logger.info(f"{x_col} vs {y_col} line series plot çizildi")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise


def visualize_all_features(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> None:
    """
    Tüm özellikler için görselleştirmeleri yapar.

    Args:
        df (pd.DataFrame): Veri seti
        numerical_cols (List[str]): Sayısal sütun isimleri
        categorical_cols (List[str]): Kategorik sütun isimleri
    """
    try:
        # Sayısal değişken dağılımları
        plot_numerical_distributions(df, numerical_cols)
        
        # Kategorik değişken dağılımları
        plot_categorical_distributions(df, categorical_cols)
        
        # Korelasyon matrisi
        plot_correlation_matrix(df, numerical_cols)
        
        # Scatter plotlar
        plot_scatter_plots(df, 'person_age', 'person_income', 'Person Age vs Income')
        plot_scatter_plots(df, 'person_income', 'loan_amnt', 'Income vs Loan Amount')
        
        # Heatmap'ler
        plot_heatmap_2d(df, 'person_home_ownership', 'loan_intent', 'Home Ownership vs Loan Intent')
        plot_heatmap_2d(df, 'loan_grade', 'cb_person_default_on_file', 'Loan Grade vs Default Status')
        
        # Line series plot
        plot_line_series(df, 'id', 'person_age', 'loan_grade', 'ID vs Age by Loan Grade')
        
        logger.info("Tüm görselleştirmeler tamamlandı")
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise 