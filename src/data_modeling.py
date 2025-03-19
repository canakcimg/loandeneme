"""
Model eğitimi ve değerlendirme işlemleri için modül.
"""
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class LoanPredictionModel:
    """Kredi tahmin modeli sınıfı."""
    
    def __init__(self):
        """Model sınıfının başlatıcısı."""
        self.classifiers = [
            ("CART", DecisionTreeClassifier(random_state=42)),
            ("RF", RandomForestClassifier(random_state=42, max_features='sqrt')),
            ("GBM", GradientBoostingClassifier(max_depth=4, random_state=42)),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))
        ]
        
        # Hiperparametre arama uzayları
        self.param_grids = {
            "CART": {
                'max_depth': range(1, 20),
                "min_samples_split": range(2, 30)
            },
            "RF": {
                "max_depth": [8, 15, None],
                "max_features": [5, 7, "sqrt"],
                "min_samples_split": [15, 20],
                "n_estimators": [200, 300, 500]
            },
            "XGBoost": {
                "learning_rate": [0.1, 0.01],
                "max_depth": [5, 8],
                "n_estimators": [100, 200, 500],
                "colsample_bytree": [0.5, 1]
            },
            "LightGBM": {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500, 1000],
                "colsample_bytree": [0.7, 1]
            }
        }
        
        self.best_models = {}
        
    def train_test_split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Veriyi eğitim ve test setlerine ayırır.

        Args:
            X (pd.DataFrame): Özellik matrisi
            y (pd.Series): Hedef değişken
            test_size (float): Test seti oranı

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Eğitim ve test setleri
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("Veri seti başarıyla bölündü")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Veri bölme hatası: {str(e)}")
            raise

    def evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Tüm modelleri değerlendirir.

        Args:
            X_train (pd.DataFrame): Eğitim özellikleri
            X_test (pd.DataFrame): Test özellikleri
            y_train (pd.Series): Eğitim hedefleri
            y_test (pd.Series): Test hedefleri

        Returns:
            Dict[str, Dict[str, float]]: Model metrikleri
        """
        try:
            results = {}
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
            
            for name, classifier in self.classifiers:
                logger.info(f"{name} modeli eğitiliyor...")
                
                # Model eğitimi
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                
                # Cross-validation skorları
                accuracy_cv = cross_val_score(
                    classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1
                ).mean()
                f1_cv = cross_val_score(
                    classifier, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1
                ).mean()
                precision_cv = cross_val_score(
                    classifier, X_train, y_train, cv=cv, scoring='precision', n_jobs=-1
                ).mean()
                recall_cv = cross_val_score(
                    classifier, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1
                ).mean()
                
                # Test skorları
                test_accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'cv_accuracy': accuracy_cv,
                    'cv_f1': f1_cv,
                    'cv_precision': precision_cv,
                    'cv_recall': recall_cv,
                    'test_accuracy': test_accuracy
                }
                
                logger.info(f"{name} sonuçları:")
                logger.info(f"CV Accuracy: {accuracy_cv:.4f}")
                logger.info(f"CV F1: {f1_cv:.4f}")
                logger.info(f"CV Precision: {precision_cv:.4f}")
                logger.info(f"CV Recall: {recall_cv:.4f}")
                logger.info(f"Test Accuracy: {test_accuracy:.4f}")
                
            return results
            
        except Exception as e:
            logger.error(f"Model değerlendirme hatası: {str(e)}")
            raise

    def plot_confusion_matrices(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """
        Her model için karmaşıklık matrislerini çizer.

        Args:
            X_test (pd.DataFrame): Test özellikleri
            y_test (pd.Series): Test hedefleri
            X_train (pd.DataFrame): Eğitim özellikleri
            y_train (pd.Series): Eğitim hedefleri
        """
        try:
            for name, classifier in self.classifiers:
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
                counts = [value for value in cm.flatten()]
                percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
                labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
                labels = np.asarray(labels).reshape(2, 2)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', square=True)
                plt.title(f'Confusion Matrix for {name}')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.show()
                
                print(f'Classification Report for {name}:\n')
                print(classification_report(y_test, y_pred))
                
        except Exception as e:
            logger.error(f"Karmaşıklık matrisi çizme hatası: {str(e)}")
            raise

    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        count: int = 15
    ) -> None:
        """
        Özellik önemlerini görselleştirir.

        Args:
            X (pd.DataFrame): Özellik matrisi
            y (pd.Series): Hedef değişken
            count (int): Gösterilecek özellik sayısı
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            for name, classifier in self.classifiers:
                if hasattr(classifier, 'feature_importances_'):
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    acc_score = accuracy_score(y_test, y_pred)
                    
                    feature_imp = pd.Series(
                        classifier.feature_importances_,
                        index=X.columns
                    ).sort_values(ascending=False)[:count]
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=feature_imp, y=feature_imp.index)
                    plt.xlabel('Değişken Önem Skorları')
                    plt.ylabel('Değişkenler')
                    plt.title(f'{name} - Feature Importance')
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Özellik önem grafiği çizme hatası: {str(e)}")
            raise

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        main_scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Hiperparametre optimizasyonu yapar.

        Args:
            X (pd.DataFrame): Özellik matrisi
            y (pd.Series): Hedef değişken
            cv (int): Cross-validation kat sayısı
            main_scoring (str): Optimizasyon metriği

        Returns:
            Dict[str, Any]: En iyi modeller ve skorları
        """
        try:
            logger.info("Hiperparametre optimizasyonu başlıyor...")
            best_models = {}
            scoring_metrics = ['accuracy', 'f1', 'recall', 'precision']
            
            for name, classifier in self.classifiers:
                if name in self.param_grids:
                    logger.info(f"########## {name} ##########")
                    
                    # Başlangıç skorları
                    initial_scores = {}
                    for metric in scoring_metrics:
                        cv_results = cross_val_score(
                            classifier, X, y, cv=cv, scoring=metric
                        )
                        mean_score = round(cv_results.mean(), 4)
                        initial_scores[metric] = mean_score
                        logger.info(f"{metric} (Before): {mean_score}")
                    
                    # RandomizedSearchCV ile optimizasyon
                    gs_best = RandomizedSearchCV(
                        classifier,
                        self.param_grids[name],
                        cv=cv,
                        scoring=main_scoring,
                        n_jobs=-1,
                        verbose=False
                    ).fit(X, y)
                    
                    final_model = classifier.set_params(**gs_best.best_params_)
                    logger.info(f"{name} best params: {gs_best.best_params_}")
                    
                    # Optimizasyon sonrası skorlar
                    optimized_scores = {}
                    for metric in scoring_metrics:
                        cv_results = cross_val_score(
                            final_model, X, y, cv=cv, scoring=metric
                        )
                        mean_score = round(cv_results.mean(), 4)
                        optimized_scores[metric] = mean_score
                        logger.info(f"{metric} (After): {mean_score}")
                    
                    best_models[name] = {
                        'final_model': final_model,
                        'initial_scores': initial_scores,
                        'optimized_scores': optimized_scores
                    }
            
            return best_models
            
        except Exception as e:
            logger.error(f"Hiperparametre optimizasyonu hatası: {str(e)}")
            raise


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[LoanPredictionModel, Dict[str, Dict[str, float]]]:
    """
    Tüm modelleri eğitir ve değerlendirir.

    Args:
        X (pd.DataFrame): Özellik matrisi
        y (pd.Series): Hedef değişken
        test_size (float): Test seti oranı

    Returns:
        Tuple[LoanPredictionModel, Dict[str, Dict[str, float]]]: Model ve metrikler
    """
    try:
        # Model oluştur
        model = LoanPredictionModel()
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = model.train_test_split_data(X, y, test_size)
        
        # Modelleri değerlendir
        results = model.evaluate_models(X_train, X_test, y_train, y_test)
        
        # Karmaşıklık matrislerini çiz
        model.plot_confusion_matrices(X_test, y_test, X_train, y_train)
        
        # Özellik önemlerini çiz
        model.plot_feature_importance(X, y)
        
        # Hiperparametre optimizasyonu
        # best_models = model.optimize_hyperparameters(X, y)
        
        return model, results
        
    except Exception as e:
        logger.error(f"Model eğitim ve değerlendirme hatası: {str(e)}")
        raise 