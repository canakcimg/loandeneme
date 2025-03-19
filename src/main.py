"""
Ana uygulama modülü.
"""
from loguru import logger

from src.data_import import load_data, combine_data
from src.data_preparation import handle_missing_values, encode_categorical_variables
from src.data_visualization import visualize_all_features
from src.data_modeling import train_and_evaluate_models


def main():
    """
    Ana uygulama fonksiyonu.
    """
    try:
        # Veri yükleme
        train_df, test_df = load_data()
        logger.info("Veri setleri yüklendi")

        # Verileri birleştirme
        combined_data = combine_data(train_df, test_df)
        logger.info("Veriler birleştirildi")

        # Eksik değerleri doldurma
        processed_data = handle_missing_values(combined_data)
        logger.info("Eksik değerler dolduruldu")

        # Kategorik değişkenleri encode etme
        encoded_data = encode_categorical_variables(processed_data)
        logger.info("Kategorik değişkenler encode edildi")

        # Veri setlerini ayırma
        train_processed = encoded_data[encoded_data['loan_status'].notna()]
        test_processed = encoded_data[encoded_data['loan_status'].isna()]
        
        # Görselleştirme
        numerical_cols = train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = train_processed.select_dtypes(include=['object']).columns.tolist()
        visualize_all_features(train_processed, numerical_cols, categorical_cols)
        logger.info("Görselleştirmeler tamamlandı")

        # Modelleme
        y = train_processed["loan_status"]
        X = train_processed.drop(["loan_status", "cb_person_default_on_file"], axis=1)
        
        model, results = train_and_evaluate_models(X, y)
        logger.info("Modelleme tamamlandı")
        
        # Sonuçları yazdır
        print("\nModel Sonuçları:")
        for model_name, metrics in results.items():
            print(f"\n{model_name} Sonuçları:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        return train_processed, test_processed, model, results

    except Exception as e:
        logger.error(f"İşlem hatası: {str(e)}")
        raise


if __name__ == "__main__":
    main() 

    