import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Loan Prediction Model",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: medium;
        color: #424242;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box.approved {
        background-color: #C8E6C9;
        border: 1px solid #4CAF50;
    }
    .prediction-box.rejected {
        background-color: #FFCDD2;
        border: 1px solid #F44336;
    }
    .feature-importance {
        padding: 10px;
        background-color: #F5F5F5;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">Loan Prediction Model</p>', unsafe_allow_html=True)
st.write("Bu sayfa kredi onay tahminleri için makine öğrenmesi modeli kullanmaktadır.")

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        # Model path is .pkl/LightGBM_best_model.pkl in the root directory
        model_path = os.path.join('..', '..', '.pkl', 'LightGBM_best_model.pkl')
        
        # Check if model exists in the specified path
        if not os.path.exists(model_path):
            # Try alternate locations
            alternate_paths = [
                '.pkl/LightGBM_best_model.pkl',
                '../../.pkl/LightGBM_best_model.pkl',
                'LightGBM_best_model.pkl',
                '../models/LightGBM_best_model.pkl',
                'models/LightGBM_best_model.pkl',
            ]
            
            for path in alternate_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                st.error(f"Model dosyası bulunamadı. Aranan konum: {model_path}")
                return None, None, None, None
        
        st.info(f"Model dosyası yükleniyor: {model_path}")
        
        # Birkaç farklı yükleme yöntemi deneyelim
        # Yöntem 1: Standart pickle
        try:
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            st.success("Model başarıyla yüklendi (Pickle)")
        except Exception as e1:
            st.warning(f"Pickle ile yükleme başarısız: {str(e1)}")
            
            # Yöntem 2: Özel encoding ile pickle
            try:
                with open(model_path, 'rb') as file:
                    model_data = pickle.load(file, encoding='latin1')
                st.success("Model başarıyla yüklendi (Latin1 encoding)")
            except Exception as e2:
                st.warning(f"Latin1 encoding ile yükleme başarısız: {str(e2)}")
                
                # Yöntem 3: joblib
                try:
                    import joblib
                    model_data = joblib.load(model_path)
                    st.success("Model başarıyla yüklendi (joblib)")
                except Exception as e3:
                    st.warning(f"Joblib ile yükleme başarısız: {str(e3)}")
                    
                    # Yöntem 4: LightGBM'in kendi load fonksiyonu
                    try:
                        import lightgbm as lgb
                        model_data = lgb.Booster(model_file=model_path)
                        st.success("Model başarıyla yüklendi (LightGBM)")
                    except Exception as e4:
                        st.error(f"Tüm yükleme yöntemleri başarısız oldu. Son hata: {str(e4)}")
                        return None, None, None, None
        
        # Log model info for debugging
        st.write(f"Model tipi: {type(model_data)}")
        
        # The model file can be just the model or a dictionary with model and preprocessors
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            encoders = model_data.get('encoders', {})
            feature_names = model_data.get('feature_names', [])
            return model, scaler, encoders, feature_names
        else:
            # LightGBM Booster nesnesi durumu
            if 'lightgbm.basic.Booster' in str(type(model_data)):
                feature_names = model_data.feature_name() if hasattr(model_data, 'feature_name') else None
                return model_data, None, None, feature_names
            # Sadece model nesnesi kaydedildiyse
            return model_data, None, None, None
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None, None

# Function to make predictions
def predict_loan_approval(model, scaler, encoders, input_data, feature_names=None):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Debug infosu
        st.subheader("Model Bilgileri (Debug)")
        st.write("Model türü:", type(model).__name__)
        
        # Bazı model türleri için farklı işlemler gerekebilir
        is_catboost = 'catboost' in str(type(model)).lower()
        is_lightgbm = 'lightgbm' in str(type(model)).lower() or 'lgb' in str(type(model)).lower() or 'booster' in str(type(model)).lower()
        
        with st.expander("Girdi verileri (Debug)"):
            st.dataframe(df)
        
        # LightGBM modeli tespit edildi
        if is_lightgbm:
            st.write("LightGBM modeli tespit edildi.")
            
            # LightGBM için özellik isimleri
            if hasattr(model, 'feature_name'):
                lgb_feature_names = model.feature_name()
                st.write(f"LightGBM model özellik isimleri: {lgb_feature_names}")
                if lgb_feature_names:
                    feature_names = lgb_feature_names
            
            # Özellik düzenlemeleri
            if feature_names:
                st.write(f"Kullanılacak özellik isimleri: {feature_names}")
                
                # Gerekli tüm özellikler mevcut mu kontrol et
                missing_features = [f for f in feature_names if f not in df.columns]
                if missing_features:
                    st.warning(f"Eksik özellikler: {missing_features}")
                    # Eksik özellikleri 0 ile doldur
                    for feat in missing_features:
                        df[feat] = 0
                
                # Fazla özellikleri kaldır
                extra_features = [f for f in df.columns if f not in feature_names]
                if extra_features:
                    st.info(f"Kullanılmayacak fazla özellikler: {extra_features}")
                    df = df[feature_names]
                
        # CatBoost modeli tespit edildi
        elif is_catboost:
            st.write("CatBoost modeli tespit edildi.")
            
            # CatBoost için kategorik özellikleri kontrol et
            if hasattr(model, 'get_cat_feature_indices'):
                cat_features = model.get_cat_feature_indices()
                st.write(f"Kategorik özellik indeksleri: {cat_features}")
            
            # CatBoost için özellik isimleri
            if hasattr(model, 'feature_names_'):
                st.write(f"Model özellik isimleri: {model.feature_names_}")
        
        # Preprocess the data if we have preprocessors
        if scaler is not None and encoders is not None and not (is_catboost or is_lightgbm):
            # LightGBM/CatBoost özellikle encode edilmiş verileri istemeyebilir
            # Handle categorical features with encoders
            for col, encoder in encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col])
            
            # Scale numerical features
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty and scaler is not None:
                df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        # Ensure the features are in the right order if feature_names is provided
        if feature_names and not (is_lightgbm and hasattr(model, 'feature_name')):
            try:
                df = df[feature_names]
                with st.expander("İşlenmiş girdi verileri (Debug)"):
                    st.dataframe(df)
            except KeyError as e:
                st.error(f"Özellik isimleri uyuşmuyor: {str(e)}")
                st.write(f"Model özellik isimleri: {feature_names}")
                st.write(f"Girdi özellikleri: {df.columns.tolist()}")
                # Try to continue without filtering
        
        # Make prediction based on model type
        try:
            if is_lightgbm:
                # LightGBM için özel tahmin yöntemi
                st.write("LightGBM modeli ile tahmin yapılıyor...")
                
                # LightGBM Booster nesnesi
                if hasattr(model, 'predict'):
                    # Veriyi doğru formata dönüştür
                    try:
                        # NumPy array'e dönüştür
                        X = df.values
                        prediction = model.predict(X)
                    except Exception as e:
                        st.warning(f"NumPy array ile tahmin hatası: {str(e)}")
                        try:
                            # DataFrame olarak dene
                            prediction = model.predict(df)
                        except Exception as e2:
                            st.warning(f"DataFrame ile tahmin hatası: {str(e2)}")
                            # Başka bir format dene
                            from scipy.sparse import csr_matrix
                            X_sparse = csr_matrix(df.values)
                            prediction = model.predict(X_sparse)
                            
                # İlk tahmin değerini al
                if isinstance(prediction, np.ndarray) and prediction.size > 0:
                    prediction = prediction[0]
                
                # Binary sınıflandırma için 0.5 eşiğini uygula
                is_regression = True
                if 0 <= prediction <= 1:
                    is_regression = False
                    binary_prediction = 1 if prediction >= 0.5 else 0
                    st.write(f"Tahmini olasılık: {prediction:.4f}, Binary tahmin: {binary_prediction}")
                    prediction = binary_prediction
                
            elif is_catboost:
                # CatBoost için özel tahmin yöntemi
                prediction = model.predict(df)
                if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
                    prediction = prediction[0]
                
                # Eğer prediction bir numpy array değilse veya sayı değilse
                if not isinstance(prediction, (int, float, np.number)):
                    st.write(f"Tahmin tipi: {type(prediction)}")
                    if hasattr(prediction, 'shape'):
                        st.write(f"Tahmin şekli: {prediction.shape}")
                    # Basit bir dönüşüm dene
                    try:
                        prediction = float(prediction)
                    except:
                        prediction = 1 if prediction > 0.5 else 0
            else:
                # Genel model tahmini
                prediction = model.predict(df)[0]
            
            # Get prediction probability if available
            try:
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(df)[0]
                    prob_value = probability[1] if len(probability) > 1 else probability[0]
                elif is_lightgbm and not is_regression:
                    # LightGBM için olasılık değeri
                    prob_value = prediction
                elif is_catboost and hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(df)[0]
                    prob_value = probability[1] if len(probability) > 1 else probability[0]
                else:
                    prob_value = None
            except Exception as e:
                st.warning(f"Olasılık hesaplanamadı: {str(e)}")
                prob_value = None
            
            # Try to get feature importance if available
            try:
                if is_lightgbm and hasattr(model, 'feature_importance'):
                    importances = model.feature_importance()
                    col_names = feature_names if feature_names else df.columns
                    feature_importance = dict(zip(col_names[:len(importances)], importances))
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    col_names = feature_names if feature_names else df.columns
                    feature_importance = dict(zip(col_names, importances))
                elif hasattr(model, 'get_feature_importance') and is_catboost:
                    importances = model.get_feature_importance()
                    col_names = feature_names if feature_names else df.columns
                    feature_importance = dict(zip(col_names[:len(importances)], importances))
                elif hasattr(model, 'coef_'):
                    importances = model.coef_[0]
                    feature_importance = dict(zip(df.columns, importances))
                else:
                    feature_importance = None
            except Exception as e:
                st.error(f"Özellik önemi hesaplanırken hata: {str(e)}")
                feature_importance = None
            
            return prediction, prob_value, feature_importance
        except Exception as e:
            st.error(f"Tahmin yapılırken hata: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None, None
            
    except Exception as e:
        st.error(f"Tahmin işleminde hata oluştu: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Load the model and preprocessors
model, scaler, encoders, feature_names = load_model()

if model is None:
    st.warning("Model dosyası bulunamadı. Lütfen model dosyasının doğru konumda olduğundan emin olun.")
    st.info("Örnek değerlerle devam ediliyor...")
    
    # Create dummy model for demo purposes
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    
    # Dummy data for demonstration
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    feature_names = ['loan_amount', 'interest_rate', 'income', 'debt_to_income', 'credit_score']
    
    # Train the model
    model.fit(X, y)
else:
    # Model yükleme başarılı oldu, bilgi vererek devam edelim
    st.success("Model başarıyla yüklendi!")
    
    # CatBoost modeli ise özel işlemler
    is_catboost = 'catboost' in str(type(model)).lower()
    
    if is_catboost:
        st.write("CatBoost modeli kullanılıyor.")
        
        # CatBoost'un catboost modülünü import etmeye çalış
        try:
            import catboost
            st.write("CatBoost kütüphanesi başarıyla yüklendi.")
        except ImportError:
            st.warning("CatBoost kütüphanesi bulunamadı. Model yine de çalışabilir, ancak bazı fonksiyonlar kullanılamayabilir.")
        
        # CatBoost özelliklerini kontrol et
        with st.expander("CatBoost Model Detayları"):
            if hasattr(model, 'get_params'):
                st.write("Model parametreleri:", model.get_params())
            
            # Özellik isimleri
            if hasattr(model, 'feature_names_'):
                st.write("Model özellik isimleri:", model.feature_names_)
                feature_names = model.feature_names_  # Modeldeki özellik isimlerini kullan
            elif feature_names:
                st.write("Yüklenen özellik isimleri:", feature_names)
            else:
                st.warning("Özellik isimleri bulunamadı.")
            
            # Kategorik özellikler
            if hasattr(model, 'get_cat_feature_indices'):
                st.write("Kategorik özellik indeksleri:", model.get_cat_feature_indices())

# Input form for user to enter loan details
st.markdown('<p class="sub-header">Kredi Başvuru Detayları</p>', unsafe_allow_html=True)

with st.form("loan_prediction_form"):
    # Create columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Kişisel bilgiler
        person_age = st.slider("Yaş", min_value=18, max_value=80, value=35, step=1)
        person_income = st.number_input("Yıllık Gelir (₺)", min_value=10000, max_value=1000000, value=60000, step=5000)
        person_home_ownership = st.selectbox("Ev Sahipliği Durumu", options=["OWN", "MORTGAGE", "RENT", "OTHER"])
        person_emp_length = st.slider("Çalışma Süresi (Yıl)", min_value=0, max_value=30, value=5, step=1)
        cb_person_cred_hist_length = st.slider("Kredi Geçmişi Uzunluğu (Yıl)", min_value=0, max_value=40, value=10, step=1)
        
    with col2:
        # Kredi bilgileri
        loan_intent = st.selectbox("Kredi Amacı", options=[
            "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
        ])
        loan_grade = st.selectbox("Kredi Notu", options=["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Kredi Miktarı (₺)", min_value=1000, max_value=100000, value=20000, step=1000)
        loan_int_rate = st.slider("Faiz Oranı (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
        loan_percent_income = st.slider("Kredi/Gelir Oranı (%)", min_value=1, max_value=100, value=int((20000/60000)*100), step=1)
    
    # Form submission button
    submitted = st.form_submit_button("Kredi Tahminini Yap")

# Process form submission
if submitted:
    # Kategorik değişkenler için one-hot encoding uygulama
    # 1. Home Ownership için one-hot encoding
    home_ownership_ohe = {
        'person_home_ownership_OWN': 1 if person_home_ownership == "OWN" else 0,
        'person_home_ownership_MORTGAGE': 1 if person_home_ownership == "MORTGAGE" else 0,
        'person_home_ownership_RENT': 1 if person_home_ownership == "RENT" else 0,
        'person_home_ownership_OTHER': 1 if person_home_ownership == "OTHER" else 0
    }
    
    # 2. Loan Intent için one-hot encoding
    loan_intent_ohe = {
        'loan_intent_EDUCATION': 1 if loan_intent == "EDUCATION" else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == "MEDICAL" else 0,
        'loan_intent_VENTURE': 1 if loan_intent == "VENTURE" else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == "PERSONAL" else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == "DEBTCONSOLIDATION" else 0
    }
    
    # 3. Loan Grade için one-hot encoding
    loan_grade_ohe = {
        'loan_grade_A': 1 if loan_grade == "A" else 0,
        'loan_grade_B': 1 if loan_grade == "B" else 0,
        'loan_grade_C': 1 if loan_grade == "C" else 0,
        'loan_grade_D': 1 if loan_grade == "D" else 0,
        'loan_grade_E': 1 if loan_grade == "E" else 0,
        'loan_grade_F': 1 if loan_grade == "F" else 0,
        'loan_grade_G': 1 if loan_grade == "G" else 0
    }
    
    # Numeric değişkenler
    numeric_features = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }
    
    # Tüm özellikleri birleştir
    input_data = {**numeric_features, **home_ownership_ohe, **loan_intent_ohe, **loan_grade_ohe}
    
    # Debug için
    with st.expander("One-Hot Encoded Değişkenler"):
        st.write(pd.DataFrame([input_data]))
    
    # CatBoost modeli için eksik özellikleri ekle
    if 'catboost' in str(type(model)).lower() and hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
        for feat in model_features:
            if feat not in input_data:
                # Formdan eksik özellikleri topla
                if f"extra_{feat}" in st.session_state:
                    try:
                        value = float(st.session_state[f"extra_{feat}"])
                    except:
                        value = st.session_state[f"extra_{feat}"]
                    input_data[feat] = value
    
    # Make prediction
    with st.spinner("Tahmin yapılıyor..."):
        prediction, probability, feature_importance = predict_loan_approval(model, scaler, encoders, input_data, feature_names)
    
    # Display prediction
    st.markdown('<p class="sub-header">Tahmin Sonucu</p>', unsafe_allow_html=True)
    
    # Show the prediction result
    if prediction is not None:
        # CatBoost için tahmin işleme
        if 'catboost' in str(type(model)).lower():
            # Eğer prediction sayısal değilse
            try:
                prediction_value = float(prediction)
                is_approved = prediction_value > 0.5
            except:
                # Eğer sayısal dönüşüm yapılamazsa
                if isinstance(prediction, (str, np.str_)):
                    is_approved = prediction.lower() in ['1', 'true', 'yes', 'approved']
                else:
                    is_approved = bool(prediction)
        else:
            is_approved = prediction == 1
        
        if is_approved:
            st.markdown('<div class="prediction-box approved">', unsafe_allow_html=True)
            st.markdown('### ✅ Kredi Onaylanabilir')
        else:
            st.markdown('<div class="prediction-box rejected">', unsafe_allow_html=True)
            st.markdown('### ❌ Kredi Onaylanmayabilir')
        
        # Display probability if available
        if probability is not None:
            st.write(f"Onay Olasılığı: {probability:.2%}")
        else:
            # Eğer olasılık değeri yoksa ve CatBoost ise
            if 'catboost' in str(type(model)).lower():
                try:
                    prob = float(prediction)
                    if 0 <= prob <= 1:
                        st.write(f"Tahmini Olasılık: {prob:.2%}")
                except:
                    pass
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display feature importance if available
        if feature_importance:
            st.markdown('<p class="sub-header">Özellik Önemi</p>', unsafe_allow_html=True)
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            
            # Sort features by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
            
            # Create a bar chart of feature importance
            importance_df = pd.DataFrame({'Özellik': sorted_importance.keys(), 'Önem': sorted_importance.values()})
            
            # Display the chart
            st.bar_chart(importance_df.set_index('Özellik'))
            
            # Display text explanation
            st.write("Kredi kararında en etkili faktörler:")
            for i, (feature, importance) in enumerate(list(sorted_importance.items())[:5]):
                st.write(f"{i+1}. {feature}: {abs(importance):.4f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Provide some explanation
        st.markdown('<p class="sub-header">Nasıl Geliştirilebilir?</p>', unsafe_allow_html=True)
        
        if not is_approved:
            st.write("Kredi onay şansınızı artırmak için şunları göz önünde bulundurabilirsiniz:")
            suggestions = []
            
            if person_age < 30:
                suggestions.append("- Yaşınız kredi değerlendirmesinde bir faktördür. Yaş ilerledikçe kredi onayı olasılığı genellikle artar.")
            
            if person_income < 50000:
                suggestions.append("- Daha yüksek gelir, kredi geri ödeme kapasitenizi artırır ve kredi onayı şansınızı yükseltir.")
            
            if person_home_ownership == "RENT":  # Orijinal değeri kullanıyoruz
                suggestions.append("- Ev sahibi olmak, finansal istikrar göstergesi olarak değerlendirilebilir.")
            
            if loan_grade in ["E", "F", "G"]:  # Orijinal değeri kullanıyoruz
                suggestions.append("- Daha düşük bir kredi notu (E, F, G) risk faktörünü artırır. Kredi skorunuzu iyileştirin.")
                
            if loan_amnt > person_income / 2:
                suggestions.append("- Gelir seviyenize göre daha düşük kredi tutarı talep etmek onay şansınızı artırabilir.")
                
            if loan_int_rate > 15:
                suggestions.append("- Yüksek faiz oranları genellikle yüksek risk göstergesidir. Daha düşük faiz oranları araştırın.")
                
            if loan_intent in ["MEDICAL", "DEBTCONSOLIDATION"]:  # Orijinal değeri kullanıyoruz
                suggestions.append("- Bazı kredi amaçları (tıbbi, borç birleştirme) diğerlerine göre daha riskli olarak değerlendirilebilir.")
            
            if loan_percent_income > 40:
                suggestions.append("- Kredi miktarınız gelirinizin yüksek bir yüzdesini oluşturuyor. Kredi miktarını azaltmak veya gelirinizi artırmak riski düşürebilir.")
            
            if cb_person_cred_hist_length < 5:
                suggestions.append("- Kısa kredi geçmişi, risk değerlendirmesini etkileyebilir. Daha uzun bir kredi geçmişi oluşturmak için çalışın.")
            
            # Add generic suggestions if the list is still empty
            if not suggestions:
                suggestions.extend([
                    "- Gelir belgelendirmenizi güçlendirmek",
                    "- Ek teminat sunmak",
                    "- Kredi geçmişinizi iyileştirmek"
                ])
            
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.write("Kredi başvurunuz onay alma olasılığına sahiptir. Yine de şunlara dikkat etmeniz önerilir:")
            st.write("- Kredi koşullarını ve geri ödeme planını dikkatlice incelemek")
            st.write("- Aylık taksitlerin bütçenizi zorlamadığından emin olmak")
            st.write("- Geri ödeme planına sadık kalmak için bir bütçe oluşturmak")

# Information about the model
with st.expander("Model Hakkında Bilgi"):
    st.write("""
    Bu tahmin modeli, geçmiş kredi verileri üzerinde eğitilmiş bir makine öğrenmesi algoritmasını kullanmaktadır. 
    Model, girilen bilgilere dayanarak kredi onay olasılığını tahmin eder.
    
    Not: Bu model sadece tahmini bir sonuç verir ve gerçek kredi kararı için resmi bir başvuru yapılması gerekir.
    Gerçek kredi değerlendirmeleri, burada kullanılandan çok daha fazla faktörü göz önünde bulundurur.
    """)

# Footer
st.markdown("---")
st.info("Bu araç, kredi tahmin modellerini test etmek ve anlamak için tasarlanmıştır. Gerçek finansal kararlar için profesyonel danışmanlık alınması önerilir.") 