import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")
#st.set_option('deprecation.showPyplotGlobalUse', False)


MODEL_DIR = Path(__file__).resolve().parent / "model"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "train_data_for_eda.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "model_feature.pkl"
ENCODER_PATH = MODEL_DIR / "one_hot_encoder.pkl"


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""
    model = Ridge()
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

@st.cache_resource
def load_encoder():
    encoder = OneHotEncoder(drop = 'first', sparse_output = False)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def prepare_data(df):
    category_columns_list = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    y_fact = df['selling_price']
    X_pred = df.drop(columns=['selling_price'])
    onehot_encoder = load_encoder()
    encoded_df = pd.DataFrame(onehot_encoder.transform(X_pred[category_columns_list]))
    encoded_df.columns = onehot_encoder.get_feature_names_out()

    X_test_cat_encoded = X_pred.join(encoded_df)
    X_test_cat_encoded.drop(columns=category_columns_list, inplace=True)

    return X_test_cat_encoded, y_fact


# --- Визуализации ---
st.header("📈 Визуализации")

#загружаем данные
st.subheader("Графики попарных распределений числовых значений")
train = pd.read_csv(DATA_PATH, index_col=0)
pair_plot = sns.pairplot(train,
             vars=train.columns,
             palette='Set1',
             plot_kws={'alpha': 0.8})

plt.suptitle('Pairplot train data set', y=1.02, fontsize=16)
st.pyplot(pair_plot.figure, clear_figure=True)

st.subheader("Тепловая карта коэффициента Пирсона")
correlation_matrix_train = train.corr()
heatmap = sns.heatmap(correlation_matrix_train, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, fmt='.2f')
st.pyplot(heatmap.get_figure(), clear_figure=True)

# Загружаем модель
try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

coef = MODEL.coef_
st.write(f"Давайте посмотрим, что же за коэффициенты у модели?")
fig2 = px.bar(pd.DataFrame([MODEL.coef_.tolist()], columns=FEATURE_NAMES.tolist()),  title="Коэффициенты модели")
st.plotly_chart(fig2, use_container_width=True)

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file, index_col=0)

try:
    data, y_test = prepare_data(df)
    y_predict = MODEL.predict(data)
    r2 = r2_score(y_test, y_predict)

    st.write(f"Получили метрику r2 для загруженного файла {r2:.2f}")


except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()
