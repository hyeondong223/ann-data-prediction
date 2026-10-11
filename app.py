
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="인공신경망 데이터 예측",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def train_model():
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return data, model, X_test, y_test, predictions, metrics

st.title("🧠 인공신경망 기반 데이터 예측")
st.write(
    "공개 당뇨병 데이터셋을 이용해 다층 퍼셉트론 "
    "(MLP) 회귀 모델의 예측 과정을 체험하는 교육용 웹사이트입니다."
)

st.info(
    "주의: 이 프로그램은 교육 및 알고리즘 실습용입니다. "
    "의료 진단이나 치료에 사용하지 마세요."
)

with st.spinner("공개 데이터를 불러오고 인공신경망을 학습하고 있습니다..."):
    data, model, X_test, y_test, predictions, metrics = train_model()

st.subheader("1. 모델 성능")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['MAE']:.2f}")
c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
c3.metric("R²", f"{metrics['R2']:.3f}")

st.caption(
    "MAE와 RMSE는 작을수록 오차가 작으며, "
    "R²는 테스트 데이터에서 모델의 설명력을 나타냅니다."
)

st.subheader("2. 입력 변수 설정")
st.write(
    "각 변수의 값을 조정한 뒤 예측 버튼을 누르세요. "
    "입력 범위는 공개 데이터셋의 관측 범위를 기준으로 합니다."
)

feature_names = {
    "age": "나이 관련 지표 (age)",
    "sex": "성별 관련 지표 (sex)",
    "bmi": "체질량지수 관련 지표 (bmi)",
    "bp": "혈압 관련 지표 (bp)",
    "s1": "혈청 측정값 1 (s1)",
    "s2": "혈청 측정값 2 (s2)",
    "s3": "혈청 측정값 3 (s3)",
    "s4": "혈청 측정값 4 (s4)",
    "s5": "혈청 측정값 5 (s5)",
    "s6": "혈청 측정값 6 (s6)"
}

with st.form("prediction_form"):
    values = {}

    cols = st.columns(2)

    for i, feature in enumerate(data.data.columns):
        col = cols[i % 2]
        with col:
            values[feature] = st.slider(
                feature_names[feature],
                min_value=float(data.data[feature].min()),
                max_value=float(data.data[feature].max()),
                value=float(data.data[feature].median()),
                help="공개 데이터에 기록된 표준화된 변수입니다."
            )

    submitted = st.form_submit_button(
        "예측하기",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([values], columns=data.data.columns)
    result = float(model.predict(input_df)[0])

    st.subheader("3. 예측 결과")
    st.metric("예측된 목표 수치", f"{result:.2f}")
    st.write(
        "이 수치는 모델이 입력 변수로부터 예측한 값이며, "
        "개인의 실제 건강 상태를 진단하는 결과가 아닙니다."
    )

st.subheader("4. 실제값과 예측값 비교")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, predictions, alpha=0.7)

low = min(float(y_test.min()), float(predictions.min()))
high = max(float(y_test.max()), float(predictions.max()))

ax.plot([low, high], [low, high], linestyle="--")
ax.set_xlabel("Actual value")
ax.set_ylabel("Predicted value")
ax.set_title("Actual vs Predicted")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.subheader("5. 알고리즘 처리 순서")
st.markdown("""
1. 공개 데이터셋을 불러옵니다.
2. 데이터를 학습용과 테스트용으로 나눕니다.
3. 입력 변수를 표준화합니다.
4. 은닉층 2개를 가진 인공신경망을 학습합니다.
5. 사용자 입력에 대한 예측값을 계산합니다.
6. 테스트 데이터의 성능 지표와 그래프를 표시합니다.
""")

st.caption(
    "데이터 출처: scikit-learn Diabetes Dataset | "
    "모델: MLPRegressor"
)
