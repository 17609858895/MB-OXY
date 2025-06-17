import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="MB Degradation Prediction",
    layout="centered"
)

# 🌿 样式设置
st.markdown("""
    <style>
    .stApp {
        max-width: 700px;
        margin: auto;
        background-color: #e0f7fa;  /* 浅蓝色背景 */
        padding: 2rem;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .custom-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #222;
    }
    .stMarkdown h1 + p {
        font-size: 1.02rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stNumberInput label {
        font-size: 0.98rem;
        font-weight: 500;
        color: #333;
    }
    .stButton>button {
        background-color: #0288d1;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        margin-top: 1rem;
    }
    .stDownloadButton>button {
        background-color: white;
        color: #333;
        font-weight: 500;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
    }
    .stSuccess {
        background-color: #b3e5fc;
        color: #01579b;
        padding: 0.85rem;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型
def load_model():
    return joblib.load("gbtboost.pkl")

model = load_model()

# 🌐 语言切换
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

# 文本包
text = {
    "English": {
        "title": "🔬 ML prediction of MB degradation via advanced oxidation",
        "description": "This app predicts the methylene blue degradation under specified experimental conditions.",
        "input_labels": [
            "🌡 Reaction temperature (K)",
            "💧 MB concentration (mg/L)",
            "⚗️ Oxidant concentration (mmol/L)",
            "🧪 Catalyst dosage (g/L)",
            "⏱ Reaction time (min)",
            "🌡 pH value"
        ],
        "button_predict": "🔍 Predict Degradation",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted MB degradation",
        "result_unit": "%",
        "file_name": "prediction_result.csv"
    },
    "中文": {
        "title": "🔬 高级氧化降解亚甲蓝的机器学习预测",
        "description": "本应用基于实验条件预测亚甲蓝的降解效果。",
        "input_labels": [
            "🌡 反应温度 (K)",
            "💧 亚甲蓝浓度 (mg/L)",
            "⚗️ 氧化剂浓度 (mmol/L)",
            "🧪 催化剂用量 (g/L)",
            "⏱ 反应时间 (min)",
            "🌡 溶液 pH"
        ],
        "button_predict": "🔍 预测降解率",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的亚甲蓝降解率",
        "result_unit": "%",
        "file_name": "预测结果.csv"
    }
}[lang]

# 标题 + 描述
st.markdown(f'<div class="custom-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(text["description"])

# 输入字段
temp_k       = st.number_input(text["input_labels"][0], min_value=0.0, value=298.0, step=1.0)
mb_conc      = st.number_input(text["input_labels"][1], min_value=0.0, value=50.0, step=1.0)
oxidant_conc = st.number_input(text["input_labels"][2], min_value=0.0, value=10.0, step=0.1)
catalyst_dos = st.number_input(text["input_labels"][3], min_value=0.0, value=0.5, step=0.1)
react_time   = st.number_input(text["input_labels"][4], min_value=0.0, value=60.0, step=1.0)
pH_val       = st.number_input(text["input_labels"][5], min_value=1.0, max_value=14.0, value=7.0, step=0.1)

# 预测结果
prediction = None
df_result = None

if st.button(text["button_predict"]):
    input_data = np.array([[temp_k, mb_conc, oxidant_conc, catalyst_dos, react_time, pH_val]])
    prediction = model.predict(input_data)[0]
    # 显示带百分号的结果
    st.success(f"{text['result_prefix']}: **{prediction:.2f}{text['result_unit']}**")

    # 结果表格带单位列标题
    df_result = pd.DataFrame([{  
        text["input_labels"][0]: temp_k,
        text["input_labels"][1]: mb_conc,
        text["input_labels"][2]: oxidant_conc,
        text["input_labels"][3]: catalyst_dos,
        text["input_labels"][4]: react_time,
        text["input_labels"][5]: pH_val,
        f"{text['result_prefix']} (%)": round(prediction, 2)
    }])

# 导出 CSV
if prediction is not None and df_result is not None:
    towrite = BytesIO()
    df_result.to_csv(towrite, index=False)
    st.download_button(
        label=text["button_export"],
        data=towrite.getvalue(),
        file_name=text["file_name"],
        mime="text/csv"
    )
