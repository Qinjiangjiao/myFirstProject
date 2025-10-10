
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============ 页面配置 ============
st.set_page_config(
    page_title="AHR Nomogram",
    layout="centered",
    page_icon="🫁"
)

# ============ 多语言切换 ============
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"])

APPLE_CMAP = "RdYlGn_r"  # Apple极简风格配色：低风险绿，高风险红

# ============ 英文界面 ============
if lang == "English":
    st.markdown("<h2 style='text-align:center;'>Nomogram for Predicting Airway Hyperresponsiveness (AHR)</h2>", unsafe_allow_html=True)
    st.markdown("Enter the values below to estimate the probability of AHR.")

    FeNO = st.number_input("FeNO (ppb)", min_value=0, max_value=200, value=20, step=1)
    RR = st.number_input("Respiratory Rate (bpm)", min_value=10, max_value=80, value=25, step=1)
    PTEF = st.number_input("PTEF/TEF25 (%)", min_value=40, max_value=350, value=150, step=1)
    Wheeze = st.selectbox("Wheeze", options=["No", "Yes"])

    # 模型参数
    b0 = -10
    b1, b2, b3, b4 = 0.06, 0.09, 0.01, 1.80
    wheeze_val = 1 if Wheeze == "Yes" else 0

    logit_p = b0 + b1*FeNO + b2*RR + b3*PTEF + b4*wheeze_val
    logit_p = np.clip(logit_p, -50, 50)
    p = 1 / (1 + np.exp(-logit_p))
    p = float(p)

    st.markdown(f"### Predicted Probability of AHR: **{p*100:.1f}%**")

    if p < 0.3:
        risk_level = "Low risk"
    elif p < 0.7:
        risk_level = "Moderate risk"
    else:
        risk_level = "High risk"
    st.info(f"**Risk Level:** {risk_level}")

    # 风险条
    fig, ax = plt.subplots(figsize=(6, 0.6))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=APPLE_CMAP, extent=[0, 100, 0, 1])
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks([])
    ax.set_xlabel("AHR Risk (%)", fontsize=10)
    ax.axvline(p*100, color='black', linestyle='--', linewidth=2)
    ax.text(p*100, 1.1, f"{p*100:.1f}%", ha='center', va='bottom', fontsize=10, color='black')
    st.pyplot(fig)

    # 页脚
    st.markdown(
        """
        <hr style="margin-top:30px;margin-bottom:10px;">
        <p style="color:gray; font-size:13px; text-align:center;">
        Predicting Airway Hyperresponsiveness in Preschool Asthma: A Nomogram Based on FeNO and Tidal Breathing Parameters<br>
        <b>Jiangjiao Qin</b>, et al., Children's Hospital of Chongqing Medical University
        </p>
        """,
        unsafe_allow_html=True
    )

# ============ 中文界面 ============
else:
    st.markdown("<h2 style='text-align:center;'>预测气道高反应性的列线图 (AHR Nomogram)</h2>", unsafe_allow_html=True)
    st.markdown("请输入以下参数以估算气道高反应性的概率：")

    FeNO = st.number_input("FeNO (ppb)", min_value=0, max_value=200, value=20, step=1)
    RR = st.number_input("呼吸频率 (次/分)", min_value=10, max_value=80, value=25, step=1)
    PTEF = st.number_input("PTEF/TEF25 (%)", min_value=40, max_value=350, value=150, step=1)
    Wheeze = st.selectbox("是否存在喘息", options=["否", "是"])

    # 模型参数
    b0 = -10
    b1, b2, b3, b4 = 0.06, 0.09, 0.01, 1.80
    wheeze_val = 1 if Wheeze == "是" else 0

    logit_p = b0 + b1*FeNO + b2*RR + b3*PTEF + b4*wheeze_val
    logit_p = np.clip(logit_p, -50, 50)
    p = 1 / (1 + np.exp(-logit_p))
    p = float(p)

    st.markdown(f"### 预测AHR概率：**{p*100:.1f}%**")

    if p < 0.3:
        risk_level = "低风险"
    elif p < 0.7:
        risk_level = "中等风险"
    else:
        risk_level = "高风险"
    st.info(f"**风险等级：** {risk_level}")

    fig, ax = plt.subplots(figsize=(6, 0.6))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=APPLE_CMAP, extent=[0, 100, 0, 1])
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks([])
    ax.set_xlabel("AHR风险（%）", fontsize=10)
    ax.axvline(p*100, color='black', linestyle='--', linewidth=2)
    ax.text(p*100, 1.1, f"{p*100:.1f}%", ha='center', va='bottom', fontsize=10, color='black')
    st.pyplot(fig)

    st.markdown(
        """
        <hr style="margin-top:30px;margin-bottom:10px;">
        <p style="color:gray; font-size:13px; text-align:center;">
        学龄前哮喘儿童气道高反应性预测模型：基于FeNO与潮气呼吸参数的列线图<br>
        <b>秦江蛟</b> 等，重庆医科大学附属儿童医院
        </p>
        """,
        unsafe_allow_html=True
    )

# ============ 页面CSS微调 ============
st.markdown(
    """
    <style>
    .block-container {
        max-width: 700px;
        margin: auto;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
