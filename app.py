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

# ============ 模型参数（新7参数模型） ============
b0 = 0.87  # 更新后的截距值
b_FeNO = 0.03
b_Age = -0.04
b_Family_History = 1.36
b_Rhinitis = 0.53
b_Allergy = 0.85
b_TPTEF_TE = -0.03
b_Wheeze = 1.35

# ============ 英文界面 ============
if lang == "English":
    st.markdown("<h2 style='text-align:center;'>Nomogram for Predicting Airway Hyperresponsiveness (AHR)</h2>", unsafe_allow_html=True)
    st.markdown("Enter the values below to estimate the probability of AHR.")

    FeNO = st.number_input("FeNO (ppb)", min_value=0, max_value=200, value=20, step=1)
    Age = st.number_input("Age (months)", min_value=0, max_value=48, value=24, step=1)
    TPTEF_TE = st.number_input("TPTEF/TE (%)", min_value=10, max_value=60, value=30, step=1)
    Wheeze = st.selectbox("Wheeze", options=["No", "Yes"])
    Family_History = st.selectbox("Family History of Asthma", options=["No", "Yes"])
    Rhinitis = st.selectbox("History of Rhinitis", options=["No", "Yes"])
    Allergy = st.selectbox("History of Allergy", options=["No", "Yes"])

    # 二元变量转换
    wheeze_val = 1 if Wheeze == "Yes" else 0
    family_val = 1 if Family_History == "Yes" else 0
    rhinitis_val = 1 if Rhinitis == "Yes" else 0
    allergy_val = 1 if Allergy == "Yes" else 0

    # 计算概率
    logit_p = (b0 + b_FeNO * FeNO + b_Age * Age + b_TPTEF_TE * TPTEF_TE +
               b_Wheeze * wheeze_val + b_Family_History * family_val +
               b_Rhinitis * rhinitis_val + b_Allergy * allergy_val)
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

    # 模型性能说明（可展开）
    with st.expander("Model Performance Notes"):
        st.markdown("""
        - (A) Decision curve analysis (DCA) in the training cohort. The Y-axis indicates net benefit. The solid line represents the clinical net benefit of using the prediction model across threshold probabilities. The grey dashed line assumes all patients are positive; the horizontal line assumes none are positive. The model demonstrates clinical utility between 0.25–0.8.
        - (B) DCA in the validation cohort, confirming consistent clinical benefit.
        - (C) Calibration curve of the training cohort. The diagonal dashed line represents the ideal prediction. The red line indicates the apparent performance of the model on the training data, and the blue line shows the bias-corrected performance via bootstrap resampling (1,000 iterations). Hosmer–Lemeshow test: p=0.249.
        - (D) Calibration curve of the validation cohort, with similar annotation as in (C). Hosmer–Lemeshow test: p=0.241.
        """)

    # 页脚
    st.markdown(
        """
        <hr style="margin-top:30px;margin-bottom:10px;">
        <p style="color:gray; font-size:13px; text-align:center;">
        Predicting Airway Hyperresponsiveness in Preschool Asthma: A Nomogram Based on FeNO and Tidal Breathing Parameters<br>
        <b>Jiangjiao Qin</b>, Children's Hospital of Chongqing Medical University
        </p>
        """,
        unsafe_allow_html=True
    )

# ============ 中文界面 ============
else:
    st.markdown("<h2 style='text-align:center;'>预测气道高反应性的列线图 (AHR Nomogram)</h2>", unsafe_allow_html=True)
    st.markdown("请输入以下参数以估算气道高反应性的概率：")

    FeNO = st.number_input("FeNO (ppb)", min_value=0, max_value=200, value=20, step=1)
    Age = st.number_input("年龄 (月)", min_value=0, max_value=48, value=24, step=1)
    TPTEF_TE = st.number_input("TPTEF/TE (%)", min_value=10, max_value=60, value=30, step=1)
    Wheeze = st.selectbox("是否存在喘息", options=["否", "是"])
    Family_History = st.selectbox("哮喘家族史", options=["否", "是"])
    Rhinitis = st.selectbox("鼻炎史", options=["否", "是"])
    Allergy = st.selectbox("过敏史", options=["否", "是"])

    # 二元变量转换
    wheeze_val = 1 if Wheeze == "是" else 0
    family_val = 1 if Family_History == "是" else 0
    rhinitis_val = 1 if Rhinitis == "是" else 0
    allergy_val = 1 if Allergy == "是" else 0

    # 计算概率
    logit_p = (b0 + b_FeNO * FeNO + b_Age * Age + b_TPTEF_TE * TPTEF_TE +
               b_Wheeze * wheeze_val + b_Family_History * family_val +
               b_Rhinitis * rhinitis_val + b_Allergy * allergy_val)
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

    # 风险条
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

    # 模型性能说明（可展开）
    with st.expander("模型性能说明"):
        st.markdown("""
        - (A) 训练队列的决策曲线分析 (DCA)。Y轴表示净收益。实线表示使用预测模型在阈值概率下的临床净收益。灰色虚线假设所有患者均为阳性；水平线假设无一为阳性。该模型在0.25–0.8之间显示临床效用。
        - (B) 验证队列的DCA，确认一致的临床益处。
        - (C) 训练队列的校准曲线。对角虚线表示理想预测。红线表示模型在训练数据上的表观性能，蓝线显示通过bootstrap重采样（1,000次迭代）的偏差校正性能。Hosmer–Lemeshow测试：p=0.249。
        - (D) 验证队列的校准曲线，与(C)类似标注。Hosmer–Lemeshow测试：p=0.241。
        """)

    # 页脚
    st.markdown(
        """
        <hr style="margin-top:30px;margin-bottom:10px;">
        <p style="color:gray; font-size:13px; text-align:center;">
        学龄前哮喘儿童气道高反应性预测模型：基于FeNO与潮气呼吸参数的列线图<br>
        <b>秦江蛟</b>，重庆医科大学附属儿童医院
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
