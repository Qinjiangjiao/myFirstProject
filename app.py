import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============ 页面配置 ============
st.set_page_config(
    page_title="AHR Nomogram v2026",
    layout="centered",
    page_icon="🫁"
)

# ============ 核心参数 ============
# 根据 2026.1.5 统计结果校准
B0 = 0.87  # 截距 (Intercept)
COEFFICIENTS = {
    "FeNO": 0.03,
    "Age": -0.04,
    "FamilyHistory": 1.36,
    "Rhinitis": 0.53,
    "Allergy": 0.85,
    "TPTEF_TE": -0.03,
    "Wheeze": 1.35
}

# ============ 多语言内容 ============
CONTENT = {
    "English": {
        "title": "AHR Prediction Nomogram (7-Parameter Model)",
        "desc": "Predicting Airway Hyperresponsiveness in Preschool Children",
        "prob_text": "Predicted Probability of AHR:",
        "risk_level": "Risk Level:",
        "levels": ["Low", "Moderate", "High"],
        "labels": ["FeNO (ppb)", "Age (Months)", "TPTEF/TE (%)", "Wheeze", 
                   "Family History of Asthma", "History of Rhinitis", "History of Allergy"],
        "footer": "<b>Jiangjiao Qin</b>, et al., Children's Hospital of Chongqing Medical University"
    },
    "中文": {
        "title": "气道高反应性预测列线图 (7参数模型)",
        "desc": "学龄前哮喘儿童气道高反应性风险评估",
        "prob_text": "预测 AHR 概率：",
        "risk_level": "风险等级：",
        "levels": ["低风险", "中等风险", "高风险"],
        "labels": ["FeNO (ppb)", "月龄 (Months)", "TPTEF/TE (%)", "当前喘息", 
                   "哮喘家族史", "鼻炎史", "过敏史"],
        "footer": "<b>秦江蛟</b> 等，重庆医科大学附属儿童医院"
    }
}

lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"])
c = CONTENT[lang]

# ============ 界面渲染 ============
st.markdown(f"<h2 style='text-align:center;'>{c['title']}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:gray;'>{c['desc']}</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    feno = st.number_input(c['labels'][0], 0, 200, 25)
    age = st.number_input(c['labels'][1], 0, 48, 34)
    tptef = st.number_input(c['labels'][2], 5, 60, 23)

with col2:
    wheeze = st.selectbox(c['labels'][3], ["No/否", "Yes/是"])
    fam = st.selectbox(c['labels'][4], ["No/否", "Yes/是"])
    rhinitis = st.selectbox(c['labels'][5], ["No/否", "Yes/是"])
    allergy = st.selectbox(c['labels'][6], ["No/否", "Yes/是"])

# ============ 计算逻辑 ============
# 转换分类变量
val_w = 1 if "Yes" in wheeze else 0
val_f = 1 if "Yes" in fam else 0
val_r = 1 if "Yes" in rhinitis else 0
val_a = 1 if "Yes" in allergy else 0

# 构建 Logit 公式 
logit_p = (B0 + 
           COEFFICIENTS["FeNO"] * feno + 
           COEFFICIENTS["Age"] * age + 
           COEFFICIENTS["FamilyHistory"] * val_f + 
           COEFFICIENTS["Rhinitis"] * val_r + 
           COEFFICIENTS["Allergy"] * val_a + 
           COEFFICIENTS["TPTEF_TE"] * tptef + 
           COEFFICIENTS["Wheeze"] * val_w)

p = 1 / (1 + np.exp(-logit_p))

# ============ 结果展示 ============
st.markdown("---")
st.markdown(f"### {c['prob_text']} **{p*100:.1f}%**")

if p < 0.3:
    st.success(f"**{c['risk_level']}** {c['levels'][0]}")
elif p < 0.7:
    st.warning(f"**{c['risk_level']}** {c['levels'][1]}")
else:
    st.error(f"**{c['risk_level']}** {c['levels'][2]}")

# 风险梯度条
fig, ax = plt.subplots(figsize=(6, 0.6))
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap="RdYlGn_r", extent=[0, 100, 0, 1])
ax.axvline(p*100, color='black', linestyle='--', linewidth=2)
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel(f"{c['prob_text']} (%)", fontsize=10)
st.pyplot(fig)

st.markdown(f"<br><p style='color:gray; font-size:12px; text-align:center;'>{c['footer']}</p>", unsafe_allow_html=True)

