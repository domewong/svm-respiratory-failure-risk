# app.py
# -*- coding: utf-8 -*-

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# SHAP 依赖较重，放在后面按需 import 也可以
import shap
import matplotlib.pyplot as plt


# ======================
# 1) 页面设置
# ======================
st.set_page_config(
    page_title="Respiratory Failure Risk Calculator (SVM)",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
# 🫁 Respiratory Failure Risk Calculator (SVM)
输入临床变量 → 输出个体风险（概率）+ 单例 SHAP 解释（waterfall）
"""
)
st.info("提示：该工具用于科研展示与辅助决策，不替代临床医生判断。")


# ======================
# 2) 路径与常量
# ======================
BASE_DIR = Path(__file__).resolve().parent  # Streamlit/Cloud 环境可用
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
BG_PATH = BASE_DIR / "shap_background.pkl"

FEATURE_COLS = ["Age", "PaO2", "PF_ratio", "pneumonia", "ISS"]


# ======================
# 3) 加载模型/Scaler/SHAP background
# ======================
@st.cache_resource(show_spinner=True)
def load_assets():
    # 文件存在性检查
    missing = []
    for p in [MODEL_PATH, SCALER_PATH, BG_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    # 读取 pkl
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(BG_PATH, "rb") as f:
        bg = pickle.load(f)

    # background 期望是 (n, 5)
    bg = np.array(bg)
    if bg.ndim != 2 or bg.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"shap_background shape should be (n,{len(FEATURE_COLS)}), got {bg.shape}")

    return model, scaler, bg


def safe_predict_proba(model, X_scaled_df: pd.DataFrame) -> float:
    """返回正类概率"""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba(). 请确认训练时开启 probability=True 的 SVC。")
    proba = model.predict_proba(X_scaled_df)
    return float(proba[0, 1])


def build_kernel_explainer(model, bg_scaled_df: pd.DataFrame):
    """KernelExplainer：用 predict_proba 输出概率，link='logit' 更适合分类概率"""
    # shap 需要函数：输入 numpy -> 输出概率矩阵
    def f(X_np):
        X_df = pd.DataFrame(X_np, columns=FEATURE_COLS)
        return model.predict_proba(X_df)

    explainer = shap.KernelExplainer(f, bg_scaled_df.values, link="logit")
    return explainer


def plot_shap_waterfall(explainer, x_scaled_df: pd.DataFrame, feature_names):
    """
    生成 waterfall 图（matplotlib），返回 fig
    """
    # shap_values: (1, n_features) for class=1
    shap_values = explainer.shap_values(x_scaled_df.values, nsamples=200)
    # 二分类 KernelExplainer 可能返回 list: [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    x_raw = x_scaled_df.iloc[0].values

    # 构造 Explanation
    exp = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=x_raw,
        feature_names=list(feature_names),
    )

    fig = plt.figure(figsize=(7.2, 4.2), dpi=160)
    shap.plots.waterfall(exp, max_display=len(feature_names), show=False)
    plt.tight_layout()
    return fig, sv


# ======================
# 4) 侧边栏输入
# ======================
with st.sidebar:
    st.header("Input features")

    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
    pao2 = st.number_input("PaO₂ (mmHg)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
    pf = st.number_input("PF ratio (PaO₂/FiO₂)", min_value=0.0, max_value=800.0, value=250.0, step=5.0)

    pneumonia = st.selectbox("Pulmonary infection / Pneumonia (0/1)", [0, 1], index=1)
    iss = st.number_input("ISS (Injury Severity Score)", min_value=0.0, max_value=75.0, value=25.0, step=1.0)

    st.divider()

    pt_custom = st.slider("Decision threshold (pt)", min_value=0.05, max_value=0.95, value=0.40, step=0.01)
    st.caption("建议用于论文阈值解释：pt=0.20 / 0.40 / 0.60（三档）")


# ======================
# 5) 主逻辑：加载 + 预测 + SHAP
# ======================
try:
    model, scaler, bg = load_assets()
except Exception as e:
    st.error("模型资源加载失败（请检查 app.py 同目录下的 pkl 文件是否齐全且可读取）")
    st.exception(e)
    st.stop()

# raw 输入
X_raw = pd.DataFrame(
    [[age, pao2, pf, pneumonia, iss]],
    columns=FEATURE_COLS
)

# 标准化
try:
    X_scaled_np = scaler.transform(X_raw.values)
    X_scaled = pd.DataFrame(X_scaled_np, columns=FEATURE_COLS)
except Exception as e:
    st.error("标准化 scaler.transform 失败：请确认 scaler 与特征列顺序一致。")
    st.exception(e)
    st.stop()

# 预测概率
try:
    prob = safe_predict_proba(model, X_scaled)
except Exception as e:
    st.error("predict_proba 失败：请确认你的 SVM 训练时设置了 probability=True，并且模型可正常加载。")
    st.exception(e)
    st.stop()

risk_label = "High risk" if prob >= pt_custom else "Low risk"
cost_benefit = pt_custom / (1 - pt_custom)


# ======================
# 6) 页面布局：两列
# ======================
col1, col2 = st.columns([1.05, 1.0], gap="large")

with col1:
    st.subheader("Prediction")

    st.metric("Predicted risk (probability)", f"{prob:.3f}")

    if prob >= pt_custom:
        st.error(f"Decision (pt={pt_custom:.2f}): {risk_label}")
    else:
        st.success(f"Decision (pt={pt_custom:.2f}): {risk_label}")

    st.caption(f"Cost:Benefit ratio = pt/(1-pt) = {cost_benefit:.3f}")

    st.write("Raw input:")
    st.dataframe(X_raw, use_container_width=True)

    # 下载结果
    out = X_raw.copy()
    out["pred_prob"] = prob
    out["decision_pt"] = pt_custom
    out["risk_label"] = risk_label
    st.download_button(
        "Download this case (CSV)",
        out.to_csv(index=False).encode("utf-8-sig"),
        file_name="svm_single_case_result.csv",
        mime="text/csv",
    )

with col2:
    st.subheader("Single-case SHAP (waterfall)")

    # background 也要用 scaler 标准化后的版本
    try:
        bg_scaled = pd.DataFrame(scaler.transform(bg), columns=FEATURE_COLS)
        explainer = build_kernel_explainer(model, bg_scaled)

        with st.spinner("Computing SHAP (KernelExplainer)…"):
            fig, sv = plot_shap_waterfall(explainer, X_scaled, FEATURE_COLS)

        st.pyplot(fig, clear_figure=True)

        # Top 贡献表
        contrib = (
            pd.DataFrame({"Feature": FEATURE_COLS, "SHAP": sv})
            .assign(absSHAP=lambda d: d["SHAP"].abs())
            .sort_values("absSHAP", ascending=False)
            .drop(columns="absSHAP")
        )
        st.write("Top contributors (absolute SHAP):")
        st.dataframe(contrib, use_container_width=True)

    except Exception as e:
        st.warning("SHAP 解释生成失败（不影响概率输出）。常见原因：shap/numba 在云端构建不兼容或计算超时。")
        st.exception(e)


# ======================
# 7) 下方：三档阈值解释（论文友好）
# ======================
st.divider()
st.subheader("Clinical threshold interpretation (recommended for reporting)")

thr_list = [0.20, 0.40, 0.60]
thr_table = pd.DataFrame({
    "Threshold (pt)": thr_list,
    "Clinical strategy": ["Low threshold (high sensitivity / screening)",
                          "Middle threshold (balanced)",
                          "High threshold (high specificity / confirmatory)"],
    "Cost:Benefit (pt/(1-pt))": [t/(1-t) for t in thr_list],
})
thr_table["Cost:Benefit (pt/(1-pt))"] = thr_table["Cost:Benefit (pt/(1-pt))"].map(lambda x: f"{x:.3f}")
st.dataframe(thr_table, use_container_width=True)

st.caption("写作建议：不要只报告 Youden。可以用 DCA + CIC 在 pt=0.20/0.40/0.60 三个点分别解读，形成低/中/高阈值的临床策略描述。")
