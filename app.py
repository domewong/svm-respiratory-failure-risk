# app.py
# -*- coding: utf-8 -*-

import os
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# SHAP（在云端可能略慢，KernelExplainer 本就慢一些）
import shap


# ======================
# 基本配置
# ======================
st.set_page_config(
    page_title="Respiratory Failure Risk Calculator (SVM)",
    page_icon="🫁",
    layout="wide",
)

FEATURE_COLS = ["Age", "PaO2", "PF_ratio", "pneumonia", "ISS"]

# 和你论文阈值解释一致：可把 0.2/0.4/0.6 做成快捷按钮；默认 0.40
DEFAULT_PT = 0.40


# ======================
# 资源加载
# ======================
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "svm_model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    bg_path = os.path.join(base_dir, "shap_background.pkl")

    for p in [model_path, scaler_path, bg_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(bg_path, "rb") as f:
        bg = pickle.load(f)

    # 背景数据转成 DataFrame，确保列名一致（避免 sklearn 的 “no valid feature names” 警告）
    if isinstance(bg, np.ndarray):
        bg_df = pd.DataFrame(bg, columns=FEATURE_COLS)
    elif isinstance(bg, pd.DataFrame):
        bg_df = bg[FEATURE_COLS].copy()
    else:
        # 兜底：尝试转 DataFrame
        bg_df = pd.DataFrame(np.array(bg), columns=FEATURE_COLS)

    # KernelExplainer：用 predict_proba 输出概率；对二分类会返回每类的解释
    explainer = shap.KernelExplainer(model.predict_proba, bg_df)

    return model, scaler, explainer


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ======================
# SHAP waterfall（修复：只画正类/事件类=1）
# ======================
def plot_shap_waterfall(explainer, X_one_df: pd.DataFrame, positive_class: int = 1):
    """
    explainer: shap explainer
    X_one_df: shape (1, n_features) 的 DataFrame，列名=FEATURE_COLS
    positive_class: 二分类事件类通常为 1
    """
    # shap_values 可能是 list 或 Explanation 或 array，统一成单个 Explanation
    sv = explainer.shap_values(X_one_df)

    # 情况A：旧版常见 -> list，sv[0] 为 class0，sv[1] 为 class1，形状 (1, n_features)
    if isinstance(sv, list):
        vals = np.array(sv[positive_class])[0]
        base = explainer.expected_value[positive_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        exp = shap.Explanation(
            values=vals,
            base_values=base,
            data=X_one_df.iloc[0].values,
            feature_names=list(X_one_df.columns),
        )

    else:
        # 情况B：array / Explanation
        arr = np.array(sv)

        # 常见：KernelExplainer 返回 (1, n_features, 2) 或 (n_features, 2) 或 (1, n_features)
        if arr.ndim == 3:
            # (1, n_features, 2)
            vals = arr[0, :, positive_class]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            # (n_features, 2)
            vals = arr[:, positive_class]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # (1, n_features)
            vals = arr[0]
        else:
            # (n_features,)
            vals = arr

        base = explainer.expected_value[positive_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        exp = shap.Explanation(
            values=vals,
            base_values=base,
            data=X_one_df.iloc[0].values,
            feature_names=list(X_one_df.columns),
        )

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(exp, max_display=len(FEATURE_COLS), show=False)
    plt.tight_layout()
    return fig


# ======================
# 页面
# ======================
st.title("🫁 Respiratory Failure Risk Calculator (SVM)")
st.caption("输入临床变量 → 输出个体风险（概率） + 单例 SHAP 解释（waterfall）。")

st.info("提示：该工具用于科研展示与辅助决策，不替代临床医生判断。", icon="ℹ️")

# 加载模型
try:
    model, scaler, explainer = load_assets()
except Exception as e:
    st.error(f"模型/文件加载失败：{e}")
    st.stop()

# ======================
# 侧边栏输入
# ======================
with st.sidebar:
    st.header("Input features")

    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
    pao2 = st.number_input("PaO₂ (mmHg)", min_value=0.0, max_value=600.0, value=82.0, step=1.0)
    pf_ratio = st.number_input("PF ratio (PaO₂/FiO₂)", min_value=0.0, max_value=1000.0, value=250.0, step=5.0)

    # ✅ 修正：只显示 Pneumonia + 0/1 解释
    pneumonia = st.selectbox("Pneumonia (0=No, 1=Yes)", options=[0, 1], index=1)

    iss = st.number_input("ISS (Injury Severity Score)", min_value=0.0, max_value=75.0, value=26.0, step=1.0)

    st.markdown("---")

    pt = st.slider("Decision threshold (pt)", min_value=0.05, max_value=0.95, value=float(DEFAULT_PT), step=0.01)
    st.caption("建议用于论文阈值解释：pt=0.20 / 0.40 / 0.60（三档）")


# ======================
# 组织输入 + 标准化 + 预测
# ======================
X_raw = pd.DataFrame(
    [[age, pao2, pf_ratio, pneumonia, iss]],
    columns=FEATURE_COLS
)

# 标准化：保持 DataFrame 列名一致（避免 sklearn 警告）
X_scaled_np = scaler.transform(X_raw)
X_scaled = pd.DataFrame(X_scaled_np, columns=FEATURE_COLS)

prob = float(model.predict_proba(X_scaled)[0, 1])
pred_label = int(prob >= pt)

cost_benefit = pt / (1 - pt)

# ======================
# 主区布局
# ======================
col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.subheader("Prediction")

    st.metric("Predicted risk (probability)", f"{prob:.3f}")

    if pred_label == 1:
        st.error(f"Decision (pt={pt:.2f}): High risk")
    else:
        st.success(f"Decision (pt={pt:.2f}): Low risk")

    st.caption(f"Cost:Benefit ratio = pt/(1-pt) = {cost_benefit:.3f}")

    st.write("Raw input:")
    st.dataframe(X_raw, use_container_width=True)

    # 下载该个案
    csv_bytes = X_raw.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download this case (CSV)",
        data=csv_bytes,
        file_name="svm_case_input.csv",
        mime="text/csv"
    )


with col_right:
    st.subheader("Single-case SHAP (waterfall)")

    # 画 SHAP
    try:
        fig = plot_shap_waterfall(explainer, X_scaled, positive_class=1)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(
            "SHAP 解释生成失败（不影响概率输出）。常见原因：云端环境 shap/numba 兼容或计算较慢/超时。",
            icon="⚠️"
        )
        st.exception(e)


st.markdown("---")
st.caption("Tip: 如果页面异常空白，优先检查 GitHub 仓库中的 app.py 是否为空（0KB）以及 pkl 文件是否已上传到同目录。")
