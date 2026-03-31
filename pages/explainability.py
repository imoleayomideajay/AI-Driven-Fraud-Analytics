from __future__ import annotations

import streamlit as st

from src.explain import global_feature_importance, local_reason_codes, shap_values_for_row
from src.features import build_features



def render_explainability(df, pipeline, feature_columns) -> None:
    st.header("Explainability")

    featured = build_features(df.copy())
    X = featured.drop(columns=["label_fraud", "transaction_id", "customer_id", "timestamp"], errors="ignore")
    X = X.reindex(columns=feature_columns)

    importance = global_feature_importance(pipeline, X.head(3000))
    st.subheader("Global Fraud Drivers")
    st.dataframe(importance.head(20))
    st.bar_chart(importance.head(15).set_index("feature"))

    st.subheader("Local Explanation")
    index = st.number_input("Row index", min_value=0, max_value=max(len(X) - 1, 0), value=0, step=1)
    shap_df = shap_values_for_row(pipeline, X.head(1000), int(index))

    if shap_df is not None:
        st.success("SHAP explanation available")
        st.dataframe(shap_df.head(15))
    else:
        st.info("SHAP unavailable or unsupported for current model. Using rule-based fallback reasons.")
        reasons = local_reason_codes(df.iloc[[int(index)]])
        st.write(reasons[["transaction_id", "reason_codes"]])
