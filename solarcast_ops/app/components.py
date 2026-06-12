from __future__ import annotations

import streamlit as st


def kpi(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def warning_box(message: str) -> None:
    st.warning(message, icon="⚠️")


def info_box(message: str) -> None:
    st.info(message)
