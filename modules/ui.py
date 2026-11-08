"""Shared look & feel: navy/cyan cards, hero header, and dashboard charts."""
import altair as alt
import pandas as pd
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0F2342 0%, #0A1A2F 100%);
    border-right: 1px solid #1E3A5F;}
[data-testid="stMetric"] {background: linear-gradient(135deg, #12294A 0%, #0F2342 100%);
    border: 1px solid #1E3A5F; border-left: 4px solid #00B8D9; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);}
[data-testid="stMetricLabel"] {color: #8FB3D9;}
[data-testid="stMetricValue"] {color: #00D4F5; font-weight: 700;}
.hero {background: linear-gradient(120deg, #12294A 0%, #0B3A5C 55%, #00677F 100%);
    border: 1px solid #1E4A6F; border-radius: 18px; padding: 26px 30px; margin-bottom: 22px;
    box-shadow: 0 6px 22px rgba(0,184,217,0.15);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #FFFFFF;}
.hero p {margin: 6px 0 0 0; color: #B9D6F2; font-size: 1.02rem;}
.vcard {background: linear-gradient(135deg, #12294A 0%, #0F2342 100%); border: 1px solid #1E3A5F;
    border-radius: 16px; padding: 20px; height: 100%; margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25); transition: transform .15s, border-color .15s;}
.vcard:hover {transform: translateY(-3px); border-color: #00B8D9;}
.vcard .icon {font-size: 2rem;}
.vcard h4 {margin: 8px 0 4px 0; color: #FFFFFF; font-size: 1.1rem;}
.vcard p {margin: 0; color: #8FB3D9; font-size: .92rem;}
.stButton > button {border-radius: 10px; border: 1px solid #00B8D9;}
.stButton > button[kind="primary"] {background: linear-gradient(90deg, #00B8D9, #0090B0); color: #fff; border: none;}
[data-testid="stImage"] img {border-radius: 12px;}
[data-testid="stFileUploader"] section {border: 2px dashed #1E4A6F; border-radius: 14px; background: #0F2342;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle="", icon=""):
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(icon, title, text):
    return (f'<div class="vcard"><div class="icon">{icon}</div>'
            f'<h4>{title}</h4><p>{text}</p></div>')


def _theme(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelColor="#B9D6F2", titleColor="#B9D6F2", gridColor="#1E3A5F")
            .configure_legend(labelColor="#B9D6F2", titleColor="#B9D6F2"))


def donut_chart(counts_df, top=8):
    """counts_df has columns: label, count"""
    df = counts_df.head(top)
    chart = (alt.Chart(df).mark_arc(innerRadius=70, outerRadius=130)
             .encode(theta=alt.Theta("count:Q"),
                     color=alt.Color("label:N", scale=alt.Scale(scheme="tealblues"),
                                     legend=alt.Legend(title="Object")),
                     tooltip=["label", "count"])
             .properties(height=320))
    return _theme(chart)


def timeline_chart(history_df):
    """history_df has a 'timestamp' column. Groups detections per hour."""
    df = history_df.copy()
    df["time"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    per = df.groupby("time").size().reset_index(name="detections")
    base = alt.Chart(per).encode(x=alt.X("time:T", title="Time"),
                                 y=alt.Y("detections:Q", title="Detections"),
                                 tooltip=["time:T", "detections:Q"])
    chart = (base.mark_area(color="#00B8D9", opacity=0.25, line=False)
             + base.mark_line(color="#00D4F5", strokeWidth=3)
             + base.mark_point(color="#FFFFFF", size=60, filled=True)).properties(height=320)
    return _theme(chart)


def source_chart(history_df):
    df = history_df["source"].value_counts().reset_index()
    df.columns = ["source", "count"]
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=6, color="#00B8D9")
             .encode(x=alt.X("count:Q", title="Detections"), y=alt.Y("source:N", sort="-x", title=""),
                     tooltip=["source", "count"]).properties(height=200))
    return _theme(chart)
