import streamlit as st

from modules import database as db
from modules.ui import apply_style, donut_chart, hero, source_chart, timeline_chart

st.set_page_config(page_title="Detection History", page_icon="📊", layout="wide")
apply_style()
hero("Detection History & Analytics", "Everything the app has detected, with charts and reports.", "📊")

stats = db.totals()
c1, c2, c3 = st.columns(3)
c1.metric("Total detections", stats["total"])
c2.metric("Object types", stats["labels"])
c3.metric("Average confidence", f"{stats['avg_conf']}%")

history = db.get_history()
if history.empty:
    st.info("No history yet. Analyze an image or use the live camera first.")
else:
    st.write("")
    a, b = st.columns([1, 1.3])
    with a:
        st.subheader("Objects detected")
        st.altair_chart(donut_chart(db.label_counts()), use_container_width=True)
    with b:
        st.subheader("Activity over time")
        st.altair_chart(timeline_chart(history), use_container_width=True)

    st.subheader("Detections by source")
    st.altair_chart(source_chart(history), use_container_width=True)

    st.subheader("Records")
    sources = st.multiselect("Filter by source", sorted(history["source"].unique()),
                             default=sorted(history["source"].unique()))
    view = history[history["source"].isin(sources)]
    st.dataframe(view, use_container_width=True)

    d1, d2 = st.columns([1, 5])
    d1.download_button("Download CSV", view.to_csv(index=False),
                       file_name="detection_report.csv", mime="text/csv")
    if d2.button("Clear all history"):
        db.clear()
        st.rerun()
