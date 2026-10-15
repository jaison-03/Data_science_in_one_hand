"""Charts for entity extraction statistics."""

import plotly.express as px
import streamlit as st


def render_statistics(summary):
    first, second, third, fourth, fifth = st.columns(5)
    first.metric("AWS entities", summary["total_entities"])
    second.metric("AWS attributes", summary["total_attributes"])
    third.metric("AWS categories", summary["unique_categories"])
    confidence = summary["average_confidence"]
    fourth.metric("Mean confidence", f"{confidence:.1%}" if confidence is not None else "Not returned")
    duration = summary["duration_seconds"]
    fifth.metric("API duration", f"{duration:.2f} s" if duration is not None else "—")
    if summary["categories"]:
        st.caption("Category and type charts count AWS entities and their nested AWS attributes.")
        st.plotly_chart(
            px.bar(
                x=list(summary["categories"].keys()),
                y=list(summary["categories"].values()),
                labels={"x": "AWS category", "y": "Returned entities"},
                title="Entities by AWS category",
            ),
            use_container_width=True,
        )
    if summary["types"]:
        st.markdown("**Most frequent returned types**")
        st.dataframe(
            [
                {"AWS type": name, "Count": count}
                for name, count in sorted(summary["types"].items(), key=lambda pair: (-pair[1], pair[0]))
            ],
            hide_index=True,
            use_container_width=True,
        )
