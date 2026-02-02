"""Shared styles for ML Gap Finder UI."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Force light blue theme on everything */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Sidebar - force readable dark text on light blue background */
[data-testid="stSidebar"] {
    background-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1e3a8a !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #475569 !important;
}

/* All headers - dark blue */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1e3a8a !important;
}

/* All body text - dark gray */
p, span, label, li, td, th,
.stMarkdown p, .stMarkdown span, .stMarkdown li {
    color: #334155 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid #cbd5e1 !important;
}
[data-testid="stMetricLabel"] {
    color: #1e3a8a !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stSlider label,
.stCheckbox label,
.stRadio label {
    color: #1e3a8a !important;
}

/* Buttons - blue */
.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}
button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #e2e8f0 !important;
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    background-color: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #1e3a8a !important;
    background-color: #ffffff !important;
}

/* Expanders */
.stExpander {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    color: #1e3a8a !important;
}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #1e3a8a !important;
}

/* Alert boxes */
[data-testid="stAlert"] {
    color: #1e293b !important;
}
.stSuccess, [data-baseweb="notification"][kind="positive"] {
    background-color: #dcfce7 !important;
    color: #166534 !important;
}
.stError, [data-baseweb="notification"][kind="negative"] {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
}
.stWarning, [data-baseweb="notification"][kind="warning"] {
    background-color: #fef3c7 !important;
    color: #92400e !important;
}
.stInfo, [data-baseweb="notification"][kind="info"] {
    background-color: #dbeafe !important;
    color: #1e40af !important;
}

/* Form container */
[data-testid="stForm"] {
    background-color: #ffffff !important;
    padding: 1.5rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid #cbd5e1 !important;
}

/* Divider */
hr, [data-testid="stDivider"] {
    border-color: #cbd5e1 !important;
}

/* Captions and small text */
.stCaption, small, .caption {
    color: #64748b !important;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #3b82f6 !important;
    color: #475569 !important;
    background-color: #f1f5f9 !important;
}

/* Code blocks */
code {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
}
pre {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}
</style>
"""


def apply_custom_styles():
    """Apply custom CSS styles to the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
