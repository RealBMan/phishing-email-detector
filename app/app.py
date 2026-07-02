"""Streamlit UI for the phishing detector.

Run from the project root:  streamlit run app/app.py
Requires the saved model — build it first with:  python -m src.train_model
"""
import sys
import re
import html as html_lib
from pathlib import Path

# Make the project root importable so `from src...` works under Streamlit.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import streamlit as st
import streamlit.components.v1 as components

from src.preprocess import clean_text
from src.parse import extract_body_from_string

MODEL_DIR = ROOT / "models"

# CVD-safe diverging pair for word highlighting (warm = phishing, cool = legit).
PHISH_RGB = "208, 59, 59"
LEGIT_RGB = "42, 120, 214"

PHISH_EXAMPLE = """Subject: Urgent: Your account has been temporarily suspended

Dear Customer,

We detected unusual sign-in activity on your account and have temporarily
suspended access as a security precaution. To restore your account, you must
verify your identity within 24 hours or it will be permanently closed.

Please confirm your login details by clicking the secure link below:

http://account-verify-secure.xyz/login

Failure to verify will result in immediate suspension of all services.
Kindly act now to avoid interruption.

Sincerely,
The Security Team"""

LEGIT_EXAMPLE = """Subject: Thursday meeting + Q3 numbers

Hi team,

Attaching the Q3 pipeline spreadsheet ahead of Thursday's meeting. Let me know
if the gas volume figures on tab 2 look right before I send them to finance.

Thanks,
Sam"""


@st.cache_resource
def load_artifacts():
    """Load once, cache across reruns (this is why the app is instant)."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    return model, vectorizer


def word_contributions(text, model, vectorizer):
    """Exact per-word contribution for a linear model: tfidf * coefficient.
    Returns (contrib_map, cleaned_text)."""
    cleaned = clean_text(text)
    x = vectorizer.transform([cleaned])
    contrib = x.multiply(model.coef_[0]).toarray()[0]
    names = vectorizer.get_feature_names_out()
    contrib_map = {names[i]: contrib[i] for i in contrib.nonzero()[0]}
    return contrib_map, cleaned


def highlight_email(cleaned, contrib_map):
    """Render the cleaned email with each word tinted by its contribution:
    warm = pushed toward phishing, cool = toward legit, intensity by strength."""
    if not contrib_map:
        return "<em>No recognized words.</em>"
    max_abs = max(abs(v) for v in contrib_map.values())

    def repl(m):
        word = m.group(0)
        c = contrib_map.get(word.lower())
        if c is None or abs(c) < 0.02 * max_abs:
            return word
        alpha = min(0.85, 0.18 + abs(c) / max_abs * 0.62)
        rgb = PHISH_RGB if c > 0 else LEGIT_RGB
        return (f'<span style="background:rgba({rgb},{alpha:.2f});'
                f'border-radius:3px;padding:0 2px;">{word}</span>')

    escaped = html_lib.escape(cleaned)
    body = re.sub(r"\b\w\w+\b", repl, escaped)
    return (f'<div style="line-height:1.95;white-space:pre-wrap;'
            f'font-size:0.92rem;">{body}</div>')


def verdict_style(proba):
    """3-band status: color + label + icon (never color alone)."""
    if proba >= 0.65:
        return "#d03b3b", "Phishing", "🚨"
    if proba >= 0.35:
        return "#ec835a", "Suspicious", "⚠️"
    return "#0ca30c", "Legitimate", "✅"


def risk_meter(proba, color):
    pct = proba * 100
    return f"""
    <div style="position:relative;margin:30px 0 4px;">
      <div style="position:absolute;left:{pct:.1f}%;top:-24px;transform:translateX(-50%);
           font-weight:800;font-size:0.85rem;color:{color};white-space:nowrap;">{pct:.0f}%</div>
      <div style="height:16px;border-radius:8px;
           background:linear-gradient(90deg,#0ca30c 0%,#fab219 50%,#d03b3b 100%);"></div>
      <div style="position:absolute;left:{pct:.1f}%;top:-5px;transform:translateX(-50%);
           width:5px;height:26px;background:#111;border:2px solid #fff;border-radius:3px;
           box-shadow:0 0 4px rgba(0,0,0,0.55);"></div>
      <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-top:6px;">
        <span>Legit</span><span>Suspicious</span><span>Phishing</span>
      </div>
    </div>"""


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣", layout="centered")
st.title("🎣 Phishing Email Detector")
st.caption("Paste an email or upload a .eml file. You get a phishing score and see "
           "exactly which words drove it — highlighted in the text.")

model, vectorizer = load_artifacts()

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

# Example / clear buttons — set the shared text state, then the box re-renders.
c1, c2, c3 = st.columns(3)
if c1.button("📥 Phishing example", use_container_width=True):
    st.session_state.email_text = PHISH_EXAMPLE
if c2.button("📥 Legit example", use_container_width=True):
    st.session_state.email_text = LEGIT_EXAMPLE
if c3.button("🗑 Clear", use_container_width=True):
    st.session_state.email_text = ""

uploaded = st.file_uploader("…or upload a .eml file", type=["eml"])
if uploaded is not None and uploaded.name != st.session_state.get("last_file"):
    st.session_state.email_text = extract_body_from_string(uploaded.read().decode("latin-1"))
    st.session_state.last_file = uploaded.name

email_text = st.text_area("Email text", key="email_text", height=240,
                          placeholder="Paste the email body here…")

if st.button("Analyze", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste some email text or upload a .eml file.")
    else:
        contrib_map, cleaned = word_contributions(email_text, model, vectorizer)
        proba = model.predict_proba(vectorizer.transform([cleaned]))[0, 1]
        color, label, icon = verdict_style(proba)

        # Anchored header so we can auto-scroll here after Analyze.
        st.subheader("Result", anchor="analysis")

        # Result header: soft badge + big score + meter.
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;margin-top:6px;">'
            f'<span style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},'
            f'{int(color[5:7],16)},0.15);color:{color};padding:5px 14px;border-radius:20px;'
            f'font-weight:700;">{icon} {label}</span>'
            f'<span style="font-size:2.1rem;font-weight:800;color:{color};">{proba:.0%}</span>'
            f'<span style="color:#888;">phishing score</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(risk_meter(proba, color), unsafe_allow_html=True)

        st.subheader("Why — the model's reasoning, on your text")
        st.markdown(
            f'<div style="font-size:0.8rem;color:#888;margin-bottom:6px;">'
            f'<span style="background:rgba({PHISH_RGB},0.5);padding:0 4px;border-radius:3px;">warm</span>'
            f' = pushed toward phishing &nbsp;·&nbsp; '
            f'<span style="background:rgba({LEGIT_RGB},0.5);padding:0 4px;border-radius:3px;">cool</span>'
            f' = pushed toward legitimate</div>',
            unsafe_allow_html=True,
        )
        st.markdown(highlight_email(cleaned, contrib_map), unsafe_allow_html=True)

        # Secondary (table) view of the strongest words — accessibility + detail.
        with st.expander("Top contributing words (table)"):
            ranked = sorted(contrib_map.items(), key=lambda kv: -abs(kv[1]))[:12]
            rows = [{"word": w,
                     "pushes toward": "phishing" if c > 0 else "legitimate",
                     "contribution": round(float(c), 3)} for w, c in ranked]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.caption("Contributions are exact for this linear model: "
                   "tf-idf value × learned weight per word. Threshold for the "
                   "verdict is 0.5 — lower it to catch more phishing at the cost "
                   "of more false alarms.")

        # Jump the viewport to the Result header so the user doesn't scroll.
        components.html(
            "<script>"
            "const el = window.parent.document.getElementById('analysis');"
            "if (el) el.scrollIntoView({behavior:'smooth', block:'start'});"
            "</script>",
            height=0,
        )
