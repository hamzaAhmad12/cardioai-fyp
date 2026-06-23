import streamlit as st
import requests

st.set_page_config(
    page_title="CardioAI — Heart Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://localhost:8000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] { background: #0f1117 !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.875rem !important; padding: 0.35rem 0 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.875rem !important; }

.main-hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(239,68,68,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    line-height: 1.15;
}
.hero-title span { color: #ef4444; }
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 300;
    margin: 0;
    max-width: 520px;
    line-height: 1.6;
}

.stat-card {
    background: #1a1f2e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #f1f5f9;
    margin: 0;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.2rem 0 0;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    font-weight: 500;
    margin: 0 0 0.75rem;
}
.card-dark {
    background: #1a1f2e;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-light {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.risk-badge-high {
    display: inline-block;
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    border-radius: 999px;
    padding: 0.25rem 0.875rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-badge-medium {
    display: inline-block;
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    border-radius: 999px;
    padding: 0.25rem 0.875rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-badge-low {
    display: inline-block;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 999px;
    padding: 0.25rem 0.875rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.result-metric {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
}
.result-metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #f1f5f9;
    margin: 0;
}
.result-metric-lbl {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.25rem 0 0;
}

.chat-user-msg {
    background: #ef4444;
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 4rem;
    font-size: 0.9rem;
    line-height: 1.5;
}
.chat-bot-msg {
    background: #1a1f2e;
    color: #e2e8f0;
    border: 1px solid #1e2130;
    border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 4rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.stButton > button {
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #dc2626 !important;
    transform: translateY(-1px) !important;
}

.stTextInput input, .stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border-color: #1e2130 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #0f1117;
    margin: 0 0 0.25rem;
}
.page-sub {
    font-size: 0.875rem;
    color: #64748b;
    margin: 0 0 1.5rem;
}

.tech-tag {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
}

.source-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 0.15rem;
}

div[data-testid="stForm"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    background: #f8fafc !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; }

hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

backend_ok = check_backend()

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-size:1.3rem; font-weight:600; color:#f1f5f9; font-family:'DM Serif Display',serif; letter-spacing:-0.01em;">
            🫀 CardioAI
        </div>
        <div style="font-size:0.72rem; color:#475569; margin-top:0.15rem; text-transform:uppercase; letter-spacing:0.1em;">
            CVD Risk Assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if backend_ok:
        st.markdown('<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#4ade80;padding:0.5rem 0;"><span style="width:7px;height:7px;background:#4ade80;border-radius:50%;display:inline-block;"></span> System online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#f87171;padding:0.5rem 0;"><span style="width:7px;height:7px;background:#f87171;border-radius:50%;display:inline-block;"></span> Backend offline</div>', unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "",
        ["Home", "Heart Disease Prediction", "Medical Chatbot", "Integrated Analysis", "Technical Details", "About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="position:fixed; bottom:1.5rem; font-size:0.7rem; color:#334155; line-height:1.7;">
        <div style="font-weight:500; color:#475569; margin-bottom:0.3rem;">Final Year Project · 2025</div>
        Hamza · Roll No. 157<br>
        Tufail Abbas · Roll No. 164<br>
        <span style="color:#64748b;">Supervisor: Dr. Rehman Ali</span>
    </div>
    """, unsafe_allow_html=True)


# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div class="main-hero">
        <p class="section-label" style="color:#ef4444;">AI-Powered Clinical Tool</p>
        <h1 class="hero-title">Cardiovascular Risk<br><span>Prediction System</span></h1>
        <p class="hero-sub">Machine learning combined with evidence-based medical guidelines to assess heart disease risk and deliver personalized clinical recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><p class="stat-val">85.7%</p><p class="stat-lbl">ML Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><p class="stat-val">87.2%</p><p class="stat-lbl">Recall</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><p class="stat-val">298</p><p class="stat-lbl">Patients trained</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><p class="stat-val">2</p><p class="stat-lbl">Clinical guidelines</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card-dark">
            <p class="section-label">ML Component</p>
            <div style="font-size:1.1rem; font-weight:500; color:#f1f5f9; margin-bottom:0.5rem;">Heart Disease Prediction</div>
            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.7;">
                Logistic Regression trained on 13 UCI clinical features. Predicts disease probability with confidence scoring and risk stratification (Low / Medium / High).
            </div>
            <div style="margin-top:1rem;">
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">Scikit-learn</span>
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">Logistic Regression</span>
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">UCI Dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card-dark">
            <p class="section-label">RAG Component</p>
            <div style="font-size:1.1rem; font-weight:500; color:#f1f5f9; margin-bottom:0.5rem;">Medical Guidelines Chatbot</div>
            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.7;">
                Retrieval-Augmented Generation over AHA Hypertension and CHD clinical guidelines. Answers ground in source documents with page-level citations.
            </div>
            <div style="margin-top:1rem;">
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">LangChain</span>
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">Groq · Llama 3.1</span>
                <span class="tech-tag" style="background:#1a1f2e;color:#94a3b8;border-color:#1e2130;">ChromaDB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:12px;padding:1.25rem 1.5rem;margin-top:0.5rem;">
        <p class="section-label" style="color:#b91c1c;">Key integration</p>
        <div style="font-size:0.9rem; color:#7f1d1d; line-height:1.6;">
            The Integrated Analysis page passes the ML prediction result and full patient profile directly into the RAG prompt — recommendations are generated for the specific patient, not generically from guidelines.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── HEART DISEASE ─────────────────────────────────────────────────────────────
elif page == "Heart Disease Prediction":
    st.markdown('<h1 class="page-title">Heart Disease Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter clinical parameters to generate an ML-based cardiovascular risk score.</p>', unsafe_allow_html=True)

    if not backend_ok:
        st.error("Backend offline — start the server with `python main.py`.")
        st.stop()

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age (years)", 20, 100, 55)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            st.markdown("**Cardiac symptoms**")
            cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda x: {1:"Typical angina",2:"Atypical angina",3:"Non-anginal pain",4:"Asymptomatic"}[x])
            exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            st.markdown("**Vitals**")
            trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
            thalach = st.number_input("Max heart rate (bpm)", 60, 220, 150)
            fbs = st.selectbox("Fasting blood sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col3:
            st.markdown("**ECG & Imaging**")
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LV hypertrophy"}[x])
            oldpeak = st.number_input("ST depression", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("ST slope", [1, 2, 3], format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
            ca = st.number_input("Major vessels (0–3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversible defect"}[x])

        submitted = st.form_submit_button("Run prediction", use_container_width=True)

    if submitted:
        with st.spinner("Running model inference…"):
            try:
                payload = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
                           "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
                           "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
                r = requests.post(f"{BACKEND_URL}/api/ml/predict", json=payload, timeout=10)

                if r.status_code == 200:
                    res = r.json()
                    pred_label = res.get("prediction_label", "Unknown")
                    risk = res.get("risk_level", "Unknown")
                    prob = res.get("probability_disease", 0)

                    st.markdown("---")

                    badge_map = {"High": "risk-badge-high", "Medium": "risk-badge-medium", "Low": "risk-badge-low"}
                    badge_cls = badge_map.get(risk, "risk-badge-medium")

                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
                        <span style="font-size:1.1rem;font-weight:500;color:#0f1117;">{pred_label}</span>
                        <span class="{badge_cls}">{risk} risk</span>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="result-metric"><p class="result-metric-val">{float(prob)*100:.1f}%</p><p class="result-metric-lbl">Disease probability</p></div>', unsafe_allow_html=True)
                    with c2:
                        hp = res.get("probability_healthy", 0)
                        st.markdown(f'<div class="result-metric"><p class="result-metric-val">{float(hp)*100:.1f}%</p><p class="result-metric-lbl">Healthy probability</p></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="result-metric"><p class="result-metric-val">{risk}</p><p class="result-metric-lbl">Risk level</p></div>', unsafe_allow_html=True)

                    with st.expander("Raw response"):
                        st.json(res)
                else:
                    st.error(f"Error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# ── CHATBOT ───────────────────────────────────────────────────────────────────
elif page == "Medical Chatbot":
    st.markdown('<h1 class="page-title">Medical Guidelines Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Ask questions grounded in AHA Hypertension and CHD clinical guidelines.</p>', unsafe_allow_html=True)

    if not backend_ok:
        st.error("Backend offline.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                src_html = "".join([
                    f'<span class="source-pill">{s.get("guideline","").replace("_"," ").replace(".pdf","")} · p.{s.get("page","?")}</span>'
                    for s in msg["sources"]
                ])
                st.markdown(f'<div style="margin:-0.3rem 4rem 0.5rem 0;">{src_html}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about cardiovascular health, hypertension, angina…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-user-msg">{prompt}</div>', unsafe_allow_html=True)

        with st.spinner("Searching guidelines…"):
            try:
                r = requests.post(f"{BACKEND_URL}/api/rag/chat", json={"message": prompt}, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("bot_response", "No response received.")
                    sources = data.get("sources", [])
                    conf = data.get("confidence", 0)

                    st.markdown(f'<div class="chat-bot-msg">{answer}</div>', unsafe_allow_html=True)

                    if sources:
                        src_html = "".join([
                            f'<span class="source-pill">{s.get("guideline","").replace("_"," ").replace(".pdf","")} · p.{s.get("page","?")}</span>'
                            for s in sources
                        ])
                        st.markdown(f'<div style="margin:-0.3rem 4rem 0.75rem 0;">{src_html}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
                else:
                    st.error(f"Error {r.status_code}")
            except requests.Timeout:
                st.error("Request timed out. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ── INTEGRATED ANALYSIS ───────────────────────────────────────────────────────
elif page == "Integrated Analysis":
    st.markdown('<h1 class="page-title">Integrated Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">ML prediction and patient profile are passed together to the RAG system for personalized clinical recommendations.</p>', unsafe_allow_html=True)

    if not backend_ok:
        st.error("Backend offline.")
        st.stop()

    with st.form("combined_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age (years)", 20, 100, 55)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            st.markdown("**Cardiac symptoms**")
            cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda x: {1:"Typical angina",2:"Atypical angina",3:"Non-anginal pain",4:"Asymptomatic"}[x])
            exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            st.markdown("**Vitals**")
            trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
            thalach = st.number_input("Max heart rate (bpm)", 60, 220, 150)
            fbs = st.selectbox("Fasting blood sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col3:
            st.markdown("**ECG & Imaging**")
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LV hypertrophy"}[x])
            oldpeak = st.number_input("ST depression", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("ST slope", [1, 2, 3], format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
            ca = st.number_input("Major vessels (0–3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversible defect"}[x])

        st.markdown("---")
        question = st.text_area("Clinical question (optional)", placeholder="e.g. What do my results mean? What lifestyle changes are recommended?")
        submitted = st.form_submit_button("Run integrated analysis", use_container_width=True)

    if submitted:
        with st.spinner("Running ML inference and retrieving guideline context…"):
            try:
                payload = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
                           "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
                           "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal,
                           "patient_question": question if question else None}
                r = requests.post(f"{BACKEND_URL}/api/combined/analyze", json=payload, timeout=120)

                if r.status_code == 200:
                    res = r.json()

                    pred_label = res.get("prediction_label", res.get("ml_prediction",{}).get("prediction_label","Unknown"))
                    risk = res.get("risk_level", res.get("ml_prediction",{}).get("risk_level","Unknown"))
                    prob = res.get("risk_score", res.get("ml_prediction",{}).get("probability_disease", 0))

                    badge_map = {"High":"risk-badge-high","Medium":"risk-badge-medium","Low":"risk-badge-low"}
                    badge_cls = badge_map.get(risk, "risk-badge-medium")

                    st.markdown("---")
                    st.markdown(f"""
                    <p class="section-label">ML model output</p>
                    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
                        <span style="font-size:1.05rem;font-weight:500;color:#0f1117;">{pred_label}</span>
                        <span class="{badge_cls}">{risk} risk</span>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="result-metric"><p class="result-metric-val">{float(prob)*100:.1f}%</p><p class="result-metric-lbl">Disease probability</p></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="result-metric"><p class="result-metric-val">{risk}</p><p class="result-metric-lbl">Risk level</p></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown('<p class="section-label">Clinical recommendations · RAG</p>', unsafe_allow_html=True)

                    analysis = res.get("ai_analysis", res.get("rag_analysis",{}).get("answer",""))
                    if analysis:
                        st.markdown(f'<div class="card-light" style="font-size:0.9rem;color:#1e293b;line-height:1.7;">{analysis}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No analysis returned.")

                    clinical = res.get("clinical_summary","")
                    if clinical:
                        with st.expander("Clinical summary"):
                            st.code(clinical, language="text")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")
            except requests.Timeout:
                st.error("Request timed out. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")


# ── TECHNICAL DETAILS ─────────────────────────────────────────────────────────
elif page == "Technical Details":
    st.markdown('<h1 class="page-title">Technical Details</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Implementation reference for academic presentation.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "ML model", "RAG system", "Deployment"])

    with tab1:
        st.markdown("#### System architecture")
        st.code("""
Streamlit UI
    └─ HTTP / REST ──► FastAPI (port 8000)
                            ├─ /api/ml/predict
                            │       └─ Scikit-learn · Logistic Regression
                            ├─ /api/rag/chat
                            │       └─ LangChain → ChromaDB → Groq
                            └─ /api/combined/analyze
                                    └─ ML result + patient profile → RAG prompt
        """, language="text")

        st.markdown("#### Stack")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Frontend**")
            for t in ["Streamlit", "Python 3.14", "requests"]:
                st.markdown(f"<span class='tech-tag'>{t}</span>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("**Backend**")
            for t in ["FastAPI", "Uvicorn", "Pydantic"]:
                st.markdown(f"<span class='tech-tag'>{t}</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("**AI layer**")
            for t in ["LangChain", "Groq · Llama 3.1 8B", "ChromaDB", "all-MiniLM-L6-v2", "CrossEncoder"]:
                st.markdown(f"<span class='tech-tag'>{t}</span>", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Machine learning model")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Dataset | UCI Heart Disease |
| Patients | 298 (cleaned) |
| Features | 13 clinical |
| Train / test split | 70 / 30 % |
| Scaler | StandardScaler |
""")
        with cols[1]:
            st.markdown("""
| Metric | Score |
|---|---|
| Accuracy | 85.7 % |
| Recall | 87.2 % |
| Precision | 84.3 % |
| F1-score | 85.7 % |
| ROC-AUC | 0.92 |
""")

        st.markdown("#### Input features")
        feats = [
            ("age","Patient age in years"),
            ("sex","Gender — 1 Male, 0 Female"),
            ("cp","Chest pain type — 1 typical angina … 4 asymptomatic"),
            ("trestbps","Resting blood pressure (mmHg)"),
            ("chol","Serum cholesterol (mg/dL)"),
            ("fbs","Fasting blood sugar > 120 mg/dL (0/1)"),
            ("restecg","Resting ECG results — 0 normal, 1 ST-T, 2 LVH"),
            ("thalach","Maximum heart rate achieved"),
            ("exang","Exercise-induced angina (0/1)"),
            ("oldpeak","ST depression induced by exercise"),
            ("slope","Slope of peak exercise ST segment — 1–3"),
            ("ca","Number of major vessels coloured by fluoroscopy (0–3)"),
            ("thal","Thalassemia — 1 normal, 2 fixed, 3 reversible defect"),
        ]
        for name, desc in feats:
            st.markdown(f"<code>{name}</code> — <span style='color:#64748b;font-size:0.875rem;'>{desc}</span>", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### RAG pipeline")
        st.code("""
1. Query received
2. Embed with all-MiniLM-L6-v2 (384-dim, local CPU)
3. Similarity search — ChromaDB, k=10
4. Re-rank with CrossEncoder ms-marco-MiniLM-L-6-v2 → top 4
5. Build prompt with patient context + guideline chunks
6. Generate with Groq · Llama 3.1 8B Instant (~1 s)
7. Return answer + source citations
        """, language="text")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Knowledge base**")
            st.write("AHA Hypertension Guidelines (2017)")
            st.write("CHD Clinical Guidelines — MUS_D1")
            st.markdown("**Chunking**")
            st.write("Chunk size: 1 000 characters")
            st.write("Overlap: 200 characters")
        with cols[1]:
            st.markdown("**Embedding model**")
            st.write("all-MiniLM-L6-v2 · 384 dims · local")
            st.markdown("**Reranker**")
            st.write("CrossEncoder ms-marco-MiniLM-L-6-v2")
            st.markdown("**LLM**")
            st.write("Groq · Llama 3.1 8B Instant · ~1 s response")

        st.markdown("#### ML → RAG integration")
        st.write("The `/api/combined/analyze` endpoint feeds the ML prediction result, risk level, and full patient parameter set into the RAG prompt before generation. The LLM therefore produces recommendations specific to the patient rather than generic guideline summaries.")

    with tab4:
        st.markdown("#### Running locally")
        st.code("""
# Terminal 1 — backend
cd "D:\\Final Year Project\\backend"
.\\venv\\Scripts\\Activate.ps1
python main.py          # http://localhost:8000

# Terminal 2 — frontend
cd "D:\\Final Year Project\\frontend"
python -m streamlit run app.py   # http://localhost:8501
        """, language="bash")

        st.markdown("#### API endpoints")
        endpoints = [
            ("GET", "/health", "Health check — returns service status"),
            ("POST", "/api/ml/predict", "ML prediction from 13 clinical parameters"),
            ("POST", "/api/rag/chat", "RAG chatbot — answers grounded in guidelines"),
            ("POST", "/api/combined/analyze", "Integrated ML + RAG — personalized recommendations"),
        ]
        for method, path, desc in endpoints:
            color = "#166534" if method == "GET" else "#1d4ed8"
            bg = "#f0fdf4" if method == "GET" else "#eff6ff"
            st.markdown(f"""
            <div style="display:flex;align-items:baseline;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid #f1f5f9;">
                <span style="background:{bg};color:{color};border-radius:4px;padding:0.1rem 0.45rem;font-size:0.7rem;font-weight:600;letter-spacing:0.05em;flex-shrink:0;">{method}</span>
                <code style="font-size:0.82rem;">{path}</code>
                <span style="font-size:0.82rem;color:#64748b;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Live status")
        if backend_ok:
            try:
                info = requests.get(f"{BACKEND_URL}/", timeout=3).json()
                st.json({"status": info.get("status","online"), "version": info.get("version","1.0"), "components": info.get("components",{})})
            except:
                st.success("Backend is responding on port 8000.")
        else:
            st.warning("Backend offline.")


# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif page == "About":
    st.markdown('<h1 class="page-title">About this project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">AI-Driven Cardiovascular Disease Risk Prediction System · Final Year Project 2025</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-light" style="font-size:0.9rem;color:#334155;line-height:1.7;">
        This system was developed to demonstrate the integration of supervised machine learning 
        with retrieval-augmented generation for clinical decision support. The ML component 
        predicts cardiovascular disease risk; the RAG component grounds its recommendations 
        in peer-reviewed clinical guidelines rather than model weights alone. The key 
        contribution is the combined endpoint that makes the RAG response aware of the 
        patient's specific ML-derived risk profile.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Project team")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card-light" style="text-align:center;">
            <div style="width:44px;height:44px;border-radius:50%;background:#fef2f2;border:1px solid #fecaca;display:flex;align-items:center;justify-content:center;margin:0 auto 0.75rem;font-weight:600;color:#b91c1c;font-size:0.85rem;">H</div>
            <div style="font-weight:500;color:#0f1117;">Hamza</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">Roll No. 157</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card-light" style="text-align:center;">
            <div style="width:44px;height:44px;border-radius:50%;background:#fef2f2;border:1px solid #fecaca;display:flex;align-items:center;justify-content:center;margin:0 auto 0.75rem;font-weight:600;color:#b91c1c;font-size:0.85rem;">TA</div>
            <div style="font-weight:500;color:#0f1117;">Tufail Abbas</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">Roll No. 164</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card-light" style="text-align:center;">
            <div style="width:44px;height:44px;border-radius:50%;background:#eff6ff;border:1px solid #bfdbfe;display:flex;align-items:center;justify-content:center;margin:0 auto 0.75rem;font-weight:600;color:#1d4ed8;font-size:0.85rem;">RA</div>
            <div style="font-weight:500;color:#0f1117;">Dr. Rehman Ali</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;">Supervisor</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### References")
    refs = [
        "UCI Heart Disease Dataset — Cleveland Clinic Foundation",
        "2017 ACC/AHA Hypertension Guidelines — AHA",
        "CHD Clinical Management Guidelines — MUS_D1",
        "LangChain documentation — LangChain Inc.",
        "Scikit-learn: Machine Learning in Python — Pedregosa et al., 2011",
        "Groq · Llama 3.1 — Meta AI / Groq Inc.",
    ]
    for i, ref in enumerate(refs, 1):
        st.markdown(f"<span style='font-size:0.85rem;color:#475569;'>{i}. {ref}</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#94a3b8;">This system is intended for academic and research purposes only. It does not replace professional clinical judgment. Always consult a qualified healthcare professional for medical decisions.</p>', unsafe_allow_html=True)