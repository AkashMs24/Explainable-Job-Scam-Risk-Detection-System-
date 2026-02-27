import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(
    page_title="SCAMGUARD-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
fraud_model      = joblib.load(BASE_DIR / "fraud_model.pkl")
tfidf_vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
feature_names    = joblib.load(BASE_DIR / "feature_names.pkl")

# =====================================================
# FEATURE ENGINEERING
# =====================================================
urgency_words = ["urgent","immediate","limited","apply fast","hurry","few slots","act now"]
free_domains  = ["gmail.com","yahoo.com","outlook.com","hotmail.com"]

def urgency_score(text):
    text = str(text).lower()
    return sum(word in text for word in urgency_words)

def free_email_flag(text):
    text = str(text).lower()
    return int(any(domain in text for domain in free_domains))

# =====================================================
# GLOBAL CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*, *::before, *::after { box-sizing:border-box; }
html, body, [class*="css"], .stApp { background:#0a0b0f !important; color:#e8e8f0 !important; font-family:'Space Grotesk',sans-serif !important; }
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebarNav"] { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.block-container { padding:0 !important; max-width:100% !important; }
section.main > div { padding:0 !important; }
.stTextInput,[data-testid="stTextInput"],.stTextArea,[data-testid="stTextArea"],.stButton,[data-testid="stButton"] {
  position:absolute !important; opacity:0 !important; pointer-events:none !important;
  width:1px !important; height:1px !important; overflow:hidden !important; top:-9999px !important;
}
::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:#0a0b0f} ::-webkit-scrollbar-thumb{background:#333;border-radius:4px}
@keyframes fadeIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
for k,v in [("result",None),("history",[]),
            ("job_title",""),("job_company",""),("job_email",""),("job_salary",""),("job_desc","")]:
    if k not in st.session_state:
        st.session_state[k] = v

SAMPLES = {
    "scam":  {"title":"Work From Home – Earn $5000/week","company":"GlobalEarnings LLC",
              "email":"hr@globalearnings.gmail.com","salary":"$5000/week",
              "desc":"No experience needed! Work from home and earn big. Urgent hiring — only few slots left. Send your SSN and bank details to get started immediately. Act now, limited positions!"},
    "legit": {"title":"Software Engineer Intern","company":"Google",
              "email":"recruiting@google.com","salary":"$45/hr",
              "desc":"Join Google engineering team for a 12-week internship. Work on real products. Requirements: CS degree, Python/Java. Apply via careers.google.com. Structured hiring with technical interviews."}
}

# =====================================================
# HIDDEN INPUTS
# =====================================================
job_title_v   = st.text_input("t", key="job_title",   label_visibility="hidden")
job_company_v = st.text_input("c", key="job_company", label_visibility="hidden")
job_email_v   = st.text_input("e", key="job_email",   label_visibility="hidden")
job_salary_v  = st.text_input("s", key="job_salary",  label_visibility="hidden")
job_desc_v    = st.text_area("d",  key="job_desc",    label_visibility="hidden")

c1,c2,c3 = st.columns(3)
with c1: btn_scam  = st.button("Load Scam",    key="btn_scam")
with c2: btn_legit = st.button("Load Legit",   key="btn_legit")
with c3: btn_go    = st.button("Analyze",      key="btn_go")

if btn_scam:
    for k,v in SAMPLES["scam"].items():
        st.session_state[f"job_{k}"] = v
    st.rerun()
if btn_legit:
    for k,v in SAMPLES["legit"].items():
        st.session_state[f"job_{k}"] = v
    st.rerun()

if btn_go:
    t = st.session_state.get("job_title","")
    d = st.session_state.get("job_desc","")
    if t.strip() or d.strip():
        e  = st.session_state.get("job_email","")
        co = st.session_state.get("job_company","")
        sa = st.session_state.get("job_salary","")
        X_text     = tfidf_vectorizer.transform([t + " " + d])
        dl         = len(d)
        urg        = urgency_score(d)
        fem        = free_email_flag(e + " " + co)
        X_final    = hstack([X_text, np.array([[dl, urg, fem]])])
        fp         = fraud_model.predict_proba(X_final)[0][1]
        sm         = int(sa.strip() == "")
        rs         = round(min((0.60*fp + 0.15*min(urg/5,1) + 0.15*sm + 0.10*fem)*100, 100), 2)
        conf       = "Moderate" if 0.4 <= fp <= 0.6 else "High"
        if rs < 30:   lv,ck,vi,adv = "LOW","green","✅","Looks legitimate. Standard precautions apply."
        elif rs < 60: lv,ck,vi,adv = "MEDIUM","orange","⚠️","Proceed with caution. Avoid sharing personal information."
        else:         lv,ck,vi,adv = "HIGH","red","⛔","High scam risk detected. Strongly avoid applying."
        if urg > 2:  pd = "Urgency-driven language"
        elif fem:    pd = "Use of free email domain"
        elif sm:     pd = "Lack of salary transparency"
        else:        pd = "No dominant risk driver"
        flags = []
        if urg > 0: flags.append("Urgency-driven language detected")
        if sm:      flags.append("Salary information missing")
        if fem:     flags.append("Free/personal email domain used")
        if urg > 2 and fem: ctx = "This pattern strongly resembles mass internship scam campaigns."
        elif sm and dl < 300: ctx = "Short descriptions with missing salary often indicate low-effort scams."
        elif urg > 0: ctx = "Urgency-based language suggests pressure tactics commonly used in scams."
        else: ctx = "No dominant scam pattern detected based on known behavior."
        res = {"score":rs,"level":lv,"color_key":ck,"verdict_icon":vi,"advice":adv,
               "primary_driver":pd,"confidence":conf,"fraud_prob":round(fp*100,1),
               "flags":flags,"context":ctx,"title":t,"company":co}
        st.session_state["result"]  = res
        st.session_state["history"] = [res] + st.session_state["history"][:9]
        st.rerun()

# =====================================================
# RENDER VALUES
# =====================================================
r  = st.session_state.get("result", None)
tv = (st.session_state.get("job_title","") or "").replace('"',"'")
cv = (st.session_state.get("job_company","") or "").replace('"',"'")
ev = (st.session_state.get("job_email","") or "").replace('"',"'")
sv = (st.session_state.get("job_salary","") or "").replace('"',"'")
dv = (st.session_state.get("job_desc","") or "").replace('"',"'").replace("\n","<br>")

SC = {"red":"#f87171","orange":"#fbbf24","green":"#4ade80"}
GC = {"red":"rgba(220,38,38,0.25)","orange":"rgba(255,170,0,0.2)","green":"rgba(0,230,118,0.2)"}
BC = {"red":"rgba(220,38,38,0.35)","orange":"rgba(255,170,0,0.35)","green":"rgba(0,230,118,0.35)"}
BDGE = {"red":"badge-red","orange":"badge-orange","green":"badge-green"}

if r:
    sc = SC[r["color_key"]]; gc = GC[r["color_key"]]; bc = BC[r["color_key"]]; bdg = BDGE[r["color_key"]]
    flags_html = "".join([f'<div class="flag-item"><span class="dot dot-red"></span>{f}</div>' for f in r["flags"]]) \
                 if r["flags"] else '<div class="flag-item"><span class="dot dot-green"></span>No strong scam indicators detected</div>'
    ring_dash = (r["score"]/100)*314.16
    result_html = f"""
    <div class="result-card" style="border-color:{bc};box-shadow:0 0 28px {gc};">
      <div style="display:flex;align-items:center;gap:20px;margin-bottom:18px;">
        <div style="position:relative;width:120px;height:120px;flex-shrink:0;">
          <svg width="120" height="120" style="transform:rotate(-90deg);">
            <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="9"/>
            <circle cx="60" cy="60" r="50" fill="none" stroke="{sc}" stroke-width="9"
              stroke-dasharray="{ring_dash} 314.16" stroke-linecap="round"
              style="filter:drop-shadow(0 0 8px {sc});"/>
          </svg>
          <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <span style="font-family:'Bebas Neue',cursive;font-size:28px;color:{sc};letter-spacing:1px;">{r['score']}</span>
            <span style="font-family:'Space Mono',monospace;font-size:9px;color:rgba(255,255,255,0.4);">/100</span>
          </div>
        </div>
        <div style="flex:1;">
          <div style="font-family:'Bebas Neue',cursive;font-size:24px;letter-spacing:3px;color:{sc};margin-bottom:10px;">{r['verdict_icon']} {r['level']} RISK</div>
          <span class="badge {bdg}">{r['level']} RISK</span>&nbsp;
          <span class="badge badge-neutral">CONF: {r['confidence'].upper()}</span>
        </div>
      </div>
      <div class="prog-track"><div class="prog-fill" style="width:{r['score']}%;background:{sc};box-shadow:0 0 8px {sc};"></div></div>
    </div>
    <div style="display:flex;gap:10px;margin-bottom:12px;">
      <div class="mini-card"><div class="mini-label">Primary Risk Driver</div><div class="mini-value">{r['primary_driver']}</div></div>
      <div class="mini-card"><div class="mini-label">Fraud Probability</div><div class="mini-value">{r['fraud_prob']}%</div></div>
    </div>
    <div class="insight-box">
      <div class="insight-label">🧠 RISK CONTEXT INSIGHT</div>
      <div style="font-size:13px;color:rgba(255,255,255,0.75);line-height:1.6;">{r['context']}</div>
    </div>
    <div style="padding:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;margin-bottom:12px;">
      <div style="font-family:'Space Mono',monospace;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:2px;margin-bottom:10px;">⚡ WHY FLAGGED</div>
      {flags_html}
    </div>
    <div style="padding:14px;background:{gc};border:1px solid {bc};border-radius:10px;">
      <div style="font-family:'Space Mono',monospace;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:2px;margin-bottom:6px;">📋 RECOMMENDED ACTION</div>
      <div style="font-size:13px;color:{sc};font-weight:600;">{r['advice']}</div>
    </div>"""
else:
    result_html = """
    <div style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;opacity:.25;padding:60px 0;">
      <div style="font-size:64px;">🛡️</div>
      <div style="font-family:'Bebas Neue',cursive;font-size:22px;letter-spacing:3px;">AWAITING ANALYSIS</div>
      <div style="font-family:'Space Mono',monospace;font-size:11px;color:rgba(255,255,255,0.5);text-align:center;">Submit a job posting to begin intelligence scan</div>
    </div>"""

history_html = ""
for h in st.session_state.get("history",[]):
    hc = SC[h["color_key"]]; hbc = BC[h["color_key"]]
    history_html += f"""
    <div style="display:flex;align-items:center;gap:16px;padding:14px;background:rgba(255,255,255,0.03);border:1px solid {hbc};border-radius:12px;margin-bottom:10px;">
      <div style="font-family:'Bebas Neue',cursive;font-size:32px;color:{hc};min-width:60px;text-align:center;">{h['score']}</div>
      <div style="flex:1;">
        <div style="font-size:14px;font-weight:600;color:#e8e8f0;margin-bottom:2px;">{h['title'] or 'Untitled'}</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.4);">{h['company'] or 'Unknown'}</div>
      </div>
      <span style="font-family:'Space Mono',monospace;font-size:11px;padding:4px 12px;border-radius:20px;background:rgba(255,255,255,0.05);border:1px solid {hbc};color:{hc};letter-spacing:1px;">{h['verdict_icon']} {h['level']} RISK</span>
    </div>"""
if not history_html:
    history_html = '<div style="text-align:center;padding:60px;color:rgba(255,255,255,0.2);font-family:\'Space Mono\',monospace;font-size:13px;">No scans yet.</div>'

st.markdown(f"""
<style>
.sg-wrap{{min-height:100vh;background:#0a0b0f;background-image:radial-gradient(ellipse at 20% 0%,rgba(220,38,38,0.07) 0%,transparent 50%),radial-gradient(ellipse at 80% 100%,rgba(59,130,246,0.05) 0%,transparent 50%);}}
.topbar{{border-bottom:1px solid rgba(255,255,255,0.06);padding:14px 32px;display:flex;align-items:center;gap:16px;background:rgba(0,0,0,0.45);backdrop-filter:blur(10px);position:sticky;top:0;z-index:100;}}
.logo-box{{width:38px;height:38px;background:linear-gradient(135deg,#dc2626,#7f1d1d);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:20px;}}
.tab-btn{{padding:6px 18px;border-radius:6px;border:1px solid;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;cursor:pointer;transition:all .2s;background:transparent;}}
.tab-active{{border-color:#dc2626!important;background:rgba(220,38,38,0.15)!important;color:#fff!important;}}
.tab-inactive{{border-color:rgba(255,255,255,0.1);color:rgba(255,255,255,0.45);}}
.main-grid{{max-width:1100px;margin:0 auto;padding:32px 24px;display:grid;grid-template-columns:1fr 1fr;gap:24px;}}
.section-label{{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:3px;color:#dc2626;text-transform:uppercase;margin-bottom:14px;}}
.field-label{{display:block;font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:6px;}}
.fake-input{{width:100%;padding:10px 14px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);border-radius:8px;color:#e8e8f0;font-size:13px;font-family:'Space Grotesk',sans-serif;margin-bottom:14px;min-height:40px;cursor:pointer;word-break:break-word;transition:border-color .2s;}}
.fake-input:hover{{border-color:rgba(220,38,38,0.4);}}
.fake-input.empty{{color:rgba(255,255,255,0.2);}}
.fake-textarea{{min-height:130px;line-height:1.6;}}
.analyze-btn{{width:100%;padding:14px;background:linear-gradient(135deg,#dc2626,#b91c1c);border:none;border-radius:10px;color:#fff;font-family:'Bebas Neue',cursive;font-size:15px;letter-spacing:3px;box-shadow:0 4px 24px rgba(220,38,38,0.3);cursor:pointer;transition:all .3s;margin-top:4px;}}
.analyze-btn:hover{{box-shadow:0 6px 32px rgba(220,38,38,0.5);transform:translateY(-1px);}}
.sample-btn{{padding:7px 14px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);color:rgba(255,255,255,0.5);font-family:'Space Mono',monospace;font-size:11px;letter-spacing:1px;cursor:pointer;transition:all .2s;}}
.sample-btn:hover{{border-color:#dc2626;color:#fff;}}
.result-card{{background:rgba(255,255,255,0.03);border-radius:16px;padding:24px;margin-bottom:14px;animation:fadeIn .5s ease;}}
.badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:1.5px;font-weight:700;}}
.badge-red{{background:rgba(220,38,38,0.15);border:1px solid rgba(220,38,38,0.4);color:#f87171;}}
.badge-orange{{background:rgba(255,170,0,0.15);border:1px solid rgba(255,170,0,0.4);color:#fbbf24;}}
.badge-green{{background:rgba(0,230,118,0.15);border:1px solid rgba(0,230,118,0.4);color:#4ade80;}}
.badge-neutral{{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.5);}}
.mini-card{{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:12px 14px;flex:1;}}
.mini-label{{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:rgba(255,255,255,0.35);margin-bottom:4px;text-transform:uppercase;}}
.mini-value{{font-size:13px;color:#e8e8f0;font-weight:500;}}
.prog-track{{background:rgba(255,255,255,0.07);border-radius:4px;height:6px;overflow:hidden;}}
.prog-fill{{height:100%;border-radius:4px;transition:width 1s ease;}}
.insight-box{{padding:14px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:10px;margin-bottom:12px;}}
.insight-label{{font-family:'Space Mono',monospace;font-size:10px;color:#818cf8;letter-spacing:2px;margin-bottom:6px;}}
.flag-item{{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:6px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);margin-bottom:6px;font-size:12px;color:rgba(255,255,255,0.7);}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block;}}
.dot-red{{background:#f87171;}}
.dot-green{{background:#4ade80;}}
.disclaimer{{font-family:'Space Mono',monospace;font-size:10px;color:rgba(255,255,255,0.2);text-align:center;margin-top:10px;}}
</style>

<div class="sg-wrap">
  <!-- TOPBAR -->
  <div class="topbar">
    <div class="logo-box">🛡️</div>
    <div>
      <div style="font-family:'Bebas Neue',cursive;font-size:18px;letter-spacing:3px;color:#fff;">SCAMGUARD<span style="color:#dc2626">-AI</span></div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:rgba(255,255,255,0.35);letter-spacing:2px;">INTELLIGENCE SYSTEM v2.0</div>
    </div>
    <div style="flex:1;"></div>
    <button class="tab-btn tab-active"   id="btn-analyze"  onclick="switchTab('analyze')">ANALYZE</button>
    <button class="tab-btn tab-inactive" id="btn-history"  onclick="switchTab('history')">HISTORY</button>
  </div>

  <!-- ANALYZE TAB -->
  <div id="tab-analyze" class="main-grid">
    <!-- LEFT -->
    <div>
      <div class="section-label">// Job Posting Input</div>
      <p style="font-size:13px;color:rgba(255,255,255,0.4);line-height:1.6;margin-bottom:16px;">
        Paste a job listing below. Our hybrid ML + rule-based engine will assess scam probability.
      </p>
      <div style="display:flex;gap:8px;margin-bottom:18px;">
        <button class="sample-btn" onclick="loadSample('scam')">⛔ SAMPLE 1: SCAM</button>
        <button class="sample-btn" onclick="loadSample('legit')">✅ SAMPLE 2: LEGIT</button>
      </div>

      <label class="field-label">Job Title</label>
      <div class="fake-input {'empty' if not tv else ''}" id="disp-title" onclick="openEdit('title',this)">{tv if tv else 'e.g. Software Engineer Intern'}</div>

      <label class="field-label">Company Name</label>
      <div class="fake-input {'empty' if not cv else ''}" id="disp-company" onclick="openEdit('company',this)">{cv if cv else 'e.g. Acme Corp'}</div>

      <label class="field-label">Contact Email</label>
      <div class="fake-input {'empty' if not ev else ''}" id="disp-email" onclick="openEdit('email',this)">{ev if ev else 'e.g. hr@company.com'}</div>

      <label class="field-label">Salary / Compensation</label>
      <div class="fake-input {'empty' if not sv else ''}" id="disp-salary" onclick="openEdit('salary',this)">{sv if sv else 'e.g. $20/hr or $5000/week'}</div>

      <label class="field-label">Job Description</label>
      <div class="fake-input fake-textarea {'empty' if not dv else ''}" id="disp-desc" onclick="openEdit('desc',this)">{dv if dv else 'Paste the full job description here...'}</div>

      <!-- inline edit modal -->
      <div id="edit-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:999;align-items:center;justify-content:center;">
        <div style="background:#13141a;border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:24px;width:90%;max-width:520px;">
          <div id="edit-label" style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:rgba(255,255,255,0.4);margin-bottom:10px;text-transform:uppercase;"></div>
          <textarea id="edit-ta" rows="5" style="width:100%;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);border-radius:8px;color:#e8e8f0;font-size:13px;font-family:'Space Grotesk',sans-serif;padding:10px 14px;resize:vertical;"></textarea>
          <div style="display:flex;gap:10px;margin-top:12px;justify-content:flex-end;">
            <button onclick="closeEdit()" style="padding:8px 20px;background:transparent;border:1px solid rgba(255,255,255,0.15);border-radius:8px;color:rgba(255,255,255,0.5);cursor:pointer;font-family:'Space Mono',monospace;font-size:12px;">CANCEL</button>
            <button onclick="saveEdit()" style="padding:8px 20px;background:linear-gradient(135deg,#dc2626,#b91c1c);border:none;border-radius:8px;color:#fff;cursor:pointer;font-family:'Space Mono',monospace;font-size:12px;letter-spacing:1px;">SAVE</button>
          </div>
        </div>
      </div>

      <button class="analyze-btn" onclick="runAnalyze()">🔍  ANALYZE SCAM RISK</button>
      <div class="disclaimer">⚠ Guidance only — always verify independently</div>
    </div>

    <!-- RIGHT -->
    <div>{result_html}</div>
  </div>

  <!-- HISTORY TAB -->
  <div id="tab-history" style="display:none;max-width:1100px;margin:0 auto;padding:32px 24px;">
    <div class="section-label">// SCAN HISTORY ({len(st.session_state.get('history',[]))})</div>
    {history_html}
  </div>

</div>

<script>
const SAMPLES = {{
  scam:  {{title:"Work From Home – Earn $5000/week",company:"GlobalEarnings LLC",email:"hr@globalearnings.gmail.com",salary:"$5000/week",desc:"No experience needed! Work from home and earn big. Urgent hiring — only few slots left. Send your SSN and bank details to get started immediately. Act now, limited positions!"}},
  legit: {{title:"Software Engineer Intern",company:"Google",email:"recruiting@google.com",salary:"$45/hr",desc:"Join Google engineering team for a 12-week internship. Work on real products. Requirements: CS degree, Python/Java. Apply via careers.google.com. Structured hiring with technical interviews."}}
}};

let currentField = null;
const fieldData = {{ title:"{tv}", company:"{cv}", email:"{ev}", salary:"{sv}", desc:"{dv.replace('<br>','\\n')}" }};

function getSTInput(key) {{
  const all = document.querySelectorAll('input, textarea');
  for(const el of all) {{
    if(el.getAttribute('aria-label')===key || el.id===key || el.name===key) return el;
  }}
  // try label text match
  for(const el of all) {{
    const lbl = document.querySelector(`label[for="${{el.id}}"]`);
    if(lbl && lbl.textContent.trim().toLowerCase()===key) return el;
  }}
  return null;
}}

function setSTVal(key, val) {{
  // find the hidden streamlit input by trying multiple selectors
  const frames = [document, ...(window.parent !== window ? [window.parent.document] : [])];
  for(const doc of frames) {{
    const inputs = doc.querySelectorAll('input, textarea');
    for(const el of inputs) {{
      if(el.style.top==='-9999px' || el.style.opacity==='0') {{
        // find by position in DOM order
      }}
    }}
  }}
  // Most reliable: find streamlit widget by label
  const allEls = document.querySelectorAll('[data-testid] input, [data-testid] textarea');
  // Use native setter to trigger React onChange
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value');
  const nativeTextAreaSetter   = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value');
  for(const el of document.querySelectorAll('input, textarea')) {{
    try {{
      const setter = el.tagName==='TEXTAREA' ? nativeTextAreaSetter?.set : nativeInputValueSetter?.set;
      if(setter) {{
        setter.call(el, val);
        el.dispatchEvent(new Event('input',{{bubbles:true}}));
        el.dispatchEvent(new Event('change',{{bubbles:true}}));
      }}
    }} catch(e) {{}}
  }}
}}

function openEdit(field, el) {{
  currentField = field;
  const modal = document.getElementById('edit-modal');
  const ta    = document.getElementById('edit-ta');
  const lbl   = document.getElementById('edit-label');
  lbl.textContent = field.toUpperCase();
  ta.value = fieldData[field] || '';
  ta.rows  = (field==='desc') ? 8 : 3;
  modal.style.display = 'flex';
  setTimeout(()=>ta.focus(),100);
}}

function closeEdit() {{
  document.getElementById('edit-modal').style.display = 'none';
}}

function saveEdit() {{
  const val = document.getElementById('edit-ta').value;
  fieldData[currentField] = val;
  const disp = document.getElementById('disp-' + currentField);
  if(disp) {{
    disp.innerHTML = val ? val.replace(/\\n/g,'<br>') : '';
    disp.classList.toggle('empty', !val);
  }}
  // try to set hidden streamlit input
  syncToStreamlit(currentField, val);
  closeEdit();
}}

function syncToStreamlit(field, val) {{
  // map field to streamlit key label
  const keyMap = {{title:'t',company:'c',email:'e',salary:'s',desc:'d'}};
  const label = keyMap[field];
  const inputs = document.querySelectorAll('input, textarea');
  let found = false;
  for(const el of inputs) {{
    // check aria-label or placeholder pattern
    const ph = el.getAttribute('placeholder') || '';
    if(ph.includes(label) || el.getAttribute('aria-label')===label) {{
      triggerChange(el, val); found=true; break;
    }}
  }}
  if(!found) {{
    // fallback: set all hidden ones by order
    const hidden = [...document.querySelectorAll('input,textarea')].filter(e=>e.style.position==='absolute'||e.offsetParent===null);
    const orderMap = {{title:0,company:1,email:2,salary:3,desc:4}};
    const idx = orderMap[field];
    if(hidden[idx]) triggerChange(hidden[idx], val);
  }}
}}

function triggerChange(el, val) {{
  const proto = el.tagName==='TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto,'value')?.set;
  if(setter) setter.call(el, val);
  el.dispatchEvent(new Event('input',  {{bubbles:true}}));
  el.dispatchEvent(new Event('change', {{bubbles:true}}));
}}

function loadSample(type) {{
  const s = SAMPLES[type];
  ['title','company','email','salary','desc'].forEach(f=>{{
    fieldData[f] = s[f];
    const disp = document.getElementById('disp-'+f);
    if(disp){{
      disp.innerHTML = f==='desc' ? s[f].replace(/\\n/g,'<br>') : s[f];
      disp.classList.remove('empty');
    }}
    syncToStreamlit(f, s[f]);
  }});
  // click hidden load scam/legit button
  setTimeout(()=>{{
    const btns = document.querySelectorAll('button');
    const label = type==='scam' ? 'Load Scam' : 'Load Legit';
    for(const b of btns) {{
      if(b.textContent.trim()===label) {{ b.click(); return; }}
    }}
  }}, 300);
}}

function runAnalyze() {{
  // ensure all values are synced, then click hidden Analyze button
  ['title','company','email','salary','desc'].forEach(f=>syncToStreamlit(f, fieldData[f]||''));
  setTimeout(()=>{{
    const btns = document.querySelectorAll('button');
    for(const b of btns){{
      if(b.textContent.trim()==='Analyze'){{ b.click(); return; }}
    }}
  }}, 400);
}}

function switchTab(t) {{
  document.getElementById('tab-analyze').style.display = t==='analyze'?'grid':'none';
  document.getElementById('tab-history').style.display = t==='history'?'block':'none';
  document.getElementById('btn-analyze').className = 'tab-btn '+(t==='analyze'?'tab-active':'tab-inactive');
  document.getElementById('btn-history').className = 'tab-btn '+(t==='history'?'tab-active':'tab-inactive');
}}

// click outside modal to close
document.getElementById('edit-modal')?.addEventListener('click', function(e){{
  if(e.target===this) closeEdit();
}});
</script>
""", unsafe_allow_html=True)
