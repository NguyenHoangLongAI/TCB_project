"""
Streamlit Chat Client – Techcomlife Insurance Advisor
Port 9000  |  Connects to Embedding API (8000) + RAG API (8501)

Features:
  • Register / Login (user_groups collection)
  • Choose RAG workflow (only 1: Techcomlife Insurance Advisor)
  • Chat with streaming via localhost:8501/chat
  • Displays token usage and cumulative cost

Run:
    streamlit run streamlit_chat.py --server.port 9000
"""

import streamlit as st
import requests, json, time
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_API = "http://localhost:8000"
RAG_API       = "http://localhost:8501"

st.set_page_config(
    page_title="Techcomlife AI Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

/* Page background */
.stApp { background: linear-gradient(135deg, #0a1628 0%, #0d2144 50%, #0a1628 100%); }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #051020 0%, #0c1f3d 100%);
    border-right: 1px solid #1e3a5f;
}

/* Cards */
.card {
    background: rgba(14, 30, 60, 0.85);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* Brand */
.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8c97a;
    letter-spacing: 0.5px;
}
.brand-sub {
    font-size: 0.78rem;
    color: #7a9cc0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: -4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #0d3fa6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1f6ae0, #1247b5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,86,219,0.35) !important;
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #1a3a6e, #0f2550);
    border: 1px solid #2a4f8a;
    border-radius: 14px 14px 4px 14px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    color: #cce0ff;
    font-size: 0.9rem;
}
.chat-assistant {
    background: rgba(14, 30, 60, 0.9);
    border: 1px solid #1e3a5f;
    border-radius: 14px 14px 14px 4px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    color: #d4e8ff;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-label {
    font-size: 0.72rem;
    color: #5a7fa0;
    margin-bottom: 2px;
}

/* Token badge */
.token-badge {
    display: inline-block;
    background: rgba(232,201,122,0.15);
    border: 1px solid rgba(232,201,122,0.4);
    color: #e8c97a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* Refs */
.ref-item {
    background: rgba(26,86,219,0.1);
    border: 1px solid rgba(26,86,219,0.25);
    border-radius: 6px;
    padding: 4px 10px;
    margin: 3px 0;
    font-size: 0.78rem;
    color: #7aabdd;
}

/* Input */
.stTextInput > div > div > input, .stTextArea textarea {
    background: rgba(5,16,40,0.8) !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #cce0ff !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 2px rgba(26,86,219,0.2) !important;
}

/* Select */
.stSelectbox > div > div {
    background: rgba(5,16,40,0.8) !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #cce0ff !important;
}

/* Success / Error */
.stAlert { border-radius: 8px !important; }

/* Divider */
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("logged_in",      False),
    ("user_info",      None),
    ("chat_history",   []),          # list of {role, content, tokens}
    ("total_tokens",   0),
    ("page",           "login"),     # login | register | chat
    ("workflow",       "techcomlife_insurance"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def api_login(username: str, password: str) -> Optional[dict]:
    try:
        r = requests.post(f"{EMBEDDING_API}/auth/login",
                          json={"username": username, "password": password}, timeout=10)
        if r.status_code == 200:
            return r.json().get("user")
        return None
    except Exception:
        return None


def api_register(user_id, username, password, group_id, company_id="", department_id="") -> tuple:
    try:
        r = requests.post(f"{EMBEDDING_API}/create/user", json={
            "user_id": user_id, "username": username, "password": password,
            "group_id": group_id, "company_id": company_id, "department_id": department_id,
        }, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)


def api_get_user(user_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{EMBEDDING_API}/user/{user_id}", timeout=5)
        if r.status_code == 200:
            return r.json().get("user")
        return None
    except Exception:
        return None


def chat_streaming(question: str, history: list, user_id: str):
    """
    Call RAG API in streaming mode.
    Returns (full_answer_text, references, tokens_used).
    """
    payload = {
        "question": question,
        "history":  history,
        "stream":   True,
        "user_id":  user_id,
    }
    full_text   = ""
    references  = []
    tokens_used = 0

    try:
        with requests.post(f"{RAG_API}/chat", json=payload, stream=True, timeout=90) as resp:
            if resp.status_code != 200:
                return f"[API Error {resp.status_code}]", [], 0

            placeholder = st.empty()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    t    = data.get("type")
                    if t == "chunk" and data.get("content"):
                        full_text += data["content"]
                        placeholder.markdown(
                            f'<div class="chat-assistant">{full_text}▌</div>',
                            unsafe_allow_html=True,
                        )
                    elif t == "references":
                        references = data.get("references", [])
                    elif t == "end":
                        tu = data.get("token_usage") or {}
                        tokens_used = tu.get("total_tokens", 0)
                        placeholder.empty()
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"[Connection error: {e}]", [], 0

    return full_text, references, tokens_used

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="brand-title">🛡️ Techcomlife</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI Insurance Advisor</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.logged_in and st.session_state.user_info:
        u = st.session_state.user_info
        st.markdown(f"""
        <div class="card">
            <div style="color:#e8c97a;font-weight:600;font-size:0.9rem;">👤 {u.get('username','')}</div>
            <div style="color:#5a7fa0;font-size:0.76rem;margin-top:4px;">ID: {u.get('user_id','')}</div>
            <div style="color:#5a7fa0;font-size:0.76rem;">Nhóm: {u.get('group_id','')}</div>
            {f'<div style="color:#5a7fa0;font-size:0.76rem;">Công ty: {u.get("company_id","")}</div>' if u.get('company_id') else ''}
            {f'<div style="color:#5a7fa0;font-size:0.76rem;">Phòng: {u.get("department_id","")}</div>' if u.get('department_id') else ''}
        </div>
        """, unsafe_allow_html=True)

        # Token counter
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="color:#5a7fa0;font-size:0.75rem;margin-bottom:4px;">TOKENS ĐÃ SỬ DỤNG (phiên)</div>
            <div class="token-badge">🔢 {st.session_state.total_tokens:,} tokens</div>
        </div>
        """, unsafe_allow_html=True)

        # Workflow selector
        st.markdown("**Chọn RAG Workflow**")
        workflow_options = {
            "techcomlife_insurance": "Tori AI Techcomlife",
            "search_law_TCB": "Search Legal Assistant AI"
        }
        selected = st.selectbox(
            "Workflow", list(workflow_options.keys()),
            format_func=lambda k: workflow_options[k], label_visibility="collapsed",
        )
        st.session_state.workflow = selected
        st.markdown(f"<small style='color:#5a7fa0;'>Knowledge Base: <code>Techcomlife documents database</code></small>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🚪 Đăng xuất"):
            for k in ["logged_in","user_info","chat_history","total_tokens"]:
                if k == "logged_in":         st.session_state[k] = False
                elif k == "user_info":       st.session_state[k] = None
                elif k == "chat_history":    st.session_state[k] = []
                elif k == "total_tokens":    st.session_state[k] = 0
            st.session_state.page = "login"
            st.rerun()

    else:
        st.markdown("Đăng nhập để sử dụng hệ thống tư vấn bảo hiểm AI.")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔑 Đăng nhập", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
        with col2:
            if st.button("📝 Đăng ký", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

# ── LOGIN PAGE ────────────────────────────────────────────────────────────────

if st.session_state.page == "login" and not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<h2 style="color:#e8c97a;font-family:\'Playfair Display\',serif;text-align:center;">Đăng nhập</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a7fa0;text-align:center;margin-bottom:2rem;">Hệ thống tư vấn bảo hiểm nhân thọ Techcomlife</p>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập", placeholder="Nhập tên đăng nhập …")
            password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu …")
            submitted = st.form_submit_button("🔑 Đăng nhập", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Vui lòng nhập đầy đủ thông tin.")
            else:
                with st.spinner("Đang xác thực …"):
                    user = api_login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user
                    st.session_state.page      = "chat"
                    st.success(f"Chào mừng, **{user.get('username','')}**!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        st.markdown("---")
        st.markdown('<p style="text-align:center;color:#5a7fa0;font-size:0.85rem;">Chưa có tài khoản?</p>', unsafe_allow_html=True)
        if st.button("📝 Đăng ký ngay", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

# ── REGISTER PAGE ─────────────────────────────────────────────────────────────

elif st.session_state.page == "register" and not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<h2 style="color:#e8c97a;font-family:\'Playfair Display\',serif;text-align:center;">Đăng ký tài khoản</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a7fa0;text-align:center;margin-bottom:1.5rem;">Tạo tài khoản để sử dụng hệ thống</p>', unsafe_allow_html=True)

        with st.form("register_form"):
            user_id       = st.text_input("User ID *", placeholder="VD: sell_user_001")
            username      = st.text_input("Tên đăng nhập *", placeholder="VD: nguyen_van_a")
            password      = st.text_input("Mật khẩu *", type="password", placeholder="Mật khẩu …")
            confirm_pw    = st.text_input("Xác nhận mật khẩu *", type="password")

            st.markdown("**Thông tin phân nhóm**")
            group_id      = st.text_input("Group ID *", value="Techcombank_group", placeholder="VD: Techcombank_group")
            company_id    = st.text_input("Company ID", placeholder="VD: Techcomlife (để trống = toàn group)")
            department_id = st.text_input("Department ID", placeholder="VD: Sell_Techcomlife (để trống = toàn công ty)")

            submitted = st.form_submit_button("📝 Tạo tài khoản", use_container_width=True)

        if submitted:
            errors = []
            if not user_id:    errors.append("User ID")
            if not username:   errors.append("Tên đăng nhập")
            if not password:   errors.append("Mật khẩu")
            if not group_id:   errors.append("Group ID")
            if password != confirm_pw:
                st.error("Mật khẩu xác nhận không khớp.")
            elif errors:
                st.error(f"Vui lòng điền: {', '.join(errors)}")
            else:
                with st.spinner("Đang tạo tài khoản …"):
                    ok, result = api_register(
                        user_id.strip(), username.strip(), password,
                        group_id.strip(), company_id.strip(), department_id.strip(),
                    )
                if ok:
                    st.success(f"✅ Tài khoản **{username}** đã được tạo! (token ban đầu: 0)")
                    time.sleep(1)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(f"❌ Lỗi: {result}")

        st.markdown("---")
        if st.button("← Quay lại đăng nhập", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# ── CHAT PAGE ─────────────────────────────────────────────────────────────────

elif st.session_state.logged_in:
    st.session_state.page = "chat"
    user = st.session_state.user_info

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
        <div>
            <h2 style="color:#e8c97a;font-family:'Playfair Display',serif;margin:0;">
                🛡️ Chuyên viên tư vấn bảo hiểm Techcomlife
            </h2>
            <p style="color:#5a7fa0;margin:0;font-size:0.82rem;">
                AI Workflow · Knowledge Base: <code>Techcomlife documents database</code> · 
                User: <strong style="color:#7aabdd;">{user.get('username','')}</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#3a5a7a;">
                <div style="font-size:3rem;">🤖</div>
                <p style="font-size:0.95rem;margin-top:0.5rem;">
                    Xin chào! Tôi là trợ lý tư vấn bảo hiểm Techcomlife.<br>
                    Bạn có thể hỏi tôi về các sản phẩm bảo hiểm, quyền lợi, phí bảo hiểm…
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role    = msg["role"]
                content = msg["content"]
                tokens  = msg.get("tokens", 0)
                refs    = msg.get("references", [])

                if role == "user":
                    st.markdown(f"""
                    <div class="chat-label" style="text-align:right;">Bạn</div>
                    <div class="chat-user">{content}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-label">🤖 Trợ lý Techcomlife</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)

                    # Token + refs
                    meta_parts = []
                    if tokens:
                        meta_parts.append(f'<span class="token-badge">🔢 {tokens:,} tokens</span>')
                    if refs:
                        ref_html = " ".join(
                            f'<span class="ref-item">📎 {r.get("filename") or r.get("document_id","")}</span>'
                            for r in refs if r.get("filename") or r.get("document_id")
                        )
                        if ref_html:
                            meta_parts.append(ref_html)
                    if meta_parts:
                        st.markdown(f'<div style="margin-top:4px;margin-bottom:8px;">{" ".join(meta_parts)}</div>', unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_area(
            "Câu hỏi của bạn",
            key="user_input",
            placeholder="Nhập câu hỏi về bảo hiểm Techcomlife …  (Shift+Enter để xuống dòng)",
            height=80,
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send_btn = st.button("📤 Gửi", use_container_width=True)

    if send_btn and user_input and user_input.strip():
        question = user_input.strip()

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": question, "tokens": 0, "references": []})

        # Build history for API
        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]   # exclude latest user msg
        ]

        # Stream answer
        with st.spinner("Đang truy xuất thông tin và tạo câu trả lời …"):
            answer, refs, tokens = chat_streaming(question, api_history, user.get("user_id",""))

        # Update session totals
        st.session_state.total_tokens += tokens

        # Append assistant message
        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "tokens": tokens, "references": refs,
        })

        # Refresh user token count from DB
        updated_user = api_get_user(user.get("user_id",""))
        if updated_user:
            st.session_state.user_info = updated_user

        st.rerun()