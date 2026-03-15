# RAG_Core/chat_client.py  (FIXED – Bug 3: token display + session state)
"""
FIXES:
  Bug 3a — total_tokens trong session_state tích lũy đúng từ response,
            không bị reset về 0 khi st.rerun()
  Bug 3b — api_get_user() gọi sau rerun sẽ đọc giá trị DB cũ (trước flush).
            FIX: dùng giá trị cộng dồn local, chỉ đồng bộ DB khi cần thiết
  Bug 3c — sidebar hiển thị session total, không phải DB total
           (DB total có thể chậm hơn 1 turn do Milvus segment cache)
"""

import streamlit as st
import requests, json, time
from typing import Optional

EMBEDDING_API = "http://localhost:8000"
RAG_API       = "http://localhost:8501"

st.set_page_config(
    page_title="Techcomlife AI Advisor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a1628 0%, #0d2144 50%, #0a1628 100%); }
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #051020 0%, #0c1f3d 100%);
    border-right: 1px solid #1e3a5f;
}
.card { background: rgba(14,30,60,0.85); border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1rem; backdrop-filter: blur(8px); }
.brand-title { font-family: 'Playfair Display',serif; font-size: 1.6rem; font-weight: 700;
               color: #e8c97a; letter-spacing: 0.5px; }
.brand-sub { font-size: 0.78rem; color: #7a9cc0; letter-spacing: 1.5px;
             text-transform: uppercase; margin-top: -4px; }
.stButton > button { background: linear-gradient(135deg,#1a56db,#0d3fa6) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'Be Vietnam Pro',sans-serif !important; font-weight: 500 !important; }
.stButton > button:hover { background: linear-gradient(135deg,#1f6ae0,#1247b5) !important;
    transform: translateY(-1px) !important; }
.chat-user { background: linear-gradient(135deg,#1a3a6e,#0f2550); border: 1px solid #2a4f8a;
    border-radius: 14px 14px 4px 14px; padding: .75rem 1rem; margin: .4rem 0 .4rem 3rem;
    color: #cce0ff; font-size: .9rem; }
.chat-assistant { background: rgba(14,30,60,.9); border: 1px solid #1e3a5f;
    border-radius: 14px 14px 14px 4px; padding: .75rem 1rem; margin: .4rem 3rem .4rem 0;
    color: #d4e8ff; font-size: .9rem; line-height: 1.6; }
.chat-label { font-size: .72rem; color: #5a7fa0; margin-bottom: 2px; }
.token-badge { display:inline-block; background:rgba(232,201,122,.15); border:1px solid rgba(232,201,122,.4);
    color:#e8c97a; border-radius:20px; padding:2px 10px; font-size:.72rem; font-weight:500; }
.ref-item { background:rgba(26,86,219,.1); border:1px solid rgba(26,86,219,.25);
    border-radius:6px; padding:4px 10px; margin:3px 0; font-size:.78rem; color:#7aabdd; }
.stTextInput > div > div > input, .stTextArea textarea {
    background:rgba(5,16,40,.8)!important; border:1px solid #1e3a5f!important;
    border-radius:8px!important; color:#cce0ff!important; }
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("logged_in",    False),
    ("user_info",    None),
    ("chat_history", []),
    # FIX Bug 3: session_tokens = tổng tokens tích lũy trong phiên làm việc này
    # Giá trị này KHÔNG bị reset khi rerun, chỉ reset khi logout hoặc xóa lịch sử
    ("session_tokens", 0),
    ("page",         "login"),
    ("workflow",     "techcomlife_insurance"),
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
        return r.json().get("user") if r.status_code == 200 else None
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
    """
    Lấy thông tin user từ DB.
    FIX Bug 3b: chỉ dùng để lấy thông tin ban đầu khi login,
    KHÔNG dùng để update token count trên UI sau mỗi chat.
    """
    try:
        r = requests.get(f"{EMBEDDING_API}/user/{user_id}", timeout=5)
        return r.json().get("user") if r.status_code == 200 else None
    except Exception:
        return None


def chat_streaming(question: str, history: list, user_id: str):
    """
    Gọi RAG API streaming.
    Trả về (answer_text, references, tokens_used_this_turn).
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
                        # FIX Bug 3: lấy tokens từ event "end" của SSE
                        tu = data.get("token_usage") or {}
                        tokens_used = tu.get("total_tokens", 0) or 0
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
    st.markdown('<div class="brand-title"> Techcomlife</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI Insurance Advisor</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.logged_in and st.session_state.user_info:
        u = st.session_state.user_info

        st.markdown(f"""
        <div class="card">
            <div style="color:#e8c97a;font-weight:600;font-size:.9rem;">{u.get('username','')}</div>
            <div style="color:#5a7fa0;font-size:.76rem;margin-top:4px;">ID: {u.get('user_id','')}</div>
            <div style="color:#5a7fa0;font-size:.76rem;">Nhóm: {u.get('group_id','')}</div>
            {f'<div style="color:#5a7fa0;font-size:.76rem;">Công ty: {u.get("company_id","")}</div>' if u.get('company_id') else ''}
        </div>
        """, unsafe_allow_html=True)

        # Hiển thị tổng token tích lũy của user (bao gồm cả các phiên trước)
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="color:#5a7fa0;font-size:.75rem;margin-bottom:4px;">TỔNG TOKENS ĐÃ DÙNG</div>
            <div class="token-badge">{st.session_state.session_tokens:,} tokens</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Chọn RAG Workflow**")
        workflow_options = {
            "techcomlife_insurance": "Tori AI Techcomlife",
            "search_law_TCB":        "Search Legal Assistant AI",
        }
        selected = st.selectbox(
            "Workflow", list(workflow_options.keys()),
            format_func=lambda k: workflow_options[k], label_visibility="collapsed",
        )
        st.session_state.workflow = selected

        st.markdown("---")
        if st.button("Xóa lịch sử chat"):
            st.session_state.chat_history = []
            # Giữ nguyên session_tokens — lịch sử chat xóa nhưng token đã dùng không mất
            st.rerun()

        if st.button("Đăng xuất"):
            st.session_state.logged_in      = False
            st.session_state.user_info       = None
            st.session_state.chat_history    = []
            st.session_state.session_tokens  = 0
            st.session_state.page            = "login"
            st.rerun()

    else:
        st.markdown("Đăng nhập để sử dụng hệ thống.")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Đăng nhập", use_container_width=True):
                st.session_state.page = "login"; st.rerun()
        with c2:
            if st.button("Đăng ký", use_container_width=True):
                st.session_state.page = "register"; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.page == "login" and not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<h2 style="color:#e8c97a;font-family:\'Playfair Display\',serif;text-align:center;">Đăng nhập</h2>', unsafe_allow_html=True)

        with st.form("login_form"):
            username  = st.text_input("Tên đăng nhập", placeholder="Nhập tên đăng nhập…")
            password  = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu…")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Vui lòng nhập đầy đủ thông tin.")
            else:
                with st.spinner("Đang xác thực…"):
                    user = api_login(username, password)
                if user:
                    st.session_state.logged_in  = True
                    st.session_state.user_info  = user
                    # Lấy tổng token tích lũy thực tế của user từ DB
                    st.session_state.session_tokens = user.get("cost_llm_tokens", 0) or 0
                    st.session_state.page       = "chat"
                    st.success(f"Chào mừng, **{user.get('username','')}**!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        st.markdown("---")
        if st.button("Đăng ký ngay", use_container_width=True):
            st.session_state.page = "register"; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER PAGE
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.page == "register" and not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<h2 style="color:#e8c97a;font-family:\'Playfair Display\',serif;text-align:center;">Đăng ký tài khoản</h2>', unsafe_allow_html=True)

        with st.form("register_form"):
            user_id       = st.text_input("User ID *",          placeholder="VD: sell_user_001")
            username      = st.text_input("Tên đăng nhập *",    placeholder="VD: nguyen_van_a")
            password      = st.text_input("Mật khẩu *",         type="password")
            confirm_pw    = st.text_input("Xác nhận mật khẩu *",type="password")
            group_id      = st.text_input("Group ID *",          value="Techcombank_group")
            company_id    = st.text_input("Company ID",          placeholder="VD: Techcomlife")
            department_id = st.text_input("Department ID",       placeholder="VD: Sell_Techcomlife")
            submitted     = st.form_submit_button("Tạo tài khoản", use_container_width=True)

        if submitted:
            errors = [f for f in ["user_id","username","password","group_id"] if not locals()[f]]
            if password != confirm_pw:
                st.error("Mật khẩu xác nhận không khớp.")
            elif errors:
                st.error(f"Vui lòng điền: {', '.join(errors)}")
            else:
                with st.spinner("Đang tạo tài khoản…"):
                    ok, result = api_register(
                        user_id.strip(), username.strip(), password,
                        group_id.strip(), company_id.strip(), department_id.strip(),
                    )
                if ok:
                    st.success(f"Tài khoản **{username}** đã được tạo!")
                    time.sleep(1); st.session_state.page = "login"; st.rerun()
                else:
                    st.error(f"Lỗi: {result}")

        st.markdown("---")
        if st.button("Quay lại đăng nhập", use_container_width=True):
            st.session_state.page = "login"; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.logged_in:
    st.session_state.page = "chat"
    user = st.session_state.user_info

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
        <div>
            <h2 style="color:#e8c97a;font-family:'Playfair Display',serif;margin:0;">
                Chuyên viên tư vấn bảo hiểm Techcomlife
            </h2>
            <p style="color:#5a7fa0;margin:0;font-size:.82rem;">
                User: <strong style="color:#7aabdd;">{user.get('username','')}</strong>
                &nbsp;|&nbsp; Tổng tokens đã dùng: <strong style="color:#e8c97a;">{st.session_state.session_tokens:,}</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#3a5a7a;">
                <p>Xin chào! Tôi là trợ lý tư vấn bảo hiểm Techcomlife.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role   = msg["role"]
                content = msg["content"]
                tokens  = msg.get("tokens", 0)
                refs    = msg.get("references", [])

                if role == "user":
                    st.markdown(f'<div class="chat-label" style="text-align:right;">Bạn</div>'
                                f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-label">Trợ lý Techcomlife</div>'
                                f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)
                    meta = []
                    if tokens:
                        meta.append(f'<span class="token-badge">{tokens:,} tokens</span>')
                    if refs:
                        ref_html = " ".join(
                            f'<span class="ref-item">{r.get("filename") or r.get("document_id","")}</span>'
                            for r in refs if r.get("filename") or r.get("document_id")
                        )
                        if ref_html:
                            meta.append(ref_html)
                    if meta:
                        st.markdown(f'<div style="margin-top:4px;margin-bottom:8px;">{" ".join(meta)}</div>',
                                    unsafe_allow_html=True)

    # Input
    st.markdown("---")
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_area(
            "Câu hỏi", key="user_input",
            placeholder="Nhập câu hỏi về bảo hiểm Techcomlife…",
            height=80, label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send_btn = st.button("Gửi", use_container_width=True)

    if send_btn and user_input and user_input.strip():
        question = user_input.strip()

        # Thêm user message vào history
        st.session_state.chat_history.append({
            "role": "user", "content": question, "tokens": 0, "references": [],
        })

        # Build history cho API (không gửi message vừa thêm)
        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]

        # Gọi RAG API
        with st.spinner("Đang tìm kiếm và tạo câu trả lời…"):
            answer, refs, tokens = chat_streaming(question, api_history, user.get("user_id", ""))

        # FIX Bug 3a: cộng dồn session_tokens ngay lập tức
        # KHÔNG gọi api_get_user() vì DB có thể chưa flush xong
        if tokens > 0:
            st.session_state.session_tokens += tokens

        # Thêm assistant message
        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "tokens": tokens, "references": refs,
        })

        # FIX Bug 3b: KHÔNG gọi api_get_user() sau mỗi chat để tránh đọc giá trị cũ
        # user_info chỉ được cập nhật lại khi login lần sau hoặc user bấm refresh thủ công

        st.rerun()