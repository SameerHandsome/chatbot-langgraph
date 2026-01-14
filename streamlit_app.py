"""
Streamlit Frontend for Basic Agent with True Streaming
Uses backend's database directly - no separate session storage
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime
import sqlite3

BACKEND_URL = "http://localhost:8001"
BACKEND_DB = "chat_history.db"  

st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


def get_all_sessions_from_db():
    """Get all unique sessions from backend database"""
    try:
        conn = sqlite3.connect(BACKEND_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                session_id, 
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                COUNT(*) as message_count
            FROM chat_history 
            GROUP BY session_id 
            ORDER BY last_message DESC
        """)
        sessions = cursor.fetchall()
        conn.close()
        return sessions
    except Exception as e:
        st.error(f"Database error: {e}")
        return []


def get_session_preview(session_id: str):
    """Get first user message as session preview"""
    try:
        conn = sqlite3.connect(BACKEND_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content 
            FROM chat_history 
            WHERE session_id = ? AND role = 'user' 
            ORDER BY id ASC 
            LIMIT 1
        """, (session_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            preview = result[0]
            return preview[:50] + "..." if len(preview) > 50 else preview
        return "Empty chat"
    except Exception:
        return "Unknown"


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_chat_history(session_id: str):
    """Fetch chat history from backend"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/history/{session_id}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("messages", [])
        return []
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return []


def clear_backend_history(session_id: str):
    """Clear chat history from backend"""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/history/{session_id}",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def load_session(session_id: str):
    """Load a specific session"""
    st.session_state.session_id = session_id
    st.session_state.messages = []
    
    history = get_chat_history(session_id)
    if history:
        for msg in history:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    st.rerun()


def create_new_session():
    """Create a new session"""
    new_session_id = str(uuid.uuid4())
    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    st.rerun()


def send_message_streaming(message: str):
    """Send message and stream response TOKEN BY TOKEN"""
    try:
        url = f"{BACKEND_URL}/chat/stream"
        payload = {
            "message": message,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code}"
        
        tool_placeholder = st.empty()
        message_placeholder = st.empty()
        
        full_response = ""
        tool_calls = []
        current_tools_shown = False
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        
                        if data.get("type") == "tool_call":
                            tools = data.get("tools", [])
                            tool_calls.extend(tools)
                            tool_placeholder.markdown(f"🔧 **Using tools:** {', '.join(tools)}")
                            current_tools_shown = True
                        
                        elif data.get("type") == "token":
                            full_response = data.get("content", "")
                            message_placeholder.markdown(full_response + "▌")  
                        
                        elif data.get("type") == "content":
                            full_response = data.get("content", "")
                            message_placeholder.markdown(full_response + "▌")
                        
                        elif data.get("type") == "tool_result":
                            if current_tools_shown and tool_calls:
                                tool_placeholder.markdown(f"✅ **Used tools:** {', '.join(set(tool_calls))}")
                        
                        elif data.get("type") == "done":
                            if full_response:
                                message_placeholder.markdown(full_response)
                            if current_tools_shown:
                                tool_placeholder.empty()
                            break
                        
                        elif data.get("type") == "error":
                            st.error(f"Error: {data.get('error')}")
                            return None
                    
                    except json.JSONDecodeError:
                        continue
        
        return full_response
    
    except Exception as e:
        st.error(f"Failed to send message: {e}")
        return None


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("🤖 AI Agent")
    
    is_healthy = check_backend_health()
    
    if not is_healthy:
        st.error("❌ Backend not available")
        st.info(f"Start backend: `python main.py`")
    
    st.divider()
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_session()
    
    st.divider()
    
    # Session History from backend database
    st.subheader("💬 Chat History")
    
    sessions = get_all_sessions_from_db()
    
    if sessions:
        for session_id, first_msg, last_msg, msg_count in sessions:
            is_current = session_id == st.session_state.session_id
            
            preview = get_session_preview(session_id)
            
            try:
                last_time = datetime.fromisoformat(last_msg)
                time_str = last_time.strftime("%b %d, %I:%M %p")
            except:
                time_str = "Recent"
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                button_label = f"{'📍' if is_current else '💬'} {preview}"
                if st.button(
                    button_label,
                    key=f"load_{session_id}",
                    use_container_width=True,
                    disabled=is_current,
                    help=f"{time_str} • {msg_count} messages"
                ):
                    load_session(session_id)
            
            with col2:
                if len(sessions) > 1 or not is_current:
                    if st.button("🗑️", key=f"delete_{session_id}", help="Delete"):
                        if clear_backend_history(session_id):
                            if is_current:
                                create_new_session()
                            else:
                                st.rerun()
    else:
        st.info("No chat history yet")
    
    st.divider()
    
    with st.expander("ℹ️ Current Session"):
        st.text(f"ID: {st.session_state.session_id[:16]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            if clear_backend_history(st.session_state.session_id):
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()
    
    st.divider()
    
    with st.expander("🔧 Features"):
        st.markdown("""
        - 💬 **Streaming responses**
        - 🧠 **Conversation memory**
        - 🔍 **Web search**
        - 🌤️ **Weather info**
        - 💱 **Currency conversion**
        - 📚 **Wikipedia lookup**
        - ⏰ **World time**
        """)


# Main chat interface
st.title("💬 Chat with AI Agent")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here...", disabled=not is_healthy):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = send_message_streaming(prompt)
        
        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

st.divider()
current_preview = get_session_preview(st.session_state.session_id)
st.caption(f"💬 {current_preview} | 🔗 Backend: {BACKEND_URL}")