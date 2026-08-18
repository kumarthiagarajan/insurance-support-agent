import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Set ANTHROPIC_API_KEY in your environment or .env file before running.")
    st.stop()

from langchain_core.messages import HumanMessage

from graph import build_graph
from tracing import traced_turn

st.set_page_config(page_title="Insurance Support Assistant", page_icon="🛡️")


@st.cache_resource
def get_app():
    return build_graph()


def new_state(customer_id: str) -> dict:
    return {
        "messages": [],
        "customer_id": customer_id,
        "next": "",
        "handled": [],
        "iterations": 0,
    }


def render_ai_message(message):
    speaker = getattr(message, "name", None) or "assistant"
    with st.chat_message("assistant"):
        st.markdown(f"**{speaker}**")
        st.markdown(message.content)


app = get_app()

st.title("🛡️ Insurance Support Assistant")

if "customer_id" not in st.session_state:
    st.session_state.customer_id = None

if st.session_state.customer_id is None:
    st.write("Enter a customer ID to start a conversation (try `CUST001` or `CUST002`).")
    with st.form("customer_form"):
        customer_id = st.text_input("Customer ID", value="CUST001")
        submitted = st.form_submit_button("Start")
    if submitted and customer_id.strip():
        st.session_state.customer_id = customer_id.strip()
        st.session_state.state = new_state(st.session_state.customer_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.stop()

with st.sidebar:
    st.caption(f"Customer: **{st.session_state.customer_id}**")
    if st.button("Switch customer"):
        st.session_state.customer_id = None
        st.session_state.pop("state", None)
        st.session_state.pop("session_id", None)
        st.rerun()

for message in st.session_state.state["messages"]:
    if message.type == "human":
        with st.chat_message("user"):
            st.markdown(message.content)
    elif message.type == "ai":
        render_ai_message(message)

user_input = st.chat_input("Type your message...")
if user_input:
    state = st.session_state.state
    prev_len = len(state["messages"])
    state["messages"].append(HumanMessage(content=user_input))
    state["handled"] = []
    state["iterations"] = 0

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        with traced_turn(
            customer_id=st.session_state.customer_id,
            session_id=st.session_state.session_id,
            feature="streamlit",
            user_message=user_input,
        ) as (root_span, handler):
            state = app.invoke(state, config={"callbacks": [handler]})
            new_messages = [m for m in state["messages"][prev_len + 1 :] if m.type == "ai"]
            root_span.update(
                output=[
                    {"role": "assistant", "name": m.name, "content": m.content}
                    for m in new_messages
                ]
            )
    st.session_state.state = state

    for message in new_messages:
        render_ai_message(message)
