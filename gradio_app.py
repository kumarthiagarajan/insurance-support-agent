import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment or .env file before running.")

import gradio as gr
from langchain_core.messages import HumanMessage

from graph import build_graph

_graph = build_graph()
NO_REPLY_FALLBACK = (
    "No response was returned for this step. Please try rephrasing your question, "
    "or ask to speak with a representative."
)


def new_state(customer_id: str) -> dict:
    return {
        "messages": [],
        "customer_id": customer_id,
        "next": "",
        "handled": [],
        "iterations": 0,
    }


def start_session(customer_id):
    customer_id = (customer_id or "").strip()
    if not customer_id:
        raise gr.Error("Customer ID is required.")
    return (
        new_state(customer_id),
        [],
        gr.update(visible=False),
        gr.update(visible=True),
        f"Customer: **{customer_id}**",
    )


def switch_customer():
    return (
        None,
        [],
        gr.update(visible=True),
        gr.update(visible=False),
        "",
    )


def respond(message, history, state):
    message = (message or "").strip()
    if not message or state is None:
        return history, state, ""

    prev_len = len(state["messages"])
    state["messages"].append(HumanMessage(content=message))
    state["handled"] = []
    state["iterations"] = 0
    state = _graph.invoke(state)

    history = history + [{"role": "user", "content": message}]
    for m in state["messages"][prev_len + 1 :]:
        if m.type == "ai":
            speaker = getattr(m, "name", None) or "assistant"
            content = m.content or NO_REPLY_FALLBACK
            history.append({"role": "assistant", "content": f"**{speaker}**\n\n{content}"})

    return history, state, ""


with gr.Blocks(title="Insurance Support Assistant") as demo:
    gr.Markdown("## 🛡️ Insurance Support Assistant")
    state = gr.State(None)

    with gr.Column(visible=True) as start_col:
        gr.Markdown("Enter a customer ID to start a conversation (try `CUST001` or `CUST002`).")
        customer_input = gr.Textbox(label="Customer ID", value="CUST001")
        start_btn = gr.Button("Start", variant="primary")

    with gr.Column(visible=False) as chat_col:
        customer_label = gr.Markdown()
        chatbot = gr.Chatbot(height=480)
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message...", scale=8, show_label=False
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
        switch_btn = gr.Button("Switch customer")

    session_outputs = [state, chatbot, start_col, chat_col, customer_label]
    start_btn.click(start_session, inputs=[customer_input], outputs=session_outputs)
    customer_input.submit(start_session, inputs=[customer_input], outputs=session_outputs)
    switch_btn.click(switch_customer, outputs=session_outputs)

    message_outputs = [chatbot, state, msg_input]
    send_btn.click(respond, inputs=[msg_input, chatbot, state], outputs=message_outputs)
    msg_input.submit(respond, inputs=[msg_input, chatbot, state], outputs=message_outputs)


if __name__ == "__main__":
    demo.launch()
