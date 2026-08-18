import os
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment or .env file before running.")

import gradio as gr
from langchain_core.messages import HumanMessage

from graph import build_graph
from tracing import traced_turn

_graph = build_graph()


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
        str(uuid.uuid4()),
        [],
        gr.update(visible=False),
        gr.update(visible=True),
        f"Customer: **{customer_id}**",
    )


def switch_customer():
    return (
        None,
        None,
        [],
        gr.update(visible=True),
        gr.update(visible=False),
        "",
    )


def respond(message, history, state, session_id):
    message = (message or "").strip()
    if not message or state is None:
        return history, state, ""

    prev_len = len(state["messages"])
    state["messages"].append(HumanMessage(content=message))
    state["handled"] = []
    state["iterations"] = 0

    with traced_turn(
        customer_id=state["customer_id"],
        session_id=session_id,
        feature="gradio",
        user_message=message,
    ) as (root_span, handler):
        state = _graph.invoke(state, config={"callbacks": [handler]})
        new_messages = [m for m in state["messages"][prev_len + 1 :] if m.type == "ai"]
        root_span.update(
            output=[
                {"role": "assistant", "name": m.name, "content": m.content}
                for m in new_messages
            ]
        )

    history = history + [{"role": "user", "content": message}]
    for m in new_messages:
        speaker = getattr(m, "name", None) or "assistant"
        history.append({"role": "assistant", "content": f"**{speaker}**\n\n{m.content}"})

    return history, state, ""


with gr.Blocks(title="Insurance Support Assistant") as demo:
    gr.Markdown("## 🛡️ Insurance Support Assistant")
    state = gr.State(None)
    session_id_state = gr.State(None)

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

    session_outputs = [state, session_id_state, chatbot, start_col, chat_col, customer_label]
    start_btn.click(start_session, inputs=[customer_input], outputs=session_outputs)
    customer_input.submit(start_session, inputs=[customer_input], outputs=session_outputs)
    switch_btn.click(switch_customer, outputs=session_outputs)

    message_outputs = [chatbot, state, msg_input]
    respond_inputs = [msg_input, chatbot, state, session_id_state]
    send_btn.click(respond, inputs=respond_inputs, outputs=message_outputs)
    msg_input.submit(respond, inputs=respond_inputs, outputs=message_outputs)


if __name__ == "__main__":
    demo.launch()
