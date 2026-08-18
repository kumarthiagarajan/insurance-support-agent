import os
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment or .env file before running.")

from langchain_core.messages import HumanMessage

from graph import build_graph
from tracing import langfuse, traced_turn


def main():
    app = build_graph()
    print("Insurance Support Assistant (type 'quit' to exit)\n")
    customer_id = input("Customer ID (try CUST001, CUST002): ").strip() or "CUST001"
    session_id = str(uuid.uuid4())

    state = {
        "messages": [],
        "customer_id": customer_id,
        "next": "",
        "handled": [],
        "iterations": 0,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        prev_len = len(state["messages"])
        state["messages"].append(HumanMessage(content=user_input))
        state["handled"] = []
        state["iterations"] = 0

        with traced_turn(
            customer_id=customer_id, session_id=session_id, feature="cli", user_message=user_input
        ) as (root_span, handler):
            state = app.invoke(state, config={"callbacks": [handler]})
            replies = [
                {"role": "assistant", "name": m.name, "content": m.content}
                for m in state["messages"][prev_len + 1 :]
                if m.type == "ai"
            ]
            root_span.update(output=replies)

        for reply in replies:
            print(f"\n[{reply['name'] or 'assistant'}] {reply['content']}")

    # main.py is a short-lived process -- flush the background queue before exit so
    # the last turn's trace isn't lost (see Langfuse's queuing/flushing docs).
    langfuse.shutdown()


if __name__ == "__main__":
    main()
