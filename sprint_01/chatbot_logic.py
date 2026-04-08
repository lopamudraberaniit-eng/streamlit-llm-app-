# sprint_01/chatbot_logic.py
# Building the Chatbot Backend
# API    : Groq (free)
# Data   : E-Commerce Demo Dataset
# Case   : AI-Powered Customer Support Assistant

import sys
import os

# Add parent directory to path so we can import api_client
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api_client import get_groq_response
from sprint_01.data_loader import load_order_context, get_all_order_ids


def build_system_prompt(order_context: str) -> str:
    """
    Builds the system prompt for the customer support assistant.
    Injects real order context from the Olist dataset.
    This is the 'system' role in the role-based message structure.
    """
    return f"""You are a customer support assistant for an 
e-commerce platform. Your job is to help customers with 
order status, returns, shipping queries, and product information.

Be concise, polite, and helpful. Do not make up information.
If you cannot answer from the context provided, say so clearly.

Here is the relevant order information for this conversation:

{order_context}
"""


def get_support_response(user_message: str, order_context: str) -> str:
    """
    Takes a customer message and order context.
    Builds role-based messages and calls the Groq API.
    Returns the assistant's response.

    This function demonstrates:
    - Structured role-based messages (system, user)
    - LLM API call with correct parameters
    - Real data injected as context
    """

    # Build role-based message structure
    # This is the core concept of LO2
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(order_context)
            # CONCEPT: System role sets persona and injects data context
        },
        {
            "role": "user",
            "content": user_message
            # CONCEPT: User role carries the customer's actual query
        }
    ]

    # Call Groq API using fast model for Sprint 1
    response = get_groq_response(
        messages,
        model_type="fast",   # llama-3.1-8b-instant
        max_tokens=200
    )

    return response


if __name__ == '__main__':

    print("=" * 60)
    print("Chatbot Backend")
    print("Case Study: E-Commerce Customer Support Assistant")
    print("=" * 60)

    # --------------------------------------------------------
    # DEMO STEP 1: Load a real order from the Olist dataset
    # --------------------------------------------------------

    order_ids = get_all_order_ids()

    if not order_ids:
        print("No orders found. Check your data folder.")
        sys.exit(1)

    # Use first order for demo
    # demo_order_id = order_ids[0]
    demo_order_id = input("Enter your Order ID:")
    print(f"\nOrder ID: {demo_order_id}")

    # Load real order context from dataset
    order_context = load_order_context(demo_order_id)
    print(f"\nOrder Context Loaded:\n{order_context}")

    # --------------------------------------------------------
    # DEMO STEP 2: Simulate a customer query about this order
    # --------------------------------------------------------

    customer_query = (
        f"Hi, I placed an order with ID {demo_order_id}. "
        f"Can you tell me the current status and when it will arrive?"
    )

    print(f"\n{'=' * 60}")
    print(f"Customer Query:\n{customer_query}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # DEMO STEP 3: Get the LLM response
    # --------------------------------------------------------

    print("\nCalling Groq API...")
    response = get_support_response(customer_query, order_context)

    print(f"\nAssistant Response:\n{response}")

    # --------------------------------------------------------
    # DEMO STEP 4: Show what happened under the hood
    # --------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("WHAT JUST HAPPENED — Key Concepts:")
    print("=" * 60)
    print("""
1. SYSTEM ROLE   : Set assistant persona + injected real order data
2. USER ROLE     : Carried the customer's query to the model
3. TOKENIZATION  : Both messages were tokenized before processing
4. API CALL      : Groq processed tokens and returned a response
5. REAL DATA     : Response is grounded in actual Olist order data
6. LIMITATION    : If order ID is wrong, model says so — no hallucination
    """)

    # --------------------------------------------------------
    # DEMO STEP 5: Show hallucination guard
    # Demonstrate what happens with an invalid order ID
    # --------------------------------------------------------

    print("=" * 60)
    print("BONUS: Hallucination Guard Demo")
    print("=" * 60)

    fake_order_id    = "FAKE-ORDER-99999"
    fake_context     = load_order_context(fake_order_id)
    fake_query       = f"What is the status of order {fake_order_id}?"
    fake_response    = get_support_response(fake_query, fake_context)

    print(f"\nCustomer Query : {fake_query}")
    print(f"Order Context  : {fake_context}")
    print(f"Assistant      : {fake_response}")
    print("""
OBSERVATION: With a clear system prompt and real context,
the assistant does NOT hallucinate an order status.
It correctly reports that the order was not found.
This demonstrates the importance of grounding LLM responses
in real data — a concept that leads directly into RAG in Sprint 6.
    """)