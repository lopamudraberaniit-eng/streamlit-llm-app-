# sprint_01/app.py
# Streamlit Chat UI
# AI-Powered Customer Support Assistant
# Built using Codeium scaffold + human review and integration

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sprint_01.chatbot_logic import get_support_response
from sprint_01.data_loader import (
    load_order_context,
    get_all_order_ids
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Support Assistant",
    page_icon="🛒",
    layout="centered"
)

# ============================================================
# HEADER
# ============================================================

st.title("🛒 E-Commerce Customer Support Assistant")
st.caption(
    "AI-Powered Customer Support Assistant for E-Commerce Dataset"
)
st.divider()

# ============================================================
# SIDEBAR — Order ID selector
# Uses real order IDs from the Olist demo dataset
# ============================================================

with st.sidebar:
    st.header("📦 Order Lookup")
    st.caption("Select or enter an order ID to load context")

    # Load all available order IDs from demo dataset
    available_order_ids = get_all_order_ids()

    if available_order_ids:
        # Dropdown for demo convenience
        selected_order_id = st.selectbox(
            "Select a demo order ID:",
            options=[""] + available_order_ids[:20],
            # Show first 20 for demo
            help="These are real order IDs from the dataset"
        )

        # Manual entry option
        manual_order_id = st.text_input(
            "Or enter an order ID manually:",
            placeholder="e.g. e481f51cbdc54678b7cc49136f2d6af7"
        )

        # Determine active order ID
        active_order_id = manual_order_id.strip() \
            if manual_order_id.strip() \
            else selected_order_id

        if active_order_id:
            # Load and display order context
            order_context = load_order_context(active_order_id)
            st.subheader("📋 Order Context")
            st.code(order_context, language="text")
        else:
            order_context = "No order selected. "\
                "Please select or enter an order ID."
            st.info("Select an order ID to load context.")

    else:
        st.error(
            "No order data found. "
            "Check your data folder."
        )
        order_context = "Order data unavailable."
        active_order_id = None

    st.divider()
    st.caption("🔑 Powered by Groq API (free tier)")
    st.caption("📊 Data: E-Commerce Dataset")

# ============================================================
# CHAT INTERFACE
# Initialize session state for message history
# This demonstrates conversation memory at a basic level
# Full memory implementation comes in Sprint 4
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "order_context" not in st.session_state:
    st.session_state.order_context = ""

# Reset chat if order context changes
if st.session_state.order_context != order_context:
    st.session_state.messages = []
    st.session_state.order_context = order_context

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

# Welcome message on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hello! I am your e-commerce support assistant. "
            "Please select an order ID from the sidebar, "
            "then ask me anything about your order."
        )

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about your order — e.g. What is my order status?"
)

if user_input:

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to message history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # --------------------------------------------------------
    # Get LLM response using backend from LO2
    # --------------------------------------------------------

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Call chatbot logic with user message
            # and real order context from Olist dataset
            response = get_support_response(
                user_message=user_input,
                order_context=order_context
            )

            st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ============================================================
# FOOTER — Teaching notes visible during demo
# ============================================================

st.divider()
with st.expander("🔍 What is happening under the hood?"):
    st.markdown("""
**LO1 — LLM Concepts in action:**
- User input is tokenized before being sent to the API
- Role-based messages structure the conversation
- Knowledge cutoff means the model cannot know real-time stock

**LO2 — API Call Structure:**
- System role: sets assistant persona + injects Olist order data
- User role: carries the customer query
- Groq API returns a grounded response based on real order context

**LO3 — This UI:**
- Generated with Codeium scaffold
- Reviewed, corrected, and integrated manually
- Session state manages basic conversation history
- Deployed on Streamlit Community Cloud
    """)