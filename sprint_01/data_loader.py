# sprint_01/data_loader.py
# Loads the dataset and formats it as LLM-readable context
# Used across Sprint 1 demos and practice tasks

import pandas as pd
import os

# ============================================================
# File paths — adjust if your data folder is in a different location
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

ORDERS_FILE   = os.path.join(DATA_PATH, 'AI405_S1_Orders_Data_Concept.csv')
PRODUCTS_FILE = os.path.join(DATA_PATH, 'AI405_S1_Products_Data_Concept.csv')
ITEMS_FILE    = os.path.join(DATA_PATH, 'AI405_S1_OrderItems_Data_Concept.csv')


def load_order_context(order_id: str) -> str:
    """
    Looks up an order by ID and returns a formatted string
    that can be injected into the LLM system prompt as context.
    Returns a 'not found' message if the order ID does not exist.
    """
    try:
        orders_df  = pd.read_csv(ORDERS_FILE)
        items_df   = pd.read_csv(ITEMS_FILE)
        products_df = pd.read_csv(PRODUCTS_FILE)

        # Find the order
        order = orders_df[orders_df['order_id'] == order_id]

        if order.empty:
            return f"Order ID {order_id} not found in the system."

        # Extract order details
        order_row     = order.iloc[0]
        order_status  = order_row['order_status']
        purchase_date = order_row['order_purchase_timestamp']
        delivery_date = order_row.get(
            'order_delivered_customer_date', 'Not yet delivered'
        )

        # Find items in this order
        items = items_df[items_df['order_id'] == order_id]

        # Join with products to get category names
        items_with_products = items.merge(
            products_df,
            on='product_id',
            how='left'
        )

        # Format product list
        product_lines = []
        for _, item in items_with_products.iterrows():
            category = item.get(
                'product_category_name', 'Unknown category'
            )
            price = item.get('price', 'N/A')
            product_lines.append(f"  - Category: {category}, Price: ${price}")

        products_text = "\n".join(product_lines) if product_lines \
            else "  - No product details found"

        # Format full context string
        context = f"""
ORDER DETAILS:
- Order ID     : {order_id}
- Status       : {order_status}
- Purchase Date: {purchase_date}
- Delivery Date: {delivery_date}

ITEMS IN ORDER:
{products_text}
        """.strip()

        return context

    except FileNotFoundError as e:
        return f"Data file not found: {e}. Please check your data folder."
    except Exception as e:
        return f"Error loading order context: {e}"


def get_all_order_ids() -> list:
    """
    Returns a list of all order IDs in the demo dataset.
    Useful for testing and demo purposes.
    """
    try:
        orders_df = pd.read_csv(ORDERS_FILE)
        return orders_df['order_id'].tolist()
    except Exception as e:
        print(f"Error loading order IDs: {e}")
        return []


if __name__ == '__main__':
    # Quick test — print first 3 order IDs and load context for first one
    order_ids = get_all_order_ids()

    if order_ids:
        print(f"Total orders in demo dataset: {len(order_ids)}")
        print(f"\nFirst 3 order IDs:")
        for oid in order_ids[:3]:
            print(f"  {oid}")

        print(f"\nLoading context for first order: {order_ids[0]}")
        print("\n" + load_order_context(order_ids[0]))
    else:
        print("No orders found. Check your data folder.")