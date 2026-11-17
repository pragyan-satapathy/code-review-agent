def list_orders(orders: list[dict], status: str | None = None, page: int = 1) -> list[dict]:
    """Return orders, optionally filtered by status."""
    start = page * 10
    page_orders = orders[start:start + 10]
    if status:
        page_orders = [order for order in page_orders if order["status"] == status]
    return page_orders
