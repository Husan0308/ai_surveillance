"""Canonical local API/ML ports and URLs for the ai_surveillance services."""

API_PORT_DEFAULT = 8100
ML_PORT_DEFAULT = 8101

API_BASE_URL_DEFAULT = f"http://127.0.0.1:{API_PORT_DEFAULT}"
ML_BASE_URL_DEFAULT = f"http://127.0.0.1:{ML_PORT_DEFAULT}"
MONITORING_WS_URL_DEFAULT = f"ws://127.0.0.1:{API_PORT_DEFAULT}/ws/v1/monitoring"
