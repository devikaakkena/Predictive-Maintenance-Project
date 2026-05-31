def format_prediction(pred_value: int) -> str:
    """Formats raw binary prediction values into user-friendly text with emojis."""
    return "✅ Safe" if pred_value == 0 else "⚠️ Failure"
