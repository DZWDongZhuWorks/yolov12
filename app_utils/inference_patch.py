
def get_model_names(model_id: str) -> Dict[int, str]:
    """
    Load the model (lightweight if possible) and return its class names.
    """
    try:
        model = YOLO(model_id)
        return model.names
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return {}
