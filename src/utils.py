from src.constants import CLINICAL_THRESHOLDS

def risk_from_score(label: str, score: float) -> str:
    """
    Determine the risk tier based on the label and score using clinical thresholds.

    Args:
        label (str): The label for which to determine the risk tier.
        score (float): The score associated with the label.

    Returns:
        str: The risk tier ('green', 'yellow', 'orange', 'red') based on the clinical thresholds.
    """
    thresholds = CLINICAL_THRESHOLDS.get(label)
    if thresholds is None:
        raise ValueError(f"No clinical thresholds defined for label '{label}'.")

    if score > thresholds['red']:
        return 'red'
    elif score > thresholds['orange']:
        return 'orange'
    elif score > thresholds['yellow']:
        return 'yellow'
    else:
        return 'green'