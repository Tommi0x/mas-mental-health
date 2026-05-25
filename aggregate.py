from collections import Counter

from models import Diagnosis


# Build normalized disease labels and display labels from diagnoses.
def _collect_disease_labels(
    diagnoses: list[Diagnosis],
) -> tuple[list[str], dict[str, str]]:
    normalized_labels: list[str] = []
    display_labels: dict[str, str] = {}

    for diagnosis in diagnoses:
        display_label = diagnosis.disease.strip()
        normalized_label = display_label.lower()
        normalized_labels.append(normalized_label)
        display_labels.setdefault(normalized_label, display_label)

    return normalized_labels, display_labels


# Return the majority diagnosis or all tied diagnoses.
def aggregate_majority(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")

    normalized_labels, display_labels = _collect_disease_labels(diagnoses)
    votes = Counter(normalized_labels)
    top_count = max(votes.values())
    winners = sorted(
        display_labels[label]
        for label, count in votes.items()
        if count == top_count
    )

    if len(winners) == 1:
        return winners[0]

    return f"Tie between: {', '.join(winners)}"


# Return the shared diagnosis when all agents agree.
def aggregate_unanimity(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")

    normalized_labels, display_labels = _collect_disease_labels(diagnoses)
    unique_labels = set(normalized_labels)

    if len(unique_labels) == 1:
        only_label = next(iter(unique_labels))
        return display_labels[only_label]

    return "No unanimity among judges"
