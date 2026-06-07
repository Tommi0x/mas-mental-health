import math
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


# Return the diagnosis when at least 75% of agents agree.
def aggregate_superconsent(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")

    normalized_labels, display_labels = _collect_disease_labels(diagnoses)
    votes = Counter(normalized_labels)
    threshold = math.ceil(0.75 * len(diagnoses))
    winners = sorted(
        display_labels[label]
        for label, count in votes.items()
        if count >= threshold
    )

    if len(winners) == 1:
        return winners[0]

    return "No superconsent (75% agreement required)"


# Return the diagnosis with the highest Copeland pairwise-victory score.
def aggregate_copeland(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")

    normalized_labels, display_labels = _collect_disease_labels(diagnoses)
    votes = Counter(normalized_labels)
    candidates = list(votes.keys())

    copeland_scores: dict[str, int] = {}
    for candidate in candidates:
        wins = sum(
            1
            for other in candidates
            if other != candidate and votes[candidate] > votes[other]
        )
        copeland_scores[candidate] = wins

    top_score = max(copeland_scores.values())
    winners = sorted(
        display_labels[label]
        for label, score in copeland_scores.items()
        if score == top_score
    )

    if len(winners) == 1:
        return winners[0]

    return f"Tie between: {', '.join(winners)}"
