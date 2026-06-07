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


# Return the diagnosis when at least the given share of agents agree.
def aggregate_superconsent(
    diagnoses: list[Diagnosis],
    agreement_percent: int = 75,
) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")
    if not 1 <= agreement_percent <= 100:
        raise ValueError("agreement_percent must be between 1 and 100")

    normalized_labels, display_labels = _collect_disease_labels(diagnoses)
    votes = Counter(normalized_labels)
    threshold = math.ceil(agreement_percent / 100 * len(diagnoses))
    winners = sorted(
        display_labels[label]
        for label, count in votes.items()
        if count >= threshold
    )

    if len(winners) == 1:
        return winners[0]

    return f"No superconsent ({agreement_percent}% agreement required)"


# Return the diagnosis with the highest sum of agent weights.
def aggregate_weighted(
    diagnoses: list[Diagnosis],
    agent_weights: dict[str, float],
) -> str:
    if not diagnoses:
        raise ValueError("Cannot aggregate an empty diagnosis list")

    weighted_votes: dict[str, float] = {}
    display_labels: dict[str, str] = {}

    for diagnosis in diagnoses:
        display_label = diagnosis.disease.strip()
        normalized_label = display_label.lower()
        display_labels.setdefault(normalized_label, display_label)
        weight = agent_weights.get(diagnosis.agent_name, 1.0)
        if weight <= 0:
            raise ValueError(f"Weight for agent '{diagnosis.agent_name}' must be positive")
        weighted_votes[normalized_label] = weighted_votes.get(normalized_label, 0.0) + weight

    top_score = max(weighted_votes.values())
    winners = sorted(
        display_labels[label]
        for label, score in weighted_votes.items()
        if score == top_score
    )

    if len(winners) == 1:
        return winners[0]

    return f"Tie between: {', '.join(winners)}"


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
