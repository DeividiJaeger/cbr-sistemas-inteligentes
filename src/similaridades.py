from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

import cbrkit

from src.loader import (
    PROBLEM_CATEGORICAL_FEATURES,
    PROBLEM_NUMERIC_FEATURES,
    PROBLEM_TEXT_FEATURES,
    NumericFeatureStats,
)


SimilarityFunc = Callable[[Any, Any], float]


def text_jaccard_similarity(x: str, y: str) -> float:
    """Calcula similaridade textual via Jaccard entre conjuntos de tokens.

    O texto e tokenizado com regex simples, convertido para minusculas
    e comparado por intersecao/uniao. Retorna valor no intervalo [0, 1].
    """
    tokens_x = set(re.findall(r"\w+", (x or "").lower()))
    tokens_y = set(re.findall(r"\w+", (y or "").lower()))

    if not tokens_x and not tokens_y:
        return 1.0
    if not tokens_x or not tokens_y:
        return 0.0

    inter = len(tokens_x.intersection(tokens_y))
    union = len(tokens_x.union(tokens_y))
    return inter / union if union else 0.0


def _build_local_similarity_map(numeric_stats: dict[str, NumericFeatureStats]) -> dict[str, SimilarityFunc]:
    """Monta similaridade local por atributo de acordo com o tipo.

    Regras aplicadas:
    - numericos: similaridade linear normalizada por faixa min/max
    - categoricos: igualdade exata
    - textuais: Jaccard de tokens
    """
    similarities: dict[str, SimilarityFunc] = {}

    for feature in PROBLEM_NUMERIC_FEATURES:
        stats = numeric_stats[feature]
        min_value = float(stats.minimum)
        max_value = float(stats.maximum)
        if max_value <= min_value:
            max_value = min_value + 1.0

        similarities[feature] = cbrkit.sim.numbers.linear(max=max_value, min=min_value)

    for feature in PROBLEM_CATEGORICAL_FEATURES:
        similarities[feature] = cbrkit.sim.generic.equality()

    for feature in PROBLEM_TEXT_FEATURES:
        similarities[feature] = text_jaccard_similarity

    return similarities


def build_similarity_model(numeric_stats: dict[str, NumericFeatureStats]):
    """Cria modelo global de similaridade atributo-valor.

    As similaridades locais sao agregadas pela media aritmetica, gerando
    um score global unico para cada comparacao entre casos.
    """
    local_similarities = _build_local_similarity_map(numeric_stats)
    return cbrkit.sim.attribute_value(
        attributes=local_similarities,
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )


def weighted_mode(values: list[str], weights: list[float]) -> str:
    """Retorna o valor categorico com maior peso acumulado.

    E utilizado para combinar categorias de varios vizinhos, favorecendo
    casos com maior similaridade.
    """
    counter: Counter[str] = Counter()
    for value, weight in zip(values, weights):
        counter[value] += weight

    return counter.most_common(1)[0][0] if counter else "combined"