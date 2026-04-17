from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


PROBLEM_NUMERIC_FEATURES = [
    "age",
    "anxiety_score",
    "depression_score",
    "stress_level",
    "sleep_quality",
    "sleep_hours",
    "symptom_duration_months",
    "gad7_estimate",
    "phq9_estimate",
    "irritability_level",
    "work_or_study_impairment",
    "bmi_estimate",
]

PROBLEM_CATEGORICAL_FEATURES = [
    "gender",
    "social_support",
    "physical_activity",
    "panic_symptoms",
    "concentration_difficulty",
    "appetite_change",
    "prior_treatment",
    "current_medication",
    "trauma_history",
    "substance_use_risk",
    "comorbid_profile",
    "clinical_severity",
]

PROBLEM_TEXT_FEATURES = ["main_issue"]

SOLUTION_FEATURES = [
    "intervention_type",
    "intensity",
    "weekly_frequency",
    "recommendation_text",
]


@dataclass(frozen=True)
class NumericFeatureStats:
    """Resumo estatistico de um atributo numerico da base de casos."""

    minimum: float
    maximum: float

    @property
    def value_range(self) -> float:
        """Amplitude numerica segura (nunca menor que 1.0)."""
        return max(self.maximum - self.minimum, 1.0)


def _to_float_or_default(value: Any, default: float = 0.0) -> float:
    """Converte valor para float; em falha retorna valor padrao."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_text(value: Any, default: str = "") -> str:
    """Converte valor para texto limpo; usa padrao quando vazio/invalido."""
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _normalize_problem(raw_problem: dict[str, Any]) -> dict[str, Any]:
    """Normaliza atributos de problema por tipo (numerico/categorico/texto)."""
    normalized: dict[str, Any] = {}

    for feature in PROBLEM_NUMERIC_FEATURES:
        normalized[feature] = _to_float_or_default(raw_problem.get(feature))

    for feature in PROBLEM_CATEGORICAL_FEATURES:
        normalized[feature] = _to_text(raw_problem.get(feature), default="unknown").lower()

    for feature in PROBLEM_TEXT_FEATURES:
        normalized[feature] = _to_text(raw_problem.get(feature), default="")

    return normalized


def _normalize_solution(raw_solution: dict[str, Any]) -> dict[str, Any]:
    """Normaliza atributos da solucao associada ao caso."""
    return {
        "intervention_type": _to_text(raw_solution.get("intervention_type"), "unknown").lower(),
        "intensity": _to_float_or_default(raw_solution.get("intensity")),
        "weekly_frequency": _to_float_or_default(raw_solution.get("weekly_frequency")),
        "recommendation_text": _to_text(raw_solution.get("recommendation_text")),
    }


def load_casebase(path: str) -> tuple[dict[str, dict[str, Any]], dict[str, NumericFeatureStats]]:
    """Carrega o CSV e devolve base de casos + estatisticas numericas globais.

    Estrutura de cada caso:
    {
      "problem": {...features do problema...},
      "solution": {...features da solução...}
    }
    """
    df = pd.read_csv(path)

    stats: dict[str, NumericFeatureStats] = {}
    for feature in PROBLEM_NUMERIC_FEATURES:
        col = pd.to_numeric(df[feature], errors="coerce")
        minimum = float(col.min()) if not pd.isna(col.min()) else 0.0
        maximum = float(col.max()) if not pd.isna(col.max()) else 1.0
        stats[feature] = NumericFeatureStats(minimum=minimum, maximum=maximum)

    casebase: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        case_id = _to_text(row_dict.get("case_id"), default=f"case_{len(casebase) + 1}")

        raw_problem = {feature: row_dict.get(feature) for feature in (PROBLEM_NUMERIC_FEATURES + PROBLEM_CATEGORICAL_FEATURES + PROBLEM_TEXT_FEATURES)}
        raw_solution = {feature: row_dict.get(feature) for feature in SOLUTION_FEATURES}

        casebase[case_id] = {
            "problem": _normalize_problem(raw_problem),
            "solution": _normalize_solution(raw_solution),
        }

    return casebase, stats


def build_problem_only_casebase(casebase: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Extrai apenas a parte de problema de cada caso para a recuperacao."""
    return {case_id: case["problem"] for case_id, case in casebase.items()}


def normalize_query(raw_query: dict[str, Any]) -> dict[str, Any]:
    """Normaliza um novo caso de consulta no mesmo padrao da base."""
    return _normalize_problem(raw_query)


def compute_numeric_stats_from_casebase(
    casebase: dict[str, dict[str, Any]],
) -> dict[str, NumericFeatureStats]:
    """Recalcula min/max numericos para um subconjunto de treino.

    Essa funcao e usada em validacao (LOO/K-Fold), onde o conjunto de
    treino muda a cada iteracao e as faixas numericas precisam refletir
    apenas os casos disponiveis naquele momento.
    """
    stats: dict[str, NumericFeatureStats] = {}

    for feature in PROBLEM_NUMERIC_FEATURES:
        values = [
            float(case["problem"].get(feature, 0.0))
            for case in casebase.values()
            if case and "problem" in case
        ]
        if not values:
            stats[feature] = NumericFeatureStats(minimum=0.0, maximum=1.0)
            continue

        stats[feature] = NumericFeatureStats(minimum=min(values), maximum=max(values))

    return stats