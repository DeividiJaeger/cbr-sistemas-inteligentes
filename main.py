from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import cbrkit

from src.loader import (
	build_problem_only_casebase,
	compute_numeric_stats_from_casebase,
	load_casebase,
	normalize_query,
)
from src.similaridades import build_similarity_model, text_jaccard_similarity, weighted_mode


DEFAULT_DATASET_PATH = "data/cbr_psychology_110_cases_clinical.csv"


def _retrieve_neighbors(
	training_casebase: dict[str, dict[str, Any]],
	query: dict[str, Any],
	k: int,
) -> tuple[list[str], dict[str, float]]:
	"""Recupera os k casos mais similares para uma consulta.

	A funcao monta uma base contendo apenas os atributos de problema,
	recalcula as estatisticas numericas no subconjunto de treino, cria
	o modelo de similaridade e executa a recuperacao com CBRkit.

	Retorna:
	- lista ordenada de IDs de casos (mais similar para menos similar)
	- mapa ID -> score numerico de similaridade
	"""
	if not training_casebase:
		return [], {}

	problem_casebase = build_problem_only_casebase(training_casebase)
	numeric_stats = compute_numeric_stats_from_casebase(training_casebase)
	similarity_model = build_similarity_model(numeric_stats)
	retriever = cbrkit.retrieval.dropout(
		cbrkit.retrieval.build(similarity_model),
		limit=max(1, min(k, len(problem_casebase))),
	)

	result = cbrkit.retrieval.apply_query(problem_casebase, query, retriever)
	top_case_ids = list(result.ranking)
	similarity_scores = {
		case_id: float(getattr(sim, "value", sim))
		for case_id, sim in result.similarities.items()
	}
	return top_case_ids, similarity_scores


def _severity_bonus(clinical_severity: str) -> float:
	"""Converte severidade clinica em ajuste de intensidade.

	Heuristica simples:
	- high: aumenta mais a intensidade
	- moderate: aumento intermediario
	- demais valores: pequeno decramento
	"""
	sev = (clinical_severity or "").lower()
	if sev == "high":
		return 1.0
	if sev == "moderate":
		return 0.4
	return -0.2


def _risk_bonus(query: dict[str, Any]) -> float:
	"""Calcula bonus adicional de intensidade a partir de sinais de risco.

	Soma incrementos quando escores de ansiedade, depressao, estresse e
	prejuizo funcional estao altos.
	"""
	bonus = 0.0
	if query["anxiety_score"] >= 8:
		bonus += 0.4
	if query["depression_score"] >= 8:
		bonus += 0.4
	if query["stress_level"] >= 8:
		bonus += 0.2
	if query["work_or_study_impairment"] >= 8:
		bonus += 0.2
	return bonus


def _collect_neighbor_guidelines(
	top_solutions: list[dict[str, Any]],
	normalized_weights: list[float],
	max_items: int = 2,
) -> list[str]:
	"""Extrai orientacoes curtas dos textos de casos similares.

	Seleciona frases com melhor peso dos vizinhos e remove repeticoes,
	evitando frases genericas ja montadas pela adaptacao.
	"""
	candidates: list[tuple[float, str]] = []
	seen_norm: set[str] = set()

	for solution, weight in zip(top_solutions, normalized_weights):
		text = str(solution.get("recommendation_text", "") or "")
		sentences = re.split(r"[.!?]+", text)
		for sentence in sentences:
			fragment = " ".join(sentence.strip().split())
			if len(fragment) < 28:
				continue

			norm = fragment.lower()
			if norm in seen_norm:
				continue
			if any(
				blocked in norm
				for blocked in [
					"intensidade sugerida",
					"frequencia semanal",
					"baseado em",
				]
			):
				continue

			seen_norm.add(norm)
			length_penalty = 0.015 * abs(len(fragment) - 95)
			score = max(weight - length_penalty, 0.0)
			candidates.append((score, fragment))

	candidates.sort(key=lambda item: item[0], reverse=True)
	return [text for _, text in candidates[:max_items]]


def adapt_solution(
	query: dict[str, Any],
	retrieved_case_ids: list[str],
	full_casebase: dict[str, dict[str, Any]],
	similarities: dict[str, float],
) -> dict[str, Any]:
	"""Gera solucao adaptada com base nos casos recuperados.

	Estrategia utilizada:
	- combina ate 3 solucoes de vizinhos com media ponderada pela similaridade
	- ajusta intensidade com regras de severidade e risco
	- ajusta frequencia semanal com duracao dos sintomas e qualidade do sono
	- define tipo de intervencao por voto ponderado
	- monta texto final com recomendacoes adicionais guiadas por heuristicas
	"""
	if not retrieved_case_ids:
		return {
			"intervention_type": "combined",
			"intensity": 3.0,
			"weekly_frequency": 1.0,
			"recommendation_text": "Nenhum caso similar suficiente foi encontrado para adaptação.",
		}

	top_ids = retrieved_case_ids[: min(5, len(retrieved_case_ids))]
	weights = [max(similarities.get(case_id, 0.0), 1e-6) for case_id in top_ids]
	weight_sum = sum(weights)
	normalized_weights = [w / weight_sum for w in weights]

	top_solutions = [full_casebase[case_id]["solution"] for case_id in top_ids]

	base_intensity = sum(sol["intensity"] * w for sol, w in zip(top_solutions, normalized_weights))
	base_frequency = sum(sol["weekly_frequency"] * w for sol, w in zip(top_solutions, normalized_weights))

	adapted_intensity = base_intensity + _severity_bonus(query["clinical_severity"]) + _risk_bonus(query)
	adapted_intensity = float(max(1.0, min(5.0, round(adapted_intensity, 1))))

	frequency_adjust = 0.0
	if query["symptom_duration_months"] >= 12:
		frequency_adjust += 0.5
	if query["sleep_quality"] <= 4:
		frequency_adjust += 0.5

	adapted_frequency = float(max(1.0, min(7.0, round(base_frequency + frequency_adjust, 1))))

	intervention_type = weighted_mode(
		values=[sol["intervention_type"] for sol in top_solutions],
		weights=normalized_weights,
	)
	if query["clinical_severity"] == "high" and intervention_type != "combined":
		intervention_type = "combined"

	risk_load = _risk_bonus(query)
	if query["clinical_severity"] == "high" or risk_load >= 0.8:
		priority_line = "Priorizar estabilizacao inicial e plano ativo de acompanhamento clinico."
	elif query["clinical_severity"] == "moderate":
		priority_line = "Focar em reducao progressiva de sintomas com metas semanais objetivas."
	else:
		priority_line = "Conduzir intervencao gradual com reforco de autonomia e prevencao de recaidas."

	neighbor_guidelines = _collect_neighbor_guidelines(top_solutions, normalized_weights, max_items=2)
	case_word = "caso" if len(top_ids) == 1 else "casos"

	recommendation_fragments = [
		f"Baseado em {len(top_ids)} {case_word} similares, recomenda-se uma abordagem {intervention_type}.",
		priority_line,
		f"Intensidade sugerida: {adapted_intensity} (escala 1-5).",
		f"Frequencia semanal sugerida: {adapted_frequency} sessoes.",
	]

	if query["sleep_quality"] <= 4 or query["sleep_hours"] < 6:
		recommendation_fragments.append("Incluir monitoramento de sono e rotina de higiene do sono.")
	if query["social_support"] == "low":
		recommendation_fragments.append("Adicionar plano de fortalecimento de suporte social e rede de apoio.")
	if query["prior_treatment"] == "none" and query["clinical_severity"] in {"moderate", "high"}:
		recommendation_fragments.append("Sugerir inicio de psicoterapia estruturada com metas de curto prazo.")
	if query["substance_use_risk"] in {"moderate", "high"}:
		recommendation_fragments.append("Incluir triagem e psicoeducacao sobre risco de uso de substancias.")

	for guideline in neighbor_guidelines:
		recommendation_fragments.append(f"Apoio observado em casos similares: {guideline}.")

	recommendation_fragments.append(f"Queixa principal observada: {query['main_issue']}")
	recommendation_text = " ".join(recommendation_fragments)

	return {
		"intervention_type": intervention_type,
		"intensity": adapted_intensity,
		"weekly_frequency": adapted_frequency,
		"recommendation_text": recommendation_text,
	}


def _build_demo_query(full_casebase: dict[str, dict[str, Any]]) -> dict[str, Any]:
	"""Cria um caso de consulta de demonstracao para execucoes rapidas.

	Parte do primeiro caso da base e altera alguns indicadores para
	simular agravamento clinico, sem precisar de arquivo externo de consulta.
	"""
	first_case_problem = next(iter(full_casebase.values()))["problem"].copy()
	first_case_problem["anxiety_score"] = min(10.0, first_case_problem["anxiety_score"] + 2.0)
	first_case_problem["sleep_quality"] = max(1.0, first_case_problem["sleep_quality"] - 2.0)
	first_case_problem["clinical_severity"] = "high"
	first_case_problem["main_issue"] = (
		"relata piora de ansiedade, insonia e dificuldade para manter rotina profissional"
	)
	return first_case_problem


def _load_query_from_json(path: str) -> dict[str, Any]:
	"""Carrega um novo caso a partir de JSON e aplica normalizacao de entrada."""
	payload = json.loads(Path(path).read_text(encoding="utf-8"))
	return normalize_query(payload)


def run(dataset_path: str, query: dict[str, Any], k: int) -> None:
	"""Executa o ciclo principal CBR: recuperar vizinhos e adaptar solucao.

	Passos:
	1) carregar base e preparar modelo de similaridade
	2) recuperar top-k casos mais similares
	3) adaptar recomendacao com base nos vizinhos
	4) exibir resultados no terminal
	"""
	full_casebase, numeric_stats = load_casebase(dataset_path)
	problem_casebase = build_problem_only_casebase(full_casebase)

	similarity_model = build_similarity_model(numeric_stats)
	retriever = cbrkit.retrieval.dropout(
		cbrkit.retrieval.build(similarity_model),
		limit=k,
	)

	result = cbrkit.retrieval.apply_query(problem_casebase, query, retriever)
	top_case_ids = list(result.ranking)
	similarity_scores = {
		case_id: float(getattr(sim, "value", sim))
		for case_id, sim in result.similarities.items()
	}

	print("=== Consulta CBR Psicologia ===")
	print(f"Casos recuperados (top-{k}):")
	for idx, case_id in enumerate(top_case_ids, start=1):
		score = similarity_scores[case_id]
		sol = full_casebase[case_id]["solution"]
		print(
			f"{idx}. {case_id} | similaridade={score:.4f} | "
			f"intervention_type={sol['intervention_type']} | intensity={sol['intensity']}"
		)

	adapted = adapt_solution(
		query=query,
		retrieved_case_ids=top_case_ids,
		full_casebase=full_casebase,
		similarities=similarity_scores,
	)

	print("\n=== Solucao Adaptada ===")
	print(json.dumps(adapted, indent=2, ensure_ascii=False))


def _evaluate_predictions(
	pairs: list[tuple[dict[str, Any], dict[str, Any], float]],
) -> dict[str, float]:
	"""Calcula metricas agregadas para avaliar as solucoes previstas.

	Entrada: lista de tuplas com (solucao esperada, solucao prevista,
	similaridade do melhor vizinho).
	
	Metricas calculadas:
	- acuracia de intervention_type
	- MAE de intensidade
	- MAE de frequencia semanal
	- similaridade textual Jaccard da recomendacao
	- media de similaridade top-1
	"""
	total = len(pairs)
	if total == 0:
		return {
			"cases_evaluated": 0,
			"intervention_accuracy": 0.0,
			"intensity_mae": 0.0,
			"weekly_frequency_mae": 0.0,
			"recommendation_text_jaccard": 0.0,
			"avg_top1_similarity": 0.0,
		}

	correct_intervention = 0
	intensity_error = 0.0
	frequency_error = 0.0
	text_similarity_sum = 0.0
	top1_similarity_sum = 0.0

	for expected, predicted, top1_similarity in pairs:
		if expected["intervention_type"] == predicted["intervention_type"]:
			correct_intervention += 1

		intensity_error += abs(float(expected["intensity"]) - float(predicted["intensity"]))
		frequency_error += abs(float(expected["weekly_frequency"]) - float(predicted["weekly_frequency"]))
		text_similarity_sum += text_jaccard_similarity(
			expected["recommendation_text"],
			predicted["recommendation_text"],
		)
		top1_similarity_sum += top1_similarity

	return {
		"cases_evaluated": total,
		"intervention_accuracy": correct_intervention / total,
		"intensity_mae": intensity_error / total,
		"weekly_frequency_mae": frequency_error / total,
		"recommendation_text_jaccard": text_similarity_sum / total,
		"avg_top1_similarity": top1_similarity_sum / total,
	}


def _build_kfold_splits(case_ids: list[str], folds: int) -> list[list[str]]:
	"""Divide IDs em folds com tamanhos o mais equilibrados possivel."""
	fold_sizes = [len(case_ids) // folds] * folds
	for idx in range(len(case_ids) % folds):
		fold_sizes[idx] += 1

	splits: list[list[str]] = []
	start = 0
	for size in fold_sizes:
		end = start + size
		splits.append(case_ids[start:end])
		start = end
	return splits


def _validate_kfold_splits(case_ids: list[str], fold_splits: list[list[str]]) -> None:
	"""Valida integridade dos folds para evitar vazamento e perdas de casos."""
	flattened = [case_id for fold in fold_splits for case_id in fold]

	if len(flattened) != len(case_ids):
		raise ValueError("K-Fold invalido: quantidade de casos em folds difere do total.")

	if len(set(flattened)) != len(flattened):
		raise ValueError("K-Fold invalido: ha casos repetidos em mais de um fold.")

	if set(flattened) != set(case_ids):
		raise ValueError("K-Fold invalido: alguns casos nao foram alocados aos folds.")


def validate_kfold(
	full_casebase: dict[str, dict[str, Any]],
	k: int,
	folds: int,
	seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
	"""Executa validacao K-Fold e retorna metricas por fold e agregadas.

	Fluxo:
	- embaralha casos com semente fixa para reprodutibilidade
	- em cada fold, usa uma parte para teste e o restante para treino
	- avalia cada consulta do fold e agrega resultados
	"""
	case_ids = list(full_casebase.keys())
	rng = random.Random(seed)
	rng.shuffle(case_ids)
	fold_splits = _build_kfold_splits(case_ids, folds)
	_validate_kfold_splits(case_ids, fold_splits)

	all_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
	fold_metrics: list[dict[str, Any]] = []

	for fold_index, test_ids in enumerate(fold_splits, start=1):
		test_set = set(test_ids)
		training_casebase = {
			cid: case
			for cid, case in full_casebase.items()
			if cid not in test_set
		}

		fold_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
		for case_id in test_ids:
			query_case = full_casebase[case_id]
			retrieved_ids, similarities = _retrieve_neighbors(
				training_casebase=training_casebase,
				query=query_case["problem"],
				k=k,
			)
			predicted_solution = adapt_solution(
				query=query_case["problem"],
				retrieved_case_ids=retrieved_ids,
				full_casebase=training_casebase,
				similarities=similarities,
			)
			top1_similarity = similarities.get(retrieved_ids[0], 0.0) if retrieved_ids else 0.0
			fold_pairs.append((query_case["solution"], predicted_solution, top1_similarity))

		fold_result = _evaluate_predictions(fold_pairs)
		fold_result["fold"] = fold_index
		fold_result["test_cases"] = len(test_ids)
		fold_result["train_cases"] = len(training_casebase)
		fold_metrics.append(fold_result)
		all_pairs.extend(fold_pairs)

	return _evaluate_predictions(all_pairs), fold_metrics


def run_validation(
	dataset_path: str,
	validation_method: str,
	k: int,
	folds: int,
	seed: int,
) -> None:
	"""Executa a validacao K-Fold solicitada no CLI."""
	full_casebase, _ = load_casebase(dataset_path)
	total_cases = len(full_casebase)

	if validation_method == "kfold":
		if folds < 2:
			raise ValueError("--folds deve ser >= 2 para K-Fold.")
		if folds > total_cases:
			raise ValueError(f"--folds nao pode ser maior que o numero de casos ({total_cases}).")

		print(f"=== Validacao K-Fold (k={k}, folds={folds}) ===")
		overall_metrics, fold_metrics = validate_kfold(
			full_casebase=full_casebase,
			k=k,
			folds=folds,
			seed=seed,
		)

		print("\nMetricas por fold:")
		for item in fold_metrics:
			print(json.dumps(item, ensure_ascii=False))

		print("\nMetricas agregadas:")
		print(json.dumps(overall_metrics, indent=2, ensure_ascii=False))
		return

	raise ValueError("Metodo de validacao invalido. Use: kfold.")


def parse_args() -> argparse.Namespace:
	"""Define e processa argumentos de linha de comando do programa."""
	parser = argparse.ArgumentParser(
		description="Sistema CBR para recomendacao psicologica com CBRkit."
	)
	parser.add_argument(
		"--dataset",
		default=DEFAULT_DATASET_PATH,
		help="Caminho para o CSV de casos.",
	)
	parser.add_argument(
		"--query-json",
		default=None,
		help="Arquivo JSON com um novo caso (somente atributos de problema).",
	)
	parser.add_argument(
		"-k",
		type=int,
		default=5,
		help="Numero de vizinhos mais proximos para recuperacao.",
	)
	parser.add_argument(
		"--validation",
		choices=["none", "kfold"],
		default="none",
		help="Metodo de validacao: none ou kfold.",
	)
	parser.add_argument(
		"--folds",
		type=int,
		default=5,
		help="Numero de folds para K-Fold (usado com --validation kfold).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Semente para embaralhamento no K-Fold.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.validation != "none":
		run_validation(
			dataset_path=args.dataset,
			validation_method=args.validation,
			k=args.k,
			folds=args.folds,
			seed=args.seed,
		)
		raise SystemExit(0)

	casebase, _ = load_casebase(args.dataset)

	query = _load_query_from_json(args.query_json) if args.query_json else _build_demo_query(casebase)
	run(dataset_path=args.dataset, query=query, k=args.k)
