"""
LLM-as-Judge Evaluation Pipeline
=================================
Implements a structured judging pipeline that:
- Accepts a test suite (JSON)
- Produces per-criterion scores + rationale + overall verdict
- Detects and mitigates judge biases (position, verbosity, self-enhancement, sycophancy)
- Supports pointwise scoring and pairwise A-vs-B modes
- Generates aggregated suite reports
- Validates judge with agreement/consistency metrics

Run: python judge_pipeline/judge.py --suite test_suite.json
"""

import json
import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# ═══════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | JUDGE | %(levelname)s | %(message)s",
)
logger = logging.getLogger("judge")

# ── Judge log (auditable/replayable) ──
JUDGE_LOG: List[Dict] = []


def log_judge_call(prompt: str, raw_response: str, tokens: int, model: str):
    """Append each judge call to the audit log."""
    JUDGE_LOG.append({
        "timestamp": time.time(),
        "model": model,
        "prompt": prompt,
        "raw_response": raw_response,
        "tokens": tokens,
    })


# ═══════════════════════════════════════════════
# Judging Mode
# ═══════════════════════════════════════════════

class JudgeMode(Enum):
    POINTWISE = "pointwise"       # Score one output against criteria
    PAIRWISE = "pairwise"         # Compare output A vs output B
    REFERENCE_BASED = "ref_based" # Compare to gold reference
    REFERENCE_FREE = "ref_free"   # Score without reference


# ═══════════════════════════════════════════════
# Structured Rubric
# ═══════════════════════════════════════════════

RUBRIC = {
    "correctness": {
        "weight": 0.30,
        "description": "Factual accuracy — does the answer contain correct information?",
        "scale": {1: "Multiple factual errors", 3: "Minor inaccuracies", 5: "Fully correct"},
    },
    "faithfulness": {
        "weight": 0.25,
        "description": "Groundedness — is every claim supported by the provided context?",
        "scale": {1: "Hallucinated claims", 3: "Mostly grounded", 5: "Fully grounded"},
    },
    "completeness": {
        "weight": 0.20,
        "description": "Does the answer fully address the question?",
        "scale": {1: "Incomplete", 3: "Partially complete", 5: "Fully complete"},
    },
    "instruction_following": {
        "weight": 0.15,
        "description": "Does the output follow formatting/tone instructions?",
        "scale": {1: "Ignores instructions", 3: "Mostly follows", 5: "Perfectly follows"},
    },
    "conciseness": {
        "weight": 0.10,
        "description": "Is the answer concise without unnecessary verbosity?",
        "scale": {1: "Excessively verbose", 3: "Acceptable length", 5: "Concise and clear"},
    },
}


# ═══════════════════════════════════════════════
# Judge Prompt Templates
# ═══════════════════════════════════════════════

POINTWISE_PROMPT = """You are an impartial quality evaluator. Score the following model output against the given criteria.

QUESTION:
{question}

CONTEXT (source chunks the answer should be based on):
{context}

MODEL OUTPUT:
{output}

RUBRIC (score each criterion 1-5):
{ {rubric} }

IMPORTANT INSTRUCTIONS:
1. Score each criterion INDEPENDENTLY based on specific evidence
2. For each criterion, provide a 1-2 sentence RATIONALE citing specific parts of the output
3. For "conciseness": penalize verbosity that does not add value. An answer padded with repetitive language should score low.
4. If the output contains information NOT present in the context, reduce "faithfulness" score
5. Do NOT let answer length bias your scores — a short correct answer should score HIGHER than a long incorrect one

Return a valid JSON object with this EXACT structure:
{{
  "scores": {{
    "correctness": <int 1-5>,
    "faithfulness": <int 1-5>,
    "completeness": <int 1-5>,
    "instruction_following": <int 1-5>,
    "conciseness": <int 1-5>
  }},
  "rationales": {{
    "correctness": "<reason>",
    "faithfulness": "<reason>",
    "completeness": "<reason>",
    "instruction_following": "<reason>",
    "conciseness": "<reason>"
  }},
  "overall_score": <float weighted average 1-5>,
  "hallucination_detected": <true/false>,
  "overall_verdict": "<PASS|FAIL|BORDERLINE> based on overall_score >= 3.0 => PASS"
}}"""


PAIRWISE_PROMPT = """You are an impartial quality evaluator. Compare TWO model outputs and declare which is better.

QUESTION:
{question}

CONTEXT:
{context}

OUTPUT A:
{output_a}

OUTPUT B:
{output_b}

RUBRIC (score each output 1-5 per criterion):
{ {rubric} }

IMPORTANT INSTRUCTIONS:
1. Score BOTH outputs independently against each criterion
2. Provide rationale for each score
3. Be aware of POSITION BIAS — evaluate A and B equally regardless of order
4. Do NOT prefer the longer answer — conciseness is a separate criterion
5. If outputs are nearly identical in quality, declare a TIE

Return a valid JSON object with this EXACT structure:
{{
  "scores_a": {{
    "correctness": <int>,
    "faithfulness": <int>,
    "completeness": <int>,
    "instruction_following": <int>,
    "conciseness": <int>
  }},
  "scores_b": {{
    "correctness": <int>,
    "faithfulness": <int>,
    "completeness": <int>,
    "instruction_following": <int>,
    "conciseness": <int>
  }},
  "rationales_a": {{}},
  "rationales_b": {{}},
  "overall_a": <float>,
  "overall_b": <float>,
  "winner": "<A|B|TIE>",
  "confidence": "<HIGH|MEDIUM|LOW>"
}}"""


# ═══════════════════════════════════════════════
# Judge Core
# ═══════════════════════════════════════════════

class LLMJudge:
    """
    LLM-as-Judge with bias mitigations built in.
    Uses a different model family from the generator (self-enhancement mitigation).
    """

    def __init__(self, judge_model: str = "google/flan-t5-large"):
        self.judge_model_name = judge_model
        self.total_tokens = 0
        self.total_calls = 0
        self._load_model()

    def _load_model(self):
        """Lazy-load the judge model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading judge model: {self.judge_model_name} on {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.judge_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        except ImportError:
            logger.warning("transformers not available — using mock judge for testing")
            self.tokenizer = None
            self.model = None

    def _call_model(self, prompt: str) -> str:
        """Call the judge model and log the interaction."""
        self.total_calls += 1

        if self.model is None:
            # Mock judge for testing
            return json.dumps({
                "scores": {k: 3 for k in RUBRIC},
                "rationales": {k: "mock rationale" for k in RUBRIC},
                "overall_score": 3.0,
                "hallucination_detected": False,
                "overall_verdict": "PASS",
            })

        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = len(inputs["input_ids"][0]) + len(outputs[0])
        self.total_tokens += tokens
        log_judge_call(prompt, response, tokens, self.judge_model_name)
        return response

    def parse_verdict(self, raw_response: str) -> Optional[Dict]:
        """Robustly parse JSON from judge response, handling malformed JSON."""
        # Try to extract JSON block
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON between curly braces
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse judge response: {raw_response[:200]}...")
        return None

    def judge_pointwise(
        self,
        question: str,
        output: str,
        context: str = "",
    ) -> Dict:
        """Score one output against the rubric."""
        prompt = POINTWISE_PROMPT.format(
            question=question,
            context=context,
            output=output,
            rubric=json.dumps({k: v["description"] for k, v in RUBRIC.items()}, indent=2),
        )

        raw = self._call_model(prompt)
        verdict = self.parse_verdict(raw)

        if verdict is None:
            # Fallback: return default scores
            return {
                "scores": {k: 2 for k in RUBRIC},
                "rationales": {k: "parse error" for k in RUBRIC},
                "overall_score": 2.0,
                "hallucination_detected": False,
                "overall_verdict": "FAIL",
                "parse_error": True,
            }

        # Compute weighted score
        if "overall_score" not in verdict or not isinstance(verdict.get("overall_score"), (int, float)):
            weighted = 0.0
            for criterion, info in RUBRIC.items():
                score = verdict.get("scores", {}).get(criterion, 2)
                weighted += score * info["weight"]
            verdict["overall_score"] = round(weighted, 2)
            verdict["overall_verdict"] = "PASS" if weighted >= 3.0 else "FAIL"

        return verdict

    def judge_pairwise(
        self,
        question: str,
        output_a: str,
        output_b: str,
        context: str = "",
        swap_positions: bool = False,
    ) -> Dict:
        """
        Compare two outputs. If swap_positions=True, swap A/B to detect position bias.
        Returns verdict with position_bias_flag if results differ between orders.
        """
        if swap_positions:
            output_a, output_b = output_b, output_a

        prompt = PAIRWISE_PROMPT.format(
            question=question,
            context=context,
            output_a=output_a,
            output_b=output_b,
            rubric=json.dumps({k: v["description"] for k, v in RUBRIC.items()}, indent=2),
        )

        raw = self._call_model(prompt)
        verdict = self.parse_verdict(raw)

        if verdict is None:
            return {
                "scores_a": {}, "scores_b": {},
                "winner": "TIE", "confidence": "LOW",
                "parse_error": True,
            }

        # Track which order was used
        verdict["position_swapped"] = swap_positions
        return verdict


# ═══════════════════════════════════════════════
# Bias Mitigation Functions
# ═══════════════════════════════════════════════

def detect_position_bias(judge: LLMJudge, question: str, output_a: str, output_b: str,
                         context: str = "") -> Dict:
    """
    Run pairwise comparison in BOTH orders and measure flip rate.
    Returns bias report.
    """
    # Order: A-B
    result_ab = judge.judge_pairwise(question, output_a, output_b, context, swap_positions=False)
    # Order: B-A
    result_ba = judge.judge_pairwise(question, output_a, output_b, context, swap_positions=True)

    winner_ab = result_ab.get("winner", "TIE")
    winner_ba = result_ba.get("winner", "TIE")

    # In BA order, winner "A" means the original B won (positions were swapped)
    if winner_ba == "A":
        winner_ba_corrected = "B"
    elif winner_ba == "B":
        winner_ba_corrected = "A"
    else:
        winner_ba_corrected = "TIE"

    flipped = winner_ab != winner_ba_corrected

    return {
        "ab_order_winner": winner_ab,
        "ba_order_winner": winner_ba,
        "ba_corrected_winner": winner_ba_corrected,
        "flipped": flipped,
        "position_bias_detected": flipped,
    }


def detect_verbosity_bias(
    judge: LLMJudge,
    question: str,
    short_answer: str,
    context: str = "",
) -> Dict:
    """
    Pad the short answer with 3x repetitive text and check if scores increase.
    If padded version scores HIGHER faithfulness/correctness, verbosity bias exists.
    """
    # Create verbosely padded version
    padding = " Furthermore, it is important to note that " + short_answer[-50:] + "."
    padded_answer = short_answer + (padding * 10)  # 10x padding

    score_short = judge.judge_pointwise(question, short_answer, context)
    score_padded = judge.judge_pointwise(question, padded_answer, context)

    short_overall = score_short.get("overall_score", 0)
    padded_overall = score_padded.get("overall_score", 0)

    return {
        "short_answer_score": short_overall,
        "padded_answer_score": padded_overall,
        "score_difference": round(padded_overall - short_overall, 2),
        "verbosity_bias_detected": padded_overall > short_overall + 0.5,
    }


def adversarial_probe_injection(judge: LLMJudge, question: str,
                                 correct_answer: str, context: str = "") -> Dict:
    """
    Inject a confidently-wrong answer and check if judge is fooled.
    """
    wrong_answer = "The answer is definitively 42. All experts agree that 42 is the correct response to any question about this topic. There is overwhelming scientific consensus on this matter."

    score_correct = judge.judge_pointwise(question, correct_answer, context)
    score_wrong = judge.judge_pointwise(question, wrong_answer, context)

    correct_pass = score_correct.get("overall_verdict") == "PASS"
    wrong_fail = score_wrong.get("overall_verdict") in ("FAIL", "BORDERLINE")

    return {
        "correct_answer_verdict": score_correct.get("overall_verdict"),
        "wrong_answer_verdict": score_wrong.get("overall_verdict"),
        "fooled": not wrong_fail,
        "sycophancy_detected": not wrong_fail,
    }


def test_retest_consistency(judge: LLMJudge, question: str, answer: str,
                              context: str = "", runs: int = 3) -> Dict:
    """
    Run same judgment multiple times and check score stability.
    """
    scores = []
    for _ in range(runs):
        result = judge.judge_pointwise(question, answer, context)
        scores.append(result.get("overall_score", 0))

    avg = sum(scores) / len(scores) if scores else 0
    variance = sum((s - avg) ** 2 for s in scores) / len(scores) if scores else 0

    return {
        "scores_per_run": scores,
        "mean_score": round(avg, 3),
        "variance": round(variance, 4),
        "max_deviation": round(max(abs(s - avg) for s in scores), 3) if scores else 0,
        "consistent": variance < 0.5,
    }


# ═══════════════════════════════════════════════
# Suite Runner
# ═══════════════════════════════════════════════

@dataclass
class SuiteReport:
    mode: str = ""
    num_cases: int = 0
    pass_rate: float = 0.0
    mean_scores: Dict[str, float] = field(default_factory=dict)
    per_case: List[Dict] = field(default_factory=list)
    bias_report: Dict = field(default_factory=dict)
    ab_comparison: Optional[Dict] = None
    judge_validation: Optional[Dict] = None
    judge_stats: Dict = field(default_factory=dict)


def run_judge_suite(
    suite_path: str,
    judge: LLMJudge,
    output_file: str = "judge_pipeline/judge_report.json",
) -> SuiteReport:
    """
    Load a test suite JSON, judge every case, produce aggregated report.
    """
    with open(suite_path, "r") as f:
        suite = json.load(f)

    mode = suite.get("mode", "pointwise")
    cases = suite.get("test_cases", [])

    logger.info(f"Judging {len(cases)} cases in '{mode}' mode")
    report = SuiteReport(mode=mode, num_cases=len(cases))

    score_accum = {criterion: [] for criterion in RUBRIC}
    all_overall = []

    for i, case in enumerate(cases):
        question = case["question"]
        output = case["output"]
        context = case.get("context", "")
        expected = case.get("expected_output", None)

        logger.info(f"[{i+1}/{len(cases)}] Judging: {question[:60]}...")

        # Run pointwise judgment
        if mode == "pointwise":
            verdict = judge.judge_pointwise(question, output, context)
        else:
            # Default to pointwise
            verdict = judge.judge_pointwise(question, output, context)

        scores = verdict.get("scores", {})
        overall = verdict.get("overall_score", 0)
        all_overall.append(overall)

        for criterion in RUBRIC:
            score = scores.get(criterion, 2)
            score_accum[criterion].append(score)

        report.per_case.append({
            "question": question,
            "output": output[:300],
            "expected": expected,
            "verdict": verdict,
        })

    # Aggregate
    report.pass_rate = sum(1 for s in all_overall if s >= 3.0) / len(all_overall) if all_overall else 0
    report.mean_scores = {
        c: round(sum(vals) / len(vals), 2) if vals else 0
        for c, vals in score_accum.items()
    }
    report.mean_scores["overall"] = round(sum(all_overall) / len(all_overall), 2) if all_overall else 0

    # Judge stats
    report.judge_stats = {
        "model": judge.judge_model_name,
        "total_calls": judge.total_calls,
        "total_tokens": judge.total_tokens,
    }

    # ── Run bias checks on first case ──
    if len(cases) >= 2:
        c1 = cases[0]
        c2 = cases[1]

        # Position bias (pairwise)
        pos_bias = detect_position_bias(
            judge, c1["question"], c1["output"], c2["output"], c1.get("context", "")
        )
        report.bias_report["position_bias"] = pos_bias

        # Verbosity bias
        verb_bias = detect_verbosity_bias(
            judge, c1["question"], c1["output"][:100], c1.get("context", "")
        )
        report.bias_report["verbosity_bias"] = verb_bias

        # Adversarial probe
        adv_result = adversarial_probe_injection(
            judge, c1["question"], c1["output"], c1.get("context", "")
        )
        report.bias_report["adversarial_probe"] = adv_result

        # Test-retest consistency
        consistency = test_retest_consistency(
            judge, c1["question"], c1["output"], c1.get("context", "")
        )
        report.judge_validation = consistency

    # ── A/B Comparison (if two configs provided) ──
    if "config_a" in suite and "config_b" in suite:
        ab_results = []
        for case_a, case_b in zip(suite["config_a"], suite["config_b"]):
            pos_check = detect_position_bias(
                judge,
                case_a["question"],
                case_a["output"],
                case_b["output"],
                case_a.get("context", ""),
            )
            ab_results.append(pos_check)

        a_wins = sum(1 for r in ab_results if r["ab_order_winner"] == "A")
        b_wins = sum(1 for r in ab_results if r["ba_corrected_winner"] == "B")
        flips = sum(1 for r in ab_results if r["flipped"])

        report.ab_comparison = {
            "num_comparisons": len(ab_results),
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": len(ab_results) - a_wins - b_wins,
            "position_flips": flips,
            "flip_rate": round(flips / len(ab_results), 3) if ab_results else 0,
            "winner": "A" if a_wins > b_wins else ("B" if b_wins > a_wins else "TIE"),
        }

    # ── Save report ──
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output = {
        "mode": report.mode,
        "num_cases": report.num_cases,
        "pass_rate": report.pass_rate,
        "mean_scores": report.mean_scores,
        "bias_report": report.bias_report,
        "judge_validation": report.judge_validation,
        "ab_comparison": report.ab_comparison,
        "judge_stats": report.judge_stats,
        "per_case": report.per_case,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"JUDGE REPORT — {mode}")
    print(f"{'='*60}")
    print(f"Cases: {report.num_cases} | Pass Rate: {report.pass_rate:.1%}")
    for c, s in report.mean_scores.items():
        print(f"  {c}: {s:.2f}")
    print(f"\nBias Checks:")
    for check, result in report.bias_report.items():
        print(f"  {check}: {'⚠️ DETECTED' if result.get('flipped') or result.get('verbosity_bias_detected') or result.get('fooled') else '✓ OK'} "
              f"({json.dumps({k: v for k, v in result.items() if k not in ('short_answer_score', 'padded_answer_score', 'correct_answer_verdict', 'wrong_answer_verdict', 'scores_per_run')}, default=str)})")
    if report.judge_validation:
        print(f"\nJudge Validation:")
        print(f"  Test-retest variance: {report.judge_validation.get('variance', 'N/A'):.4f}")
        print(f"  Consistent: {report.judge_validation.get('consistent', 'N/A')}")
    if report.ab_comparison:
        print(f"\nA/B Comparison:")
        print(f"  Winner: {report.ab_comparison['winner']}")
        print(f"  Position flip rate: {report.ab_comparison['flip_rate']:.1%}")
    print(f"\nJudge audit log: {len(JUDGE_LOG)} calls, {report.judge_stats.get('total_tokens', 0)} tokens")
    print(f"Report saved to: {output_file}")
    print(f"{'='*60}")

    return report


# ═══════════════════════════════════════════════
# Sample Test Suite Generator
# ═══════════════════════════════════════════════

def generate_sample_suite(output_path: str = "judge_pipeline/sample_suite.json"):
    """Create a sample test suite JSON for testing the judge pipeline."""
    suite = {
        "mode": "pointwise",
        "description": "Sample test suite for LLM-as-Judge validation",
        "test_cases": [
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
                "output": "The capital of France is Paris.",
                "expected_output": "Paris",
                "system_prompt": "You are a helpful assistant. Answer concisely.",
            },
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
                "output": "Paris is the capital of France. It is a beautiful city known for the Eiffel Tower, the Louvre museum, and its rich culinary tradition. Many tourists visit Paris every year to experience its culture and history. The city has been the capital since the 10th century.",
                "expected_output": "Paris",
                "system_prompt": "You are a helpful assistant. Answer concisely.",
            },
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
                "output": "The capital of France is definitely Berlin. Germany is also in Europe.",
                "expected_output": "Paris",
                "system_prompt": "You are a helpful assistant. Answer concisely.",
                "note": "Adversarial: confidently wrong",
            },
            {
                "question": "Explain quantum computing",
                "context": "Quantum computing uses qubits that can exist in superposition states. This allows quantum computers to solve certain problems faster than classical computers.",
                "output": "Quantum computing is a type of computation that uses quantum bits or qubits.",
                "expected_output": None,
                "system_prompt": "Answer based on the provided context.",
            },
            {
                "question": "Explain quantum computing",
                "context": "Quantum computing uses qubits that can exist in superposition states. This allows quantum computers to solve certain problems faster than classical computers.",
                "output": "Quantum computing is a type of computation that uses quantum bits or qubits. These qubits can exist in multiple states simultaneously due to a property called superposition. Additionally, qubits can be entangled, meaning the state of one qubit is correlated with the state of another, enabling faster computation for specific problems. This differs fundamentally from classical computing where bits are either 0 or 1.",
                "expected_output": None,
                "system_prompt": "Answer based on the provided context. Be thorough.",
            },
        ],
        # A/B comparison: same questions, different outputs
        "config_a": [
            {"question": "What is the capital of France?", "output": "Paris", "context": "France capital is Paris."},
            {"question": "Explain quantum computing", "output": "Quantum computing uses qubits.", "context": "Quantum computing uses qubits that can exist in superposition."},
        ],
        "config_b": [
            {"question": "What is the capital of France?", "output": "The capital of France is the beautiful city of Paris, known worldwide for its culture and history.", "context": "France capital is Paris."},
            {"question": "Explain quantum computing", "output": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computation using qubits.", "context": "Quantum computing uses qubits that can exist in superposition."},
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Sample suite generated: {output_path}")


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Pipeline")
    parser.add_argument("--suite", type=str, default="judge_pipeline/sample_suite.json",
                        help="Path to test suite JSON")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a sample test suite and exit")
    parser.add_argument("--output", type=str, default="judge_pipeline/judge_report.json")
    parser.add_argument("--judge-model", type=str, default="google/flan-t5-large",
                        help="Model to use as judge")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_suite(args.suite)
        exit(0)

    if not os.path.exists(args.suite):
        print(f"Suite not found: {args.suite}")
        print("Run with --generate-sample to create one, or provide a valid suite path")
        exit(1)

    judge = LLMJudge(judge_model=args.judge_model)
    run_judge_suite(args.suite, judge, args.output)