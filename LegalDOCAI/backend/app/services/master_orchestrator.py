import os
import time
import asyncio
import traceback
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# --- MODULE IMPORTS ---
# 1. OCR & Ingestion
from backend.app.services.ocr_service import extract_text
# 2. NLP (Translation, Lang, NER, Summarize)
from backend.app.nlp import (
    translate_to_english_async,
    detect_language,
    perform_ner_async,
    legal_summarize_async
)
# 3. Authenticity
from backend.app.services.verification_service import verify_document_async
# 4. Classification
from backend.app.services.ml_service import classifier
# 5. Analysis (Risk, Timeline, Clause)
from backend.app.services.analysis_service import (
    clause_risk_detection_async,
    build_timeline_async,
    check_mandatory_fields_async,
)
# 6. Compliance
from backend.app.services.compliance_service import jurisdiction_checks_async
# 7. Comparison
from backend.app.services.comparison_service import comparison_service
# 8. RAG
from backend.app.services.rag_service import legal_rag_service
# 9. Report
from backend.app.services.report_service import report_service

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalAIPipeline")


class PipelineConfig(BaseModel):
    """Configuration class for the pipeline."""
    enable_rag: bool = True
    enable_comparison: bool = False
    language_force: Optional[str] = None
    debug_mode: bool = False
    max_pages: int = 50
    output_dir: str = "d:\\Legal-mohan\\Legaldoc-new\\LegalDOCAI\\outputs"


class ResultAggregator:
    """Collects normalized module outputs, warnings, errors, and timings."""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.timings: Dict[str, float] = {}

    def reset(self) -> None:
        self.results = {}
        self.warnings = []
        self.errors = []
        self.timings = {}

    def add_result(self, module_name: str, data: Any, timing: float):
        self.results[module_name] = data
        self.timings[module_name] = round(timing, 3)

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def add_error(self, error: str):
        self.errors.append(error)

    def compute_overall_risk(self) -> str:
        risk_data = self.results.get("risk_analysis", {})
        if isinstance(risk_data, dict):
            return str(risk_data.get("overall_risk", "Low"))
        return "Low"

    def build_final_json(self, status: str, start_time: float) -> Dict[str, Any]:
        translation = self.results.get("translation", {})
        if not isinstance(translation, dict):
            translation = {}

        rag_index_data = self.results.get("rag_index", {})
        if isinstance(rag_index_data, dict):
            rag_ready = bool(rag_index_data.get("indexed", False))
        else:
            rag_ready = bool(rag_index_data)

        ingestion = self.results.get("ingestion", {})
        if not isinstance(ingestion, dict):
            ingestion = {}

        return {
            "status": status,
            "document_text": ingestion.get("text", ""),
            "language": translation.get("language", "en"),
            "translated_text": translation.get("translated_text", ""),
            "authenticity": self.results.get("authenticity", {}),
            "classification": self.results.get("classification", {}),
            "entities": self.results.get("ner", {}),
            "risk_analysis": self.results.get("risk_analysis", {}),
            "summary": self.results.get("summarization", {}),
            "timeline": self.results.get("timeline", []),
            "clauses_present": self.results.get("clause_checker", {}),
            "compliance": self.results.get("compliance", {}),
            "comparison": self.results.get("comparison", {}),
            "rag_ready": rag_ready,
            "rag_response": self.results.get("rag_response", {}),
            "report_path": self.results.get("report_generation", ""),
            "processing_time_sec": round(time.time() - start_time, 2),
            "module_timings": self.timings,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class PipelineConnectivityValidator:
    """Validates connectivity, schema consistency, orphan outputs, and risk points."""

    REQUIRED_MODULES = {
        "ingestion",
        "translation",
        "authenticity",
        "classification",
        "ner",
        "risk_analysis",
        "summarization",
        "timeline",
        "clause_checker",
        "compliance",
        "report_generation",
    }

    MODULE_SCHEMA_RULES = {
        "ingestion": {"type": dict, "required_keys": {"text", "num_pages"}},
        "translation": {"type": dict, "required_keys": {"language", "translated_text"}},
        "authenticity": {"type": dict, "required_keys": {"label", "confidence"}},
        "classification": {"type": dict, "required_keys": {"document_type", "confidence", "version"}},
        "ner": {"type": dict, "required_keys": {"names", "organizations", "dates", "money"}},
        "risk_analysis": {"type": dict, "required_keys": {"overall_risk", "clauses"}},
        "summarization": {"type": dict, "required_keys": {"summary", "key_points"}},
        "timeline": {"type": list, "required_keys": set()},
        "clause_checker": {"type": dict, "required_keys": {"present", "missing"}},
        "compliance": {"type": dict, "required_keys": {"jurisdiction", "checks", "overall_compliance_score", "violations"}},
        "comparison": {"type": dict, "required_keys": {"overall_change_score", "summary", "clause_changes"}},
        "rag_index": {"type": dict, "required_keys": {"enabled", "indexed", "status"}},
        "report_generation": {"type": str, "required_keys": set()},
    }

    CONSUMED_OUTPUT_KEYS = {
        "ingestion": {"text", "num_pages", "file_name", "file_path"},
        "translation": {"language", "translated_text"},
        "authenticity": {"label", "confidence", "details"},
        "classification": {"document_type", "confidence", "version"},
        "ner": {"names", "organizations", "dates", "money", "amounts", "locations", "metadata"},
        "risk_analysis": {"overall_risk", "overall_score", "clauses", "missing_mandatory_fields", "metadata"},
        "summarization": {"summary", "key_points", "method", "metadata"},
        "timeline": set(),
        "clause_checker": {"present", "missing", "details", "metadata"},
        "compliance": {"jurisdiction", "checks", "overall_compliance_score", "risk_level", "recommendations", "metadata", "violations"},
        "comparison": {"status", "overall_change_score", "summary", "clause_changes"},
        "rag_index": {"enabled", "indexed", "status"},
        "rag_response": {"answer", "confidence", "sources", "retrieval_stats", "status", "reason"},
        "report_generation": set(),
    }

    def validate(
        self,
        module_outputs: Dict[str, Any],
        config: PipelineConfig,
        warnings: List[str],
        errors: List[str],
        auto_fixes_applied: List[str],
    ) -> Dict[str, Any]:
        missing_connections: List[str] = []
        schema_mismatches: List[str] = []
        unused_outputs: List[str] = []
        risk_points: List[str] = []

        required = set(self.REQUIRED_MODULES)
        if config.enable_comparison:
            required.add("comparison")
        if config.enable_rag:
            required.add("rag_index")

        for module_name in sorted(required):
            if module_name not in module_outputs:
                missing_connections.append(f"Missing module output: {module_name}")

        for module_name, data in module_outputs.items():
            rule = self.MODULE_SCHEMA_RULES.get(module_name)
            if not rule:
                continue
            expected_type = rule["type"]
            if not isinstance(data, expected_type):
                schema_mismatches.append(
                    f"{module_name}: expected {expected_type.__name__}, got {type(data).__name__}"
                )
                continue
            required_keys = rule.get("required_keys", set())
            if isinstance(data, dict) and required_keys:
                missing_keys = sorted(k for k in required_keys if k not in data)
                if missing_keys:
                    schema_mismatches.append(f"{module_name}: missing keys {missing_keys}")

        for module_name, data in module_outputs.items():
            consumed = self.CONSUMED_OUTPUT_KEYS.get(module_name)
            if consumed is None:
                unused_outputs.append(f"{module_name}: entire output is orphaned")
                continue
            if isinstance(data, dict) and consumed:
                extra_keys = sorted(k for k in data.keys() if k not in consumed)
                for k in extra_keys:
                    unused_outputs.append(f"{module_name}.{k}")

        for warning in warnings:
            risk_points.append(f"warning: {warning}")
        for error in errors:
            risk_points.append(f"error: {error}")

        if config.enable_rag and "rag_response" not in module_outputs:
            risk_points.append("RAG enabled without retrieval response path")
        if config.enable_comparison:
            comp = module_outputs.get("comparison", {})
            if isinstance(comp, dict) and comp.get("status") == "skipped":
                risk_points.append("Comparison enabled but skipped due to missing/invalid compare_path")

        if errors or missing_connections:
            pipeline_health = "broken"
        elif schema_mismatches or unused_outputs or risk_points:
            pipeline_health = "warning"
        else:
            pipeline_health = "healthy"

        confidence = 0.98
        confidence -= 0.10 * len(errors)
        confidence -= 0.05 * len(missing_connections)
        confidence -= 0.03 * len(schema_mismatches)
        confidence -= 0.01 * len(unused_outputs)
        confidence -= 0.01 * len(risk_points)
        confidence = max(0.0, min(1.0, round(confidence, 2)))

        return {
            "pipeline_health": pipeline_health,
            "missing_connections": missing_connections,
            "schema_mismatches": schema_mismatches,
            "unused_outputs": unused_outputs,
            "risk_points": risk_points,
            "auto_fixes_applied": auto_fixes_applied,
            "confidence": confidence,
        }


class LegalAIPipeline:
    """Master pipeline orchestrator with connectivity validation and auto-fixes."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.aggregator = ResultAggregator()
        self.validator = PipelineConnectivityValidator()
        self.module_outputs: Dict[str, Any] = {}
        self.auto_fixes_applied: List[str] = []

    def _reset_runtime_state(self) -> None:
        self.aggregator.reset()
        self.module_outputs = {}
        self.auto_fixes_applied = []

    def _record_auto_fix(self, message: str) -> None:
        if message not in self.auto_fixes_applied:
            self.auto_fixes_applied.append(message)

    def _save_module_result(self, module_name: str, data: Any, timing: float) -> None:
        self.module_outputs[module_name] = data
        self.aggregator.add_result(module_name, data, timing)

    async def _run_module(self, module_name: str, coro, optional: bool = False) -> Any:
        try:
            return await coro
        except Exception as exc:
            err = f"{module_name} failed: {exc}"
            if optional:
                self.aggregator.add_warning(err)
                return None
            self.aggregator.add_error(err)
            raise

    def _normalize_translation(self, source_language: str, translated_payload: Any, original_text: str) -> Dict[str, Any]:
        language = source_language if source_language else "en"

        if isinstance(translated_payload, dict):
            translated_text = translated_payload.get("translated_text") or translated_payload.get("text") or original_text
            language = translated_payload.get("language", language)
            if "translated_text" not in translated_payload:
                self._record_auto_fix("Normalized translation payload dict missing translated_text")
            return {"language": language, "translated_text": translated_text}

        if isinstance(translated_payload, str):
            self._record_auto_fix("Normalized translation output from string to structured dict")
            return {"language": language, "translated_text": translated_payload or original_text}

        self._record_auto_fix("Recovered translation output from unsupported type")
        return {"language": language, "translated_text": original_text}

    def _normalize_authenticity(self, res: Dict[str, Any]) -> Dict[str, Any]:
        marker = res.get("marker") or res.get("label") or "Unknown"
        ai_conf = res.get("ai_confidence", 0.0)
        conf = round(float(ai_conf) / 100.0, 4) if ai_conf and ai_conf > 1 else round(float(ai_conf or 0.0), 4)
        return {"label": marker, "confidence": conf, "details": res.get("visual_forensics", {})}

    def _normalize_summarization(self, res: Any) -> Dict[str, Any]:
        if isinstance(res, dict):
            if "summary" not in res:
                self._record_auto_fix("Added missing summary key in summarization output")
            return {
                "summary": str(res.get("summary", "")),
                "key_points": res.get("key_points", []) if isinstance(res.get("key_points", []), list) else [],
                "method": res.get("method", "unknown"),
                "metadata": res.get("metadata", {}),
            }
        if isinstance(res, str):
            self._record_auto_fix("Normalized summarization output from string to structured dict")
            return {"summary": res, "key_points": [], "method": "unknown", "metadata": {}}
        self._record_auto_fix("Recovered summarization output from unsupported type")
        return {"summary": "", "key_points": [], "method": "unknown", "metadata": {}}

    def _normalize_ner(self, res: Any) -> Dict[str, Any]:
        if not isinstance(res, dict):
            self._record_auto_fix("Recovered NER output from unsupported type")
            return {"names": [], "organizations": [], "dates": [], "money": [], "amounts": [], "locations": [], "metadata": {}}
        money = res.get("money", [])
        amounts = res.get("amounts", money if isinstance(money, list) else [])
        if "money" not in res and "amounts" in res:
            self._record_auto_fix("Aligned NER monetary key from amounts to money")
        return {
            "names": res.get("names", []),
            "organizations": res.get("organizations", []),
            "dates": res.get("dates", []),
            "money": money if isinstance(money, list) else [],
            "amounts": amounts if isinstance(amounts, list) else [],
            "locations": res.get("locations", []) if isinstance(res.get("locations", []), list) else [],
            "metadata": res.get("metadata", {}),
        }

    def _normalize_risk(self, res: Any) -> Dict[str, Any]:
        if not isinstance(res, dict):
            self._record_auto_fix("Recovered risk output from unsupported type")
            return {"overall_risk": "Low", "overall_score": 0, "clauses": [], "missing_mandatory_fields": [], "metadata": {}}
        clauses = res.get("clauses", [])
        if not isinstance(clauses, list):
            clauses = []
            self._record_auto_fix("Normalized risk clauses to list")
        return {
            "overall_risk": res.get("overall_risk", "Low"),
            "overall_score": res.get("overall_score", 0),
            "clauses": clauses,
            "missing_mandatory_fields": res.get("missing_mandatory_fields", []),
            "metadata": res.get("metadata", {}),
        }

    def _normalize_timeline(self, res: Any) -> List[Dict[str, Any]]:
        if isinstance(res, list):
            return [r for r in res if isinstance(r, dict)]
        self._record_auto_fix("Normalized timeline output to list")
        return []

    def _normalize_clause_checker(self, res: Any) -> Dict[str, Any]:
        if not isinstance(res, dict):
            self._record_auto_fix("Recovered clause checker output from unsupported type")
            return {"present": [], "missing": [], "details": {}, "metadata": {}}
        return {
            "present": res.get("present", []),
            "missing": res.get("missing", []),
            "details": res.get("details", {}),
            "metadata": res.get("metadata", {}),
        }

    def _normalize_compliance(self, res: Any) -> Dict[str, Any]:
        if not isinstance(res, dict):
            self._record_auto_fix("Recovered compliance output from unsupported type")
            return {
                "jurisdiction": "Unknown",
                "checks": [],
                "overall_compliance_score": 0,
                "risk_level": "Unknown",
                "recommendations": [],
                "metadata": {},
                "violations": [],
            }

        checks = res.get("checks", [])
        if not isinstance(checks, list):
            checks = []
            self._record_auto_fix("Normalized compliance checks to list")

        violations = res.get("violations")
        if not isinstance(violations, list):
            violations = [
                c.get("rule", "Unknown check")
                for c in checks
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            self._record_auto_fix("Derived compliance violations from failed checks")

        return {
            "jurisdiction": res.get("jurisdiction", "Unknown"),
            "checks": checks,
            "overall_compliance_score": res.get("overall_compliance_score", 0),
            "risk_level": res.get("risk_level", "Unknown"),
            "recommendations": res.get("recommendations", []),
            "metadata": res.get("metadata", {}),
            "violations": violations,
        }

    async def run_ingestion(self, file_path: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 1: Ingestion & OCR starting...")
        raw = extract_text(file_path)
        if not isinstance(raw, dict):
            self._record_auto_fix("Normalized ingestion output to dict")
            raw = {"text": "", "num_pages": 0}
        data = {
            "text": raw.get("text", "") if isinstance(raw.get("text", ""), str) else "",
            "num_pages": int(raw.get("num_pages", 0) or 0),
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
        }
        self._save_module_result("ingestion", data, time.time() - start)
        return data

    async def run_translation(self, text: str) -> str:
        start = time.time()
        logger.info("Module 2: Translation starting...")
        source_lang = detect_language(text)
        payload: Any = text

        should_translate = bool(text and text.strip()) and source_lang not in {"English", "en"}
        if self.config.language_force and self.config.language_force.lower() != "auto":
            source_lang = self.config.language_force
            should_translate = True
            self._record_auto_fix("Applied language_force override for translation")

        if should_translate:
            payload = await translate_to_english_async(text, src_lang=source_lang)

        normalized = self._normalize_translation(source_lang, payload, text)
        self._save_module_result("translation", normalized, time.time() - start)
        return normalized.get("translated_text", text)

    async def run_authenticity(self, text: str, file_path: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 3: Authenticity Analysis starting...")
        res = await verify_document_async(text, file_path=file_path)
        data = self._normalize_authenticity(res if isinstance(res, dict) else {})
        self._save_module_result("authenticity", data, time.time() - start)
        return data

    async def run_classification(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 4: Classification starting...")
        res = classifier.predict_document_type(text)
        data = {
            "document_type": res[0] if isinstance(res, tuple) and len(res) > 0 else "Unknown",
            "confidence": res[1] if isinstance(res, tuple) and len(res) > 1 else 0.0,
            "version": res[2] if isinstance(res, tuple) and len(res) > 2 else "unknown",
        }
        self._save_module_result("classification", data, time.time() - start)
        return data

    async def run_ner(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 5: NER starting...")
        res = await perform_ner_async(text)
        data = self._normalize_ner(res)
        self._save_module_result("ner", data, time.time() - start)
        return data

    async def run_risk_analysis(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 6: Risk Detection starting...")
        res = await clause_risk_detection_async(text)
        data = self._normalize_risk(res)
        self._save_module_result("risk_analysis", data, time.time() - start)
        return data

    async def run_summarization(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 7: Summarization starting...")
        res = await legal_summarize_async(text)
        data = self._normalize_summarization(res)
        self._save_module_result("summarization", data, time.time() - start)
        return data

    async def run_timeline(self, text: str) -> List[Dict[str, Any]]:
        start = time.time()
        logger.info("Module 8: Timeline Extraction starting...")
        res = await build_timeline_async(text)
        data = self._normalize_timeline(res)
        self._save_module_result("timeline", data, time.time() - start)
        return data

    async def run_clause_checker(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 9: Clause Presence Checker starting...")
        res = await check_mandatory_fields_async(text)
        data = self._normalize_clause_checker(res)
        self._save_module_result("clause_checker", data, time.time() - start)
        return data

    async def run_compliance(self, text: str) -> Dict[str, Any]:
        start = time.time()
        logger.info("Module 10: Regulatory Compliance starting...")
        res = await jurisdiction_checks_async(text)
        data = self._normalize_compliance(res)
        self._save_module_result("compliance", data, time.time() - start)
        return data

    async def run_comparison(self, text: str, compare_path: Optional[str]) -> Dict[str, Any]:
        start = time.time()
        if not compare_path or not os.path.exists(compare_path):
            data = {"status": "skipped", "overall_change_score": 0.0, "summary": {}, "clause_changes": []}
            self._save_module_result("comparison", data, 0.0)
            return data

        logger.info("Module 11: Comparison starting...")
        revised_ocr = extract_text(compare_path)
        revised_text = revised_ocr.get("text", "") if isinstance(revised_ocr, dict) else ""
        res = comparison_service.compare_documents(text, revised_text)
        if not isinstance(res, dict):
            self._record_auto_fix("Recovered comparison output from unsupported type")
            res = {"overall_change_score": 0.0, "summary": {}, "clause_changes": []}

        data = {
            "status": "completed",
            "overall_change_score": res.get("overall_change_score", 0.0),
            "summary": res.get("summary", {}),
            "clause_changes": res.get("clause_changes", []),
        }
        self._save_module_result("comparison", data, time.time() - start)
        return data

    async def run_rag_index(self, text: str, query: Optional[str]) -> Dict[str, Any]:
        start = time.time()
        if not self.config.enable_rag:
            data = {"enabled": False, "indexed": False, "status": "disabled"}
            self._save_module_result("rag_index", data, 0.0)
            return data

        logger.info("Module 12: RAG starting...")
        rag_response = {"status": "skipped", "reason": "No query provided"}

        if query and query.strip() and text.strip():
            rag_res = legal_rag_service.rag_pipeline([text], query)
            if not isinstance(rag_res, dict):
                self._record_auto_fix("Recovered rag response from unsupported type")
                rag_res = {"answer": "", "confidence": 0.0, "sources": [], "retrieval_stats": {}}
            rag_response = rag_res
        elif not text.strip():
            self.aggregator.add_warning("RAG enabled but source text is empty")

        data = {"enabled": True, "indexed": bool(text.strip()), "status": "ready" if text.strip() else "empty_text"}
        self._save_module_result("rag_response", rag_response, time.time() - start)
        self._save_module_result("rag_index", data, time.time() - start)
        return data

    async def run_report_generation(self) -> str:
        start = time.time()
        logger.info("Module 13: Report Generation starting...")

        risk_data = self.aggregator.results.get("risk_analysis", {})
        if not isinstance(risk_data, dict):
            risk_data = {}
        clauses = risk_data.get("clauses", []) if isinstance(risk_data.get("clauses", []), list) else []

        compliance_data = self.aggregator.results.get("compliance", {})
        if not isinstance(compliance_data, dict):
            compliance_data = {}

        entities = self.aggregator.results.get("ner", {})
        if not isinstance(entities, dict):
            entities = {}

        report_data = {
            "document_info": self.aggregator.results.get("ingestion", {}),
            "authenticity": self.aggregator.results.get("authenticity", {}),
            "classification": self.aggregator.results.get("classification", {}),
            "risk_analysis": {
                "overall_risk": self.aggregator.compute_overall_risk(),
                "high_risk_clauses": len([c for c in clauses if (c.get("risk") if isinstance(c, dict) else "") in ["High", "Critical"]]),
                "medium_risk_clauses": len([c for c in clauses if (c.get("risk") if isinstance(c, dict) else "") == "Medium"]),
                "low_risk_clauses": len([c for c in clauses if (c.get("risk") if isinstance(c, dict) else "") == "Low"]),
            },
            "entities": {
                "names": entities.get("names", []),
                "dates": entities.get("dates", []),
                "amounts": entities.get("amounts", entities.get("money", [])),
                "locations": entities.get("locations", []),
                "organizations": entities.get("organizations", []),
            },
            "timeline": self.aggregator.results.get("timeline", []),
            "compliance": {
                "status": compliance_data.get("risk_level", "Unknown"),
                "violations": compliance_data.get("violations", []),
            },
            "summary": self.aggregator.results.get("summarization", {}).get("summary", "") if isinstance(self.aggregator.results.get("summarization"), dict) else "",
            "comparison": {
                "overall_change_score": self.aggregator.results.get("comparison", {}).get("overall_change_score", 0.0)
                if isinstance(self.aggregator.results.get("comparison"), dict) else 0.0
            },
        }

        path = report_service.generate_pdf_report(report_data)
        self._save_module_result("report_generation", path, time.time() - start)
        return path

    async def execute_pipeline(
        self,
        file_path: str,
        compare_path: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._reset_runtime_state()
        start_time = time.time()
        logger.info(f"Pipeline started for: {file_path}")

        status = "success"

        try:
            ocr_data = await self._run_module("ingestion", self.run_ingestion(file_path))
            full_text = ocr_data.get("text", "") if isinstance(ocr_data, dict) else ""
            if not full_text.strip():
                self.aggregator.add_warning("Empty document text extracted.")

            translated_text = await self._run_module("translation", self.run_translation(full_text))

            await self._run_module("authenticity", self.run_authenticity(translated_text, file_path))
            await self._run_module("classification", self.run_classification(translated_text))
            await self._run_module("ner", self.run_ner(translated_text))
            await self._run_module("risk_analysis", self.run_risk_analysis(translated_text))
            await self._run_module("summarization", self.run_summarization(translated_text))
            await self._run_module("timeline", self.run_timeline(translated_text))
            await self._run_module("clause_checker", self.run_clause_checker(translated_text))
            await self._run_module("compliance", self.run_compliance(translated_text))

            if self.config.enable_comparison:
                await self._run_module("comparison", self.run_comparison(translated_text, compare_path), optional=True)

            await self._run_module("rag_index", self.run_rag_index(translated_text, query), optional=True)
            await self._run_module("report_generation", self.run_report_generation())

            if self.aggregator.errors or self.aggregator.warnings:
                status = "warning"

        except Exception as exc:
            logger.error(f"Pipeline crashed: {exc}")
            self.aggregator.add_error(f"Fatal Pipeline Error: {str(exc)}")
            self.aggregator.add_error(traceback.format_exc())
            status = "failed"

        health_report = self.validator.validate(
            module_outputs=self.module_outputs,
            config=self.config,
            warnings=self.aggregator.warnings,
            errors=self.aggregator.errors,
            auto_fixes_applied=self.auto_fixes_applied,
        )

        if health_report["pipeline_health"] == "broken" and status != "failed":
            status = "failed"
        elif health_report["pipeline_health"] == "warning" and status == "success":
            status = "warning"

        final_output = self.aggregator.build_final_json(status, start_time)
        final_output["pipeline_health_report"] = health_report

        logger.info(
            "Pipeline completed with status: %s in %ss",
            status,
            final_output["processing_time_sec"],
        )
        return final_output


# --- EXAMPLE USAGE ---
async def main():
    config = PipelineConfig(enable_comparison=True, debug_mode=True)
    pipeline = LegalAIPipeline(config)
    results = await pipeline.execute_pipeline(
        file_path="d:\\Legal-mohan\\Legaldoc-new\\LegalDOCAI\\data\\sample.pdf",
        compare_path="d:\\Legal-mohan\\Legaldoc-new\\LegalDOCAI\\data\\sample_v2.pdf",
    )
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
