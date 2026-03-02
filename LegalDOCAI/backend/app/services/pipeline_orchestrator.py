MODULE_ORDER = [
    "Ingest",
    "OCR",
    "Lang",
    "Classify",
    "NER",
    "Clauses",
    "Risk",
    "Explain",
    "Timeline",
    "RAG",
    "Compare",
    "Report",
]

class ModuleStatus:
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"

class PipelineOrchestrator:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.module_status = {f"M{i+1}": ModuleStatus.IDLE for i in range(len(MODULE_ORDER))}
        self.module_result = {f"M{i+1}": None for i in range(len(MODULE_ORDER))}
        self.module_attempts = {f"M{i+1}": 0 for i in range(len(MODULE_ORDER))}
        self.pipeline_started_at = None
        self.pipeline_completed_at = None

    def start_module(self, module_key: str):
        self.module_status[module_key] = ModuleStatus.RUNNING
        self.module_attempts[module_key] = self.module_attempts.get(module_key, 0) + 1

    def complete_module(self, module_key: str, result=None):
        self.module_status[module_key] = ModuleStatus.SUCCESS
        self.module_result[module_key] = result

    def fail_module(self, module_key: str, error: str = None):
        self.module_status[module_key] = ModuleStatus.ERROR
        if error:
            self.module_result[module_key] = {"error": error}

    def skip_module(self, module_key: str, reason: str = None):
        self.module_status[module_key] = ModuleStatus.SKIPPED
        if reason:
            self.module_result[module_key] = {"reason": reason}

    async def run_with_retry(self, module_key: str, coro, retries: int = 0):
        self.start_module(module_key)
        try:
            res = await coro
            self.complete_module(module_key, res)
            return res
        except Exception as e:
            if retries > 0:
                try:
                    res = await coro
                    self.complete_module(module_key, res)
                    return res
                except Exception as e2:
                    self.fail_module(module_key, str(e2))
                    raise
            self.fail_module(module_key, str(e))
            raise

    def get_system_status(self):
        return {
            "doc_id": self.doc_id,
            "module_order": MODULE_ORDER,
            "module_status": self.module_status,
            "module_result": self.module_result,
            "module_attempts": self.module_attempts,
            "pipeline_started_at": self.pipeline_started_at,
            "pipeline_completed_at": self.pipeline_completed_at,
        }

_active_orchestrators = {}

def get_or_create_orchestrator(doc_id: str) -> PipelineOrchestrator:
    o = _active_orchestrators.get(doc_id)
    if not o:
        o = PipelineOrchestrator(doc_id)
        _active_orchestrators[doc_id] = o
    return o

def get_orchestrator(doc_id: str) -> PipelineOrchestrator:
    return _active_orchestrators.get(doc_id)

def clear_orchestrator(doc_id: str) -> None:
    if doc_id in _active_orchestrators:
        del _active_orchestrators[doc_id]
