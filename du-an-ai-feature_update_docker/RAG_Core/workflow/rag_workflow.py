# RAG_Core/workflow/rag_workflow.py  (FIXED – token_usage từ tất cả agent nodes)
"""
Fix: mọi node gọi LLM đều capture token_usage vào state.
     _chatter_node, _other_node, _reporter_node, _not_enough_info_node
     đọc result["token_usage"] và cộng vào state["token_usage"].
"""

from typing import Dict, Any, List, AsyncIterator, Optional
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging, asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from agents.supervisor        import SupervisorAgent
from agents.faq_agent         import FAQAgent
from agents.retriever_agent   import RetrieverAgent
from agents.grader_agent      import GraderAgent
from agents.generator_agent   import GeneratorAgent
from agents.reporter_agent    import ReporterAgent
from agents.chatter_agent     import ChatterAgent
from agents.other_agent       import OtherAgent
from agents.not_enough_info_agent import NotEnoughInfoAgent
from agents.hello_agent       import HelloAgent
from services.document_url_service import document_url_service

logger = logging.getLogger(__name__)


class ChatbotState(TypedDict):
    question:                  str
    original_question:         str
    history:                   List[Dict[str, str]]
    is_followup:               bool
    context_summary:           str
    relevant_context:          str
    current_agent:             str
    documents:                 List[Dict[str, Any]]
    qualified_documents:       List[Dict[str, Any]]
    references:                List[Dict[str, Any]]
    answer:                    str
    status:                    str
    iteration_count:           int
    supervisor_classification: Dict[str, Any]
    faq_result:                Dict[str, Any]
    retriever_result:          Dict[str, Any]
    parallel_mode:             bool
    streaming_mode:            bool
    user_id:                   Optional[str]
    token_usage:               Dict[str, Any]


class RAGWorkflow:
    def __init__(self):
        self.supervisor            = SupervisorAgent()
        self.faq_agent             = FAQAgent()
        self.retriever_agent       = RetrieverAgent()
        self.grader_agent          = GraderAgent()
        self.generator_agent       = GeneratorAgent()
        self.reporter_agent        = ReporterAgent()
        self.chatter_agent         = ChatterAgent()
        self.other_agent           = OtherAgent()
        self.not_enough_info_agent = NotEnoughInfoAgent()
        self.hello_agent           = HelloAgent()

        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RAG-Worker")
        self.workflow = self._create_workflow()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _add_tokens(self, state: ChatbotState, new_tokens: int):
        """Cộng dồn token vào state an toàn."""
        current = (state.get("token_usage") or {}).get("total_tokens", 0) or 0
        state["token_usage"] = {"total_tokens": current + new_tokens}

    def _enrich_references_with_urls(self, references):
        try:
            return document_url_service.enrich_references_with_urls(references) if references else []
        except Exception as e:
            logger.error(f"Enrich refs error: {e}")
            return references

    # ── workflow graph ────────────────────────────────────────────────────────

    def _create_workflow(self):
        wf = StateGraph(ChatbotState)
        wf.add_node("parallel_execution", self._parallel_execution_node)
        wf.add_node("decision_router",    self._decision_router_node)
        wf.add_node("grader",             self._grader_node)
        wf.add_node("generator",          self._generator_node)
        wf.add_node("not_enough_info",    self._not_enough_info_node)
        wf.add_node("chatter",            self._chatter_node)
        wf.add_node("reporter",           self._reporter_node)
        wf.add_node("other",              self._other_node)
        wf.add_node("hello",              self._hello_node)

        wf.set_entry_point("parallel_execution")
        wf.add_edge("parallel_execution", "decision_router")
        wf.add_conditional_edges(
            "decision_router", self._route_after_decision,
            {"GRADER": "grader", "CHATTER": "chatter", "REPORTER": "reporter",
             "OTHER": "other", "HELLO": "hello", "end": "__end__"},
        )
        wf.add_conditional_edges(
            "grader", self._route_next_agent,
            {"GENERATOR": "generator", "NOT_ENOUGH_INFO": "not_enough_info"},
        )
        for node in ["generator", "not_enough_info", "chatter", "reporter", "other", "hello"]:
            wf.add_edge(node, "__end__")
        return wf.compile()

    # ── parallel execution ────────────────────────────────────────────────────

    def _parallel_execution_node(self, state: ChatbotState) -> ChatbotState:
        question  = state["question"]
        history   = state.get("history", [])
        skip_faq  = state.get("streaming_mode", False)
        user_id   = state.get("user_id")

        logger.info(f"🚀 Parallel execution (user_id={user_id})")

        supervisor_result = self._get_result_with_timeout(
            self.executor.submit(self._safe_execute_supervisor, question, history),
            timeout=120,
            default={"agent": "FAQ", "contextualized_question": question,
                     "is_followup": False, "context_summary": ""},
            name="Supervisor",
        )

        # Lấy tokens của supervisor (non-streaming invoke)
        supervisor_tokens = supervisor_result.pop("token_usage", 0) or 0
        logger.info(f"💰 Supervisor tokens: {supervisor_tokens}")
        self._add_tokens(state, supervisor_tokens)

        context_summary         = supervisor_result.get("context_summary", "")
        is_followup             = supervisor_result.get("is_followup", False)
        contextualized_question = supervisor_result.get("contextualized_question", question)

        if skip_faq:
            faq_result = {"status": "SKIPPED", "answer": "", "references": []}
        else:
            faq_result = self._get_result_with_timeout(
                self.executor.submit(self._safe_execute_faq, contextualized_question, is_followup, context_summary),
                timeout=10,
                default={"status": "ERROR", "answer": "", "references": []},
                name="FAQ",
            )
            faq_tokens = faq_result.pop("token_usage", 0) or 0
            if faq_tokens:
                logger.info(f"💰 FAQ tokens: {faq_tokens}")
                self._add_tokens(state, faq_tokens)
            if faq_result.get("references"):
                faq_result["references"] = self._enrich_references_with_urls(faq_result["references"])

        retriever_result = self._get_result_with_timeout(
            self.executor.submit(
                self._safe_execute_retriever,
                question, contextualized_question, is_followup, user_id,
            ),
            timeout=10,
            default={"status": "ERROR", "documents": []},
            name="RETRIEVER",
        )

        state.update({
            "supervisor_classification": supervisor_result,
            "question":                  contextualized_question,
            "original_question":         question,
            "is_followup":               is_followup,
            "context_summary":           context_summary,
            "faq_result":                faq_result,
            "retriever_result":          retriever_result,
            "parallel_mode":             True,
        })
        return state

    def _get_result_with_timeout(self, future, timeout, default, name):
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(f"⏱️ {name} timeout")
            return default
        except Exception as e:
            logger.error(f"❌ {name} error: {e}")
            return default

    def _safe_execute_supervisor(self, question, history):
        try:
            result = self.supervisor.classify_request(question, history)
            # Supervisor gọi LLM qua invoke → lấy tokens nếu có
            # (SupervisorAgent hiện dùng llm_model.invoke, cần update nó cũng dùng invoke_with_usage)
            return result
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            return {"agent": "FAQ", "contextualized_question": question,
                    "context_summary": "", "is_followup": False, "token_usage": 0}

    def _safe_execute_faq(self, question, is_followup=False, context_summary=""):
        try:
            return self.faq_agent.process(question=question, is_followup=is_followup, context=context_summary)
        except Exception as e:
            logger.error(f"FAQ error: {e}")
            return {"status": "ERROR", "answer": "", "references": [], "next_agent": "RETRIEVER", "token_usage": 0}

    def _safe_execute_retriever(self, original_q, contextualized_q, is_followup=False, user_id=None):
        try:
            return self.retriever_agent.process(
                question=original_q, contextualized_question=contextualized_q,
                is_followup=is_followup, user_id=user_id,
            )
        except Exception as e:
            logger.error(f"RETRIEVER error: {e}")
            return {"status": "ERROR", "documents": [], "next_agent": "NOT_ENOUGH_INFO"}

    # ── nodes ─────────────────────────────────────────────────────────────────

    def _decision_router_node(self, state):
        supervisor_agent = state.get("supervisor_classification", {}).get("agent", "FAQ")
        faq_result       = state.get("faq_result", {})
        retriever_result = state.get("retriever_result", {})

        if supervisor_agent in ["CHATTER", "REPORTER", "OTHER", "HELLO"]:
            state["current_agent"] = supervisor_agent
            return state

        if faq_result.get("status") == "SUCCESS":
            faq_tokens = faq_result.pop("token_usage", 0) or 0
            if faq_tokens:
                self._add_tokens(state, faq_tokens)
            state.update({
                "status":        faq_result["status"],
                "answer":        faq_result.get("answer", ""),
                "references":    faq_result.get("references", []),
                "current_agent": "end",
            })
            return state

        if retriever_result.get("documents"):
            state.update({
                "documents":     retriever_result.get("documents", []),
                "status":        retriever_result.get("status", "SUCCESS"),
                "current_agent": "GRADER",
            })
            return state

        state["current_agent"] = "NOT_ENOUGH_INFO"
        return state

    def _grader_node(self, state):
        try:
            import inspect
            sig    = inspect.signature(self.grader_agent.process)
            kwargs = dict(
                question=state.get("original_question", state["question"]),
                documents=state.get("documents", []),
                is_followup=state.get("is_followup", False),
            )
            if "contextualized_question" in sig.parameters:
                kwargs["contextualized_question"] = state["question"]
            result = self.grader_agent.process(**kwargs)
            if result.get("references"):
                result["references"] = self._enrich_references_with_urls(result["references"])
            state.update({
                "status":              result["status"],
                "qualified_documents": result.get("qualified_documents", []),
                "references":          result.get("references", []),
                "current_agent":       result.get("next_agent", "GENERATOR"),
            })
        except Exception as e:
            logger.error(f"Grader error: {e}")
            state["current_agent"] = "NOT_ENOUGH_INFO"
        return state

    def _generator_node(self, state):
        """Generator — capture token_usage từ invoke_with_usage."""
        try:
            result = self.generator_agent.process(
                question=state["question"],
                documents=state.get("qualified_documents", []),
                references=state.get("references", []),
                history=state.get("history", []),
                is_followup=state.get("is_followup", False),
                context_summary=state.get("context_summary", ""),
            )
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            logger.info(f"💰 Generator tokens: {tokens}")
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"Generator error: {e}")
            state.update({"answer": "Lỗi tạo câu trả lời", "current_agent": "end"})
        return state

    def _not_enough_info_node(self, state):
        try:
            result = self.not_enough_info_agent.process(state["question"])
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            logger.info(f"💰 NotEnoughInfo tokens: {tokens}")
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"NotEnoughInfo error: {e}")
            state["answer"] = "Không tìm thấy thông tin"
        return state

    def _chatter_node(self, state):
        try:
            result = self.chatter_agent.process(state["question"], state.get("history", []))
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            logger.info(f"💰 Chatter tokens: {tokens}")
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"Chatter error: {e}")
            state["answer"] = "Tôi hiểu cảm xúc của bạn"
        return state

    def _reporter_node(self, state):
        try:
            result = self.reporter_agent.process(state["question"])
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"Reporter error: {e}")
            state["answer"] = "Hệ thống đang bảo trì"
        return state

    def _hello_node(self, state):
        try:
            result = self.hello_agent.process(state["question"])
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            logger.info(f"💰 Hello tokens: {tokens}")
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"Hello error: {e}")
            state["answer"] = "Xin chào! Tôi là Tori của Techcomlife!"
        return state

    def _other_node(self, state):
        try:
            result = self.other_agent.process(state["question"])
            tokens = result.get("token_usage", 0) or 0
            self._add_tokens(state, tokens)
            logger.info(f"💰 Other tokens: {tokens}")
            state.update({
                "status":        result["status"],
                "answer":        result.get("answer", ""),
                "references":    result.get("references", []),
                "current_agent": "end",
            })
        except Exception as e:
            logger.error(f"Other error: {e}")
            state["answer"] = "Đây không phải tác vụ của tôi"
        return state

    def _route_after_decision(self, state): return state.get("current_agent", "end")
    def _route_next_agent(self, state):     return state.get("current_agent", "end")

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self, question: str,
        history:  List[Dict[str, str]] = None,
        user_id:  Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            initial = self._create_initial_state(question, history, streaming_mode=False, user_id=user_id)
            final   = self.workflow.invoke(initial)
            total   = (final.get("token_usage") or {}).get("total_tokens", 0)
            logger.info(f"💰 Total tokens (non-streaming): {total}")
            return {
                "answer":      final.get("answer", "Lỗi xử lý"),
                "references":  final.get("references", []),
                "status":      final.get("status", "ERROR"),
                "token_usage": {"total_tokens": total},
            }
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {"answer": "Xin lỗi, hệ thống gặp sự cố.", "references": [],
                    "status": "ERROR", "token_usage": {"total_tokens": 0}}

    async def run_with_streaming(
        self, question: str,
        history:  List[Dict[str, str]] = None,
        user_id:  Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Streaming workflow.
        Supervisor tokens được inject vào sentinel cuối stream.
        """
        try:
            state = self._create_initial_state(question, history, streaming_mode=True, user_id=user_id)
            state = self._parallel_execution_node(state)
            state = self._decision_router_node(state)

            current_agent    = state.get("current_agent")
            supervisor_agent = state.get("supervisor_classification", {}).get("agent")
            # Tokens đã tích lũy từ supervisor + faq (parallel phase)
            base_tokens = (state.get("token_usage") or {}).get("total_tokens", 0)

            # ── HELLO — xử lý ngay, không qua FAQ/GRADER ─────────────────────
            if supervisor_agent == "HELLO":
                return {
                    "answer_stream": self._inject_base_tokens(
                        self.hello_agent.process_streaming(question=state["question"]),
                        base_tokens,
                    ),
                    "references": [],
                    "status": "STREAMING",
                }

            # ── FAQ streaming ─────────────────────────────────────────────────
            if supervisor_agent == "FAQ":
                from tools.vector_search import search_faq, rerank_faq
                from config.settings import settings as cfg

                faq_results = search_faq.invoke({"query": state["question"]})
                if not faq_results or "error" in str(faq_results):
                    current_agent = "GRADER"
                else:
                    filtered = [f for f in faq_results
                                if f.get("similarity_score", 0) >= cfg.FAQ_VECTOR_THRESHOLD]
                    if not filtered:
                        current_agent = "GRADER"
                    else:
                        reranked = rerank_faq.invoke({"query": state["question"], "faq_results": filtered})
                        if not reranked:
                            current_agent = "GRADER"
                        elif reranked[0].get("rerank_score", 0) >= cfg.FAQ_RERANK_THRESHOLD:
                            return {
                                "answer_stream": self._inject_base_tokens(
                                    self.faq_agent.process_streaming(
                                        question=state["question"],
                                        reranked_faqs=reranked,
                                        is_followup=state.get("is_followup", False),
                                        context=state.get("context_summary", ""),
                                    ),
                                    base_tokens,
                                ),
                                "references": [
                                    {"document_id": reranked[0].get("faq_id"), "type": "FAQ",
                                     "description": reranked[0].get("question", ""),
                                     "rerank_score": round(reranked[0].get("rerank_score", 0), 4)},
                                ],
                                "status": "STREAMING",
                            }
                        else:
                            current_agent = "GRADER"

            # ── GRADER → GENERATOR / NOT_ENOUGH_INFO ─────────────────────────
            if current_agent == "GRADER":
                state = self._grader_node(state)
                from config.settings import settings as cfg

                if state.get("current_agent") == "GENERATOR":
                    return {
                        "answer_stream": self._inject_base_tokens(
                            self.generator_agent.process_streaming(
                                question=state["question"],
                                documents=state.get("qualified_documents", []),
                                references=state.get("references", []),
                                history=history or [],
                                is_followup=state.get("is_followup", False),
                                context_summary=state.get("context_summary", ""),
                            ),
                            base_tokens,
                        ),
                        "references": state.get("references", []),
                        "status":     "STREAMING",
                    }
                else:
                    return {
                        "answer_stream": self._inject_base_tokens(
                            self.not_enough_info_agent.process_streaming(
                                question=state["question"],
                                support_phone=cfg.SUPPORT_PHONE,
                            ),
                            base_tokens,
                        ),
                        "references": [],
                        "status":     "STREAMING",
                    }

            # ── HELLO ─────────────────────────────────────────────────────────
            if current_agent == "HELLO":
                return {
                    "answer_stream": self._inject_base_tokens(
                        self.hello_agent.process_streaming(
                            question=state["question"],
                        ),
                        base_tokens,
                    ),
                    "references": [],
                    "status": "STREAMING",
                }

            # ── CHATTER ───────────────────────────────────────────────────────
            if current_agent == "CHATTER":
                return {
                    "answer_stream": self._inject_base_tokens(
                        self.chatter_agent.process_streaming(
                            question=state["question"],
                            history=state.get("history", []),
                            support_phone=cfg.SUPPORT_PHONE,
                        ),
                        base_tokens,
                    ),
                    "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                    "status": "STREAMING",
                }

            # ── OTHER ─────────────────────────────────────────────────────────
            if current_agent == "OTHER":
                return {
                    "answer_stream": self._inject_base_tokens(
                        self.other_agent.process_streaming(
                            question=state["question"],
                            support_phone=cfg.SUPPORT_PHONE,
                        ),
                        base_tokens,
                    ),
                    "references": [],
                    "status": "STREAMING",
                }

            # ── REPORTER (không stream) ───────────────────────────────────────
            if current_agent == "REPORTER":
                state = self._reporter_node(state)
                answer_text = state.get("answer", "")

                async def reporter_gen():
                    for word in answer_text.split():
                        yield word + " "
                        await asyncio.sleep(0.01)
                    yield {"__token_usage__": base_tokens}  # reporter không dùng LLM

                return {
                    "answer_stream": reporter_gen(),
                    "references":    state.get("references", []),
                    "status":        state.get("status", "SUCCESS"),
                }

            async def error_gen():
                yield "Xin lỗi, không thể xử lý yêu cầu này."
                yield {"__token_usage__": 0}

            return {"answer_stream": error_gen(), "references": [], "status": "ERROR"}

        except Exception as e:
            logger.error(f"Streaming workflow error: {e}", exc_info=True)

            async def error_gen():
                yield "Xin lỗi, hệ thống gặp sự cố."
                yield {"__token_usage__": 0}

            return {"answer_stream": error_gen(), "references": [], "status": "ERROR"}

    async def _inject_base_tokens(self, stream, base_tokens: int):
        """
        Wrap một async generator: forward tất cả items,
        khi gặp sentinel {"__token_usage__": N} thì cộng thêm base_tokens
        (supervisor + faq từ parallel phase).
        """
        async for item in stream:
            if isinstance(item, dict) and "__token_usage__" in item:
                agent_tokens = item["__token_usage__"]
                total = base_tokens + agent_tokens
                logger.info(
                    f"💰 inject_supervisor_tokens: supervisor+faq={base_tokens} "
                    f"agent={agent_tokens} total={total}"
                )
                yield {"__token_usage__": total}
            else:
                yield item

    # ── helpers ───────────────────────────────────────────────────────────────

    def _create_initial_state(
        self, question, history=None, streaming_mode=False, user_id=None,
    ) -> ChatbotState:
        return ChatbotState(
            question=question, original_question=question,
            history=history or [], is_followup=False, context_summary="",
            relevant_context="", current_agent="parallel_execution",
            documents=[], qualified_documents=[], references=[],
            answer="", status="", iteration_count=0,
            supervisor_classification={}, faq_result={}, retriever_result={},
            parallel_mode=False, streaming_mode=streaming_mode,
            user_id=user_id, token_usage={"total_tokens": 0},
        )

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True, timeout=5)