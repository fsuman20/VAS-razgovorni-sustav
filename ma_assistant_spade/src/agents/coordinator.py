#Napravljeno uz pomoc chatGPT-a - https://chatgpt.com/s/t_696d59dc1cc48191ac2261d8f3c73db3

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from src.protocol import (
    ResearchRequest,
    ResearchResult,
    VerifyRequest,
    VerifyResult,
    make_metadata,
    new_conversation_id,
)
from src.tools.llm import LLMClient, LLMConfig
from src.tools.logging_utils import log_msg

#Promptovi su Ai generirani uz pomoc Github Copilota

COORDINATOR_PLAN_PROMPT = """Ti si Koordinator u višeagentnom razgovornom asistentu. 

Zadatak: korisnikov upit razloži na podzadatke i pripremi istraživački upit za Istraživača.

Vrati isključivo JSON:
{
  "research_query": "...",
  "subtasks": ["...", "..."],
  "notes": "kratko"
}
"""

COORDINATOR_DRAFT_PROMPT = """Ti si Koordinator. Dobio si sažetak istraživanja i dokaze.

Napiši konačan odgovor korisniku:
- na hrvatskom
- jasno, strukturirano
- ne izmišljaj činjenice; koristi samo dokaze
- kada koristiš dokaz, dodaj citat [DOC:CHUNK]
"""

COORDINATOR_REVISION_PROMPT = """Ti si Koordinator. Dobio si nacrt i nalaz Provjeravatelja.

Ispravi nacrt:
- ukloni ili ublaži tvrdnje bez dokaza
- dodaj ograničenja i napomene gdje treba
- zadrži citate [DOC:CHUNK] samo ako se stvarno odnose na dokaze
"""


class CoordinatorAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        researcher_jid: str,
        verifier_jid: str,
        llm_model: str,
        logger,
    ):
        super().__init__(jid, password)
        self.researcher_jid = researcher_jid
        self.verifier_jid = verifier_jid
        self.logger = logger
        self.llm = LLMClient(LLMConfig(model=llm_model, max_output_tokens=900))

        self.history: list[dict[str, str]] = []

        self.user_queue: asyncio.Queue[str] = asyncio.Queue()

    async def setup(self):
        self.add_behaviour(_OrchestratorBehaviour())


class _OrchestratorBehaviour(CyclicBehaviour):
    async def run(self):
        try:
            user_text = self.agent.user_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)
            return

        if not user_text.strip():
            return

        conversation_id = new_conversation_id()
        self.agent.logger.info("conversation_id=%s", conversation_id)

        history_block = "\n".join(
            [f"U: {h['user']}\nA: {h['assistant']}" for h in self.agent.history[-3:]]
        )
        if history_block:
            user_for_plan = f"Povijest razgovora (sažeto):\n{history_block}\n\nNovi upit: {user_text}"
        else:
            user_for_plan = user_text

        # 1) PLANIRANJE
        plan_raw = self.agent.llm.complete(COORDINATOR_PLAN_PROMPT, user_for_plan)
        plan = _safe_json(plan_raw, default={"research_query": user_text, "subtasks": [], "notes": ""})
        research_query = str(plan.get("research_query") or user_text)

        # 2) PITAJ ISTRAŽIVAČA
        research_req = ResearchRequest(query=research_query, top_k=5)
        msg = Message(to=self.agent.researcher_jid)
        msg.metadata = make_metadata("request", conversation_id, {"role": "research"})
        msg.body = research_req.to_json()
        await self.send(msg)
        log_msg(self.agent.logger, "send", str(self.agent.jid), str(self.agent.researcher_jid), dict(msg.metadata), msg.body)

        research_res = await _wait_for_role(self, conversation_id, want_role="research_result", timeout=30)
        if research_res is None:
            print("[GREŠKA] Isteklo vrijeme za Istraživača.")
            return
        rr = ResearchResult.from_json(research_res.body or "{}")

        # 3) NACRT ODGOVORA
        evidence_block = "\n".join([f"- [{e['doc_id']}:{e['chunk_id']}] {e['text']}" for e in rr.evidence])
        draft_prompt = (
            f"Korisnikov upit: {user_text}\n\n"
            + (f"Kontekst (zadnja 3 turna):\n{history_block}\n\n" if history_block else "")
            + f"Plan/subtasks: {json.dumps(plan, ensure_ascii=False)}\n\n"
            + f"Sažetak istraživanja: {rr.summary}\n\n"
            + f"Dokazi:\n{evidence_block}\n\n"
            + "Napiši konačan odgovor."
        )
        draft_answer = self.agent.llm.complete(COORDINATOR_DRAFT_PROMPT, draft_prompt).strip()

        # 4) PROVJERA
        verify_req = VerifyRequest(draft_answer=draft_answer, evidence=rr.evidence)
        vmsg = Message(to=self.agent.verifier_jid)
        vmsg.metadata = make_metadata("request", conversation_id, {"role": "verify"})
        vmsg.body = verify_req.to_json()
        await self.send(vmsg)
        log_msg(self.agent.logger, "send", str(self.agent.jid), str(self.agent.verifier_jid), dict(vmsg.metadata), vmsg.body)

        verify_res = await _wait_for_role(self, conversation_id, want_role="verify_result", timeout=30)
        final_answer = draft_answer

        if verify_res is not None:
            vr = VerifyResult.from_json(verify_res.body or "{}")
            if vr.verdict in {"WARN", "FAIL"}:
                revision_prompt = (
                    f"UPIT: {user_text}\n\n"
                    f"NACRT:\n{draft_answer}\n\n"
                    f"PROVJERA (verdict={vr.verdict}):\n"
                    f"- issues: {vr.issues}\n"
                    f"- suggested_fixes: {vr.suggested_fixes}\n\n"
                    "Ispravi odgovor." 
                )
                final_answer = self.agent.llm.complete(COORDINATOR_REVISION_PROMPT, revision_prompt).strip()

            # Prikaz presude provjeravatelja
            print(f"\n[Provjeravatelj: {vr.verdict}] {(' | '.join(vr.issues[:3])) if vr.issues else ''}\n")

        # 5) ISPIS
        print("\n=== ODGOVOR ===\n")
        print(final_answer)
        print("\n==============\n")

        self.agent.history.append({"user": user_text, "assistant": final_answer})
        if len(self.agent.history) > 10:
            self.agent.history = self.agent.history[-10:]


async def _wait_for_role(behaviour: CyclicBehaviour, conversation_id: str, want_role: str, timeout: int = 20) -> Optional[Message]:
    """Čekaj dolaznu poruku s odgovarajućim ID-jem razgovora i ulogom."""
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        msg = await behaviour.receive(timeout=1)
        if not msg:
            continue
        md = dict(msg.metadata)
        if md.get("conversation-id") == conversation_id and md.get("role") == want_role:
            log_msg(behaviour.agent.logger, "recv", str(msg.sender), str(behaviour.agent.jid), md, msg.body or "")
            return msg
        # Inače: ignoriraj (ali zabilježi)
        log_msg(behaviour.agent.logger, "recv", str(msg.sender), str(behaviour.agent.jid), md, msg.body or "")
    return None


def _safe_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return default
