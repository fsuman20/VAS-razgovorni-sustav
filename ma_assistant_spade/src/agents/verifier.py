from __future__ import annotations

import json
from typing import Any, Dict, List

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from src.protocol import VerifyRequest, VerifyResult, make_metadata, ONTOLOGY
from src.tools.llm import LLMClient, LLMConfig
from src.tools.logging_utils import log_msg


VERIFIER_SYSTEM_PROMPT = """Ti si Provjeravatelj (verifier) u višeagentnom razgovornom asistentu.

Cilj: pronađi nekonzistentnosti, halucinacije i tvrdnje bez dokaza.

Upute:
- Ulaz dobivaš: nacrt odgovora i listu dokaza (iz mini-korpusa).
- Izvuci ključne tvrdnje iz nacrta i provjeri jesu li potkrijepljene dokazima.
- Ako nešto nije potkrijepljeno, označi to kao problem.
- Vrati strukturirano:
  - verdict: PASS (dobro potkrijepljeno), WARN (manji problemi), FAIL (više ključnih problema)
  - issues: lista problema
  - suggested_fixes: kako ispraviti / što tražiti dodatno
- Ne dodaj nove činjenice.
"""


class VerifierAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        llm_model: str,
        logger,
    ):
        super().__init__(jid, password)
        self.logger = logger
        self.llm = LLMClient(LLMConfig(model=llm_model, max_output_tokens=700))

    async def setup(self):
        template = Template()
        template.set_metadata("ontology", ONTOLOGY)
        template.set_metadata("role", "verify")
        self.add_behaviour(_VerifyBehaviour(), template)


class _VerifyBehaviour(CyclicBehaviour): #Github copilott otklonio probleme potvrdivanja
    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        log_msg(self.agent.logger, "recv", str(msg.sender), str(self.agent.jid), dict(msg.metadata), msg.body or "")

        # Parsiraj zahtjev
        try:
            d = json.loads(msg.body or "{}")
            req = VerifyRequest(draft_answer=str(d.get("draft_answer", "")), evidence=list(d.get("evidence", [])))
        except Exception:  # noqa: BLE001
            req = VerifyRequest(draft_answer=(msg.body or ""), evidence=[])

        evidence_block = "\n".join(
            [f"- [{e.get('doc_id')}:{e.get('chunk_id')}] {str(e.get('text',''))[:600]}" for e in req.evidence]
        )

        user_prompt = (
            "NACRT ODGOVORA:\n"
            f"{req.draft_answer}\n\n"
            "DOKAZI:\n"
            f"{evidence_block}\n\n"
            "Vrati rezultat kao JSON objekt s ključevima: verdict, issues, suggested_fixes."
        )

        raw = self.agent.llm.complete(VERIFIER_SYSTEM_PROMPT, user_prompt).strip()

        # Pokušaj parsirati JSON iz izlaza modela; inače WARN
        verdict = "WARN"
        issues: List[str] = []
        fixes: List[str] = []
        try:
            j = json.loads(_extract_json(raw))
            verdict = str(j.get("verdict", "WARN")).upper()
            issues = [str(x) for x in j.get("issues", [])]
            fixes = [str(x) for x in j.get("suggested_fixes", [])]
        except Exception:  # noqa: BLE001
            issues = ["Nije moguće parsirati JSON iz provjere; pogledaj 'raw' u logu."]
            fixes = ["U promptu zatraži striktan JSON output."]

        out = VerifyResult(verdict=verdict, issues=issues, suggested_fixes=fixes)

        reply = Message(to=str(msg.sender))
        reply.metadata = make_metadata("inform", msg.metadata.get("conversation-id", ""), {"role": "verify_result"})
        reply.body = out.to_json()

        await self.send(reply)
        log_msg(self.agent.logger, "send", str(self.agent.jid), str(msg.sender), dict(reply.metadata), reply.body)


def _extract_json(text: str) -> str:
    """Izdvoji prvi JSON objekt iz stringa."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
