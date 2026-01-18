from __future__ import annotations

import json
from typing import Any, Dict, List

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from src.protocol import ResearchRequest, ResearchResult, make_metadata, ONTOLOGY
from src.tools.corpus_search import CorpusIndex
from src.tools.llm import LLMClient, LLMConfig
from src.tools.logging_utils import log_msg

#Promptovi su Ai generirani uz pomoc Github Copilota

RESEARCHER_SYSTEM_PROMPT = """Ti si Istraživač u višeagentnom razgovornom asistentu.

Zadatak: na temelju dostavljenih *dokaza* (iz mini-korpusa) napravi kratki sažetak relevantnih činjenica.

Pravila:
- Ne izmišljaj činjenice koje nisu u dokazima.
- Ako su dokazi slabi ili ne pokrivaju upit, jasno to reci.
- Kada se pozivaš na dokaz, dodaj citat u obliku [DOC:CHUNK] (npr. [paper1:3]).
- Sažetak neka bude na hrvatskom.
"""


class ResearcherAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        corpus_dir: str,
        top_k: int,
        llm_model: str,
        logger,
    ):
        super().__init__(jid, password)
        self.corpus_dir = corpus_dir
        self.top_k = top_k
        self.logger = logger
        self.llm = LLMClient(LLMConfig(model=llm_model, max_output_tokens=600))
        self.index = CorpusIndex(corpus_dir)

    async def setup(self):
        self.index.build()

        template = Template()
        template.set_metadata("ontology", ONTOLOGY)
        template.set_metadata("role", "research")

        self.add_behaviour(_ResearchBehaviour(), template)


class _ResearchBehaviour(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        log_msg(self.agent.logger, "recv", str(msg.sender), str(self.agent.jid), dict(msg.metadata), msg.body or "")

        # Parsiraj zahtjev
        try:
            d = json.loads(msg.body or "{}")
            req = ResearchRequest(query=str(d.get("query", "")), top_k=int(d.get("top_k", self.agent.top_k)))
        except Exception:  # noqa: BLE001
            req = ResearchRequest(query=(msg.body or ""), top_k=self.agent.top_k)

        results = self.agent.index.search(req.query, top_k=req.top_k)

        evidence: List[Dict[str, Any]] = []
        for chunk, score in results:
            evidence.append(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "score": round(score, 4),
                    "text": chunk.text[:600],
                }
            )

        # Zatraži od LLM-a sažetak temeljen na dokazima
        evidence_block = "\n".join(
            [f"- [{e['doc_id']}:{e['chunk_id']}] {e['text']}" for e in evidence]
        )
        user_prompt = (
            f"Upit korisnika: {req.query}\n\n"
            f"Dokazi (mini-korpus):\n{evidence_block}\n\n"
            "Napiši sažetak (5-10 rečenica) koji odgovara na upit, koristeći samo dokaze."
        )
        summary = self.agent.llm.complete(RESEARCHER_SYSTEM_PROMPT, user_prompt).strip()

        out = ResearchResult(evidence=evidence, summary=summary)

        reply = Message(to=str(msg.sender))
        reply.metadata = make_metadata("inform", msg.metadata.get("conversation-id", ""), {"role": "research_result"})
        reply.body = out.to_json()

        await self.send(reply)
        log_msg(self.agent.logger, "send", str(self.agent.jid), str(msg.sender), dict(reply.metadata), reply.body)
