# generirano uz pomoc chatGPT-a - https://chatgpt.com/s/t_696d5be585488191b11e651d5eadb0f1
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


ONTOLOGY = "ma-assistant"
PROTOCOL = "coordination-v1"
LANGUAGE = "hr"
 

def new_conversation_id() -> str:
    return str(uuid.uuid4())


def make_metadata(performative: str, conversation_id: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    md: Dict[str, str] = {
        "performative": performative,
        "ontology": ONTOLOGY,
        "protocol": PROTOCOL,
        "language": LANGUAGE,
        "conversation-id": conversation_id,
    }
    if extra:
        md.update({k: str(v) for k, v in extra.items()})
    return md


@dataclass
class ResearchRequest:
    query: str
    top_k: int = 5

    def to_json(self) -> str:
        return json.dumps({"query": self.query, "top_k": self.top_k}, ensure_ascii=False)


@dataclass
class ResearchResult:
    evidence: list[dict]
    summary: str

    def to_json(self) -> str:
        return json.dumps({"evidence": self.evidence, "summary": self.summary}, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "ResearchResult":
        d = json.loads(s)
        return ResearchResult(evidence=d.get("evidence", []), summary=d.get("summary", ""))


@dataclass
class VerifyRequest:
    draft_answer: str
    evidence: list[dict]

    def to_json(self) -> str:
        return json.dumps({"draft_answer": self.draft_answer, "evidence": self.evidence}, ensure_ascii=False)


@dataclass
class VerifyResult:
    verdict: str  # PASS|WARN|FAIL
    issues: list[str]
    suggested_fixes: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {"verdict": self.verdict, "issues": self.issues, "suggested_fixes": self.suggested_fixes},
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(s: str) -> "VerifyResult":
        d = json.loads(s)
        return VerifyResult(
            verdict=d.get("verdict", "WARN"),
            issues=list(d.get("issues", [])),
            suggested_fixes=list(d.get("suggested_fixes", [])),
        )
