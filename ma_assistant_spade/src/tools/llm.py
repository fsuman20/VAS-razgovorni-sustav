from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

#Ukredano iz vlastitog zavrsnog rada dostuponog na foi radovi + prilagodba uz Github Copilota

@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 800


class LLMClient:
    """Omotač za OpenAI Responses API s minimalnim ponovnim pokušajima (retry/backoff)."""

    def __init__(self, config: LLMConfig, client: Optional[OpenAI] = None):
        self.config = config
        self.client = client or OpenAI()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Sinkroni dovršetak (completion). Koristi u dretvama ako je potrebno."""
        # Jednostavni retry/backoff za prolazne greške
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                )
                return getattr(resp, "output_text", "") or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.6 * (2 ** attempt))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")
