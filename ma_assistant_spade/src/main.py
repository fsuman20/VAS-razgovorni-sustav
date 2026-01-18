from __future__ import annotations

import asyncio
import os

from aioconsole import ainput
from dotenv import load_dotenv
from spade import run

from src.agents.coordinator import CoordinatorAgent
from src.agents.researcher import ResearcherAgent
from src.agents.verifier import VerifierAgent
from src.tools.logging_utils import setup_logger


async def main():
    load_dotenv()

    log_dir = os.getenv("LOG_DIR", "./logs")
    logger = setup_logger(log_dir)

    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    coord_jid = os.getenv("COORD_JID", "coordinator@localhost")
    coord_pwd = os.getenv("COORD_PASSWORD", "tajna")

    researcher_jid = os.getenv("RESEARCHER_JID", "researcher@localhost")
    researcher_pwd = os.getenv("RESEARCHER_PASSWORD", "tajna")

    verifier_jid = os.getenv("VERIFIER_JID", "verifier@localhost")
    verifier_pwd = os.getenv("VERIFIER_PASSWORD", "tajna")

    corpus_dir = os.getenv("CORPUS_DIR", "./data/corpus")
    top_k = int(os.getenv("TOP_K", "5"))
    auto_register = os.getenv("AUTO_REGISTER", "false").lower() in {"1", "true", "yes"}

    # Agenti - OPENAI predložak
    researcher = ResearcherAgent(
        researcher_jid,
        researcher_pwd,
        corpus_dir=corpus_dir,
        top_k=top_k,
        llm_model=openai_model,
        logger=logger,
    )
    verifier = VerifierAgent(
        verifier_jid,
        verifier_pwd,
        llm_model=openai_model,
        logger=logger,
    )
    coordinator = CoordinatorAgent(
        coord_jid,
        coord_pwd,
        researcher_jid=researcher_jid,
        verifier_jid=verifier_jid,
        llm_model=openai_model,
        logger=logger,
    )

    # Agenti
    await researcher.start(auto_register=auto_register)
    await verifier.start(auto_register=auto_register)
    await coordinator.start(auto_register=auto_register)

    print("\nVišeagentni asistent pokrenut. Unesite pitanje (ili 'izlaz', 'kraj').\n")

    try:
        while True:
            text = (await ainput("Ti> ")).strip()
            if text.lower() in {"izlaz", "kraj", "exit", "quit"}:
                break
            if not text:
                continue
            coordinator.user_queue.put_nowait(text)
            # Daj koordinatoru vremena da obradi red
            await asyncio.sleep(0.2)
    finally:
        await coordinator.stop()
        await researcher.stop()
        await verifier.stop()
        print("Zaustavljeno.")


if __name__ == "__main__":
    run(main())
