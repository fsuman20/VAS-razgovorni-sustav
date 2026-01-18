# Višeagentni asistent (SPADE)

Ovaj projekt pokreće višeagentni asistent (Koordinator, Istraživač, Provjeravatelj) koji koristi lokalni tekstualni korpus i OpenAI LLM za generiranje odgovora. Komunikacija među agentima ide preko XMPP (SPADE).

## Preduvjeti

- Python 3.10+ (preporučeno 3.11/3.12)
- Pokrenut XMPP poslužitelj (npr. Prosody ili ejabberd) na localhost
- OpenAI API ključ

## Instalacija

1. (Preporučeno) Koristiti VAS virtualno okruženje:

   - Linux/macOS:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Instaliraj ovisnosti:

   - `pip install spade aioconsole python-dotenv openai scikit-learn`

## Konfiguracija (.env)

U korijenu projekta postoji .env. Postavi ili ažuriraj varijable:

- `OPENAI_API_KEY` – obavezno
- `OPENAI_MODEL` – npr. `gpt-4o-mini`
- `COORD_JID`, `RESEARCHER_JID`, `VERIFIER_JID` – XMPP korisnici
- `COORD_PASSWORD`, `RESEARCHER_PASSWORD`, `VERIFIER_PASSWORD`
- `AUTO_REGISTER=true` ako želiš da SPADE automatski registrira korisnike
- `CORPUS_DIR=./data/corpus` – mapa s .txt izvorima
- `TOP_K=5` – broj najrelevantnijih chunkova
- `LOG_DIR=./logs`

Napomena: Ne dijeli .env s API ključem.

## Pokretanje

U korijenu projekta pokreni:

- `python -m src.main`

Ako je XMPP poslužitelj aktivan i vjerodajnice ispravne, aplikacija će ispisati prompt:

- `Višeagentni asistent pokrenut. Unesite pitanje (ili 'izlaz', 'kraj').`

## Korištenje

1. Upiši pitanje nakon `Ti>` i pritisni Enter.
2. Asistent će vratiti odgovor i prikazati presudu provjeravatelja.
3. Za izlaz upiši `izlaz` ili `kraj`.

## Dodavanje izvora (korpus)

Stavi .txt datoteke u `data/corpus/`. Svaki dokument se dijeli u chunkove i indeksira TF‑IDF modelom. Ako nema izvora, sustav kreira placeholder datoteku.

## Bilješke

- Aplikacija koristi OpenAI Responses API.
- Ako koristiš lokalni XMPP (npr. Prosody), provjeri da su korisnici postojeći ili omogući `AUTO_REGISTER=true`.
- Logovi se spremaju u `./logs`.
