# Fork — Product Requirements Document

## Problem Statement

Readers who love fictional universes have no way to interact with their favorite characters beyond re-reading the source material. Existing AI chatbots offer generic roleplay that quickly breaks canon, fabricates facts, and loses character voice — producing an experience that feels hollow to anyone who actually knows the books. There is no product that lets a reader have a grounded, cited, in-voice conversation with Hermione Granger, explore what would have happened if Harry had been Sorted into Slytherin, or generate a new scene that actually sounds like J.K. Rowling wrote it.

## Solution

Fork ingests a book's full text and automatically builds grounded AI agents of its characters. Every character agent is backed by:

1. A **persona card** — values, speech patterns, vocabulary fingerprint, relationships, and knowledge-by-book — extracted automatically from the corpus via a GPU burst job.
2. A **speaker-attributed, timeline-indexed vector index** — every dialogue line embedded and tagged with `{speaker, book, chapter}` so retrieval is filtered both by character and by what that character could have known at any point in the story.
3. A **grounded generation endpoint** — RAG over the filtered index + persona card → responses in the character's voice, with inline citations to the exact source lines retrieved.

Readers can: chat with characters in-canon, explore short alternate-path scenes at curated decision points, and generate fan fiction in the character's voice. The entire pipeline runs on RunPod Flash serverless GPU infrastructure — async ingestion, burst persona extraction, and a warm real-time serving endpoint.

The demo corpus is the Harry Potter series (all 7 books). Characters supported at launch: Harry Potter, Ron Weasley, Hermione Granger.

## User Stories

### Character Chat
1. As a reader, I want to pick a character from a roster and start a conversation, so that I can interact with them directly.
2. As a reader, I want the character to respond in their authentic voice and vocabulary, so that the experience feels true to the books.
3. As a reader, I want each response to include inline citations showing the exact source passages retrieved, so that I can trust the answer is grounded in canon.
4. As a reader, I want the character to only know what they knew at a given point in the story, so that Hermione in Book 3 doesn't reference events from Book 7.
5. As a reader, I want to ask the character about their opinions, relationships, and decisions, so that I can explore their inner world beyond what's on the page.
6. As a reader, I want the character to stay in-voice even when I ask questions that have no direct canon answer, so that the experience never breaks immersion.
7. As a reader, I want to see the character's persona card (speech style, values, key relationships), so that I understand how the character was modeled.

### Fork Explorer (Alternate Paths)
8. As a reader, I want to browse a set of curated "what if" decision points, so that I can explore alternate versions of the story.
9. As a reader, I want to click a fork point and see an alternate scene generated live, so that I experience the divergence as it unfolds.
10. As a reader, I want the alternate scene to stay in the characters' voices even as it diverges from canon, so that it reads like a plausible alternate universe rather than generic fiction.
11. As a reader, I want the alternate scene to be grounded in canon context up to the fork point, so that the setup feels authentic before the divergence.
12. As a reader, I want the fork generation to clearly label where canon ends and the alternate path begins, so that I'm never confused about what's invented.
13. As a reader, I want to see all four curated fork points available: Slytherin sorting, Harry not returning from the forest, Hermione staying behind, and Draco accepting Dumbledore's offer.

### Fan Fiction Generator
14. As a reader, I want to write a custom prompt describing a scene I want generated, so that I can explore scenarios beyond the pre-curated forks.
15. As a reader, I want the generated scene to sound like it was written by the original author in the character's voice, so that the output feels canonical.
16. As a reader, I want the generator to draw on the character's established relationships and vocabulary when writing the scene, so that it stays consistent with the persona.
17. As a reader, I want to specify which characters appear in the scene, so that multi-character interactions stay coherent.

### Infrastructure / Demo
18. As a judge, I want to see the 4-stage RunPod Flash pipeline architecture, so that I understand how GPU-backed serverless powers the product.
19. As a judge, I want to see Stage 2 burst to multiple workers and scale to zero after completion, so that the Flash workload pattern is demonstrated concretely.
20. As a judge, I want to see real latency numbers on the Stage 4 serving endpoint, so that I can evaluate production-readiness.
21. As a developer, I want the ingestion pipeline to run asynchronously as a Flash job, so that large corpora can be processed without blocking the serving endpoint.

## Implementation Decisions

### Pipeline Architecture

**Stage 1 — Preprocessing (Flash CPU endpoint, load-balanced)**
- Input: raw PDF of the HP corpus
- Output: structured JSONL — one record per dialogue line with fields: `{text, speaker, book, chapter, line_index}`
- Method: regex extraction of `"..." said/replied/shouted [Name]` patterns; ambiguous lines (no clean attribution) sent to a light LLM cleanup call
- Deployed as a Flash CPU endpoint (cheap, no GPU needed)

**Stage 2 — Persona Extraction (Flash GPU endpoint, vLLM Llama 3.1 8B, `workers=(0, 30)`)**
- Input: all dialogue lines for a given character
- Output: structured persona card JSON per character — fields: `{name, speech_style, vocabulary_fingerprint, core_values, key_relationships, knowledge_by_book, canonical_quotes}`
- One worker burst per character; scale to zero after completion
- This is the RunPod Flash showcase workload: one-time batch, parallelizable, burst-then-zero

**Stage 3 — Embeddings Index**
- Model: OpenAI `text-embedding-3-small`
- Every dialogue line embedded and stored in Qdrant Cloud
- Metadata per vector: `{speaker, book_number, chapter_number, line_index, raw_text}`
- Retrieval always filtered by `speaker` and `book_number <= current_book` (timeline filter)

**Stage 4 — Serving (Flash GPU endpoint, warm, wraps OpenAI GPT-4o)**
- Input: `{character, user_message, mode: "chat" | "fork" | "fanfic", fork_id?, custom_prompt?}`
- Retrieval: Qdrant filtered query → top-K lines → assembled context
- System prompt: persona card + mode-specific instruction
- For `fork` mode: context is canon up to the fork chapter; generation instruction explicitly licenses divergence post-fork
- For `chat` mode: citations extracted from retrieved lines and returned alongside response
- Output: `{response_text, citations: [{text, book, chapter}]}`
- Workers: warm (minimum 1), to avoid cold-start latency during live demo

### Frontend

- Framework: Next.js + Tailwind CSS
- Three surfaces accessible from a shared navigation: Character Chat, Fork Explorer, Fan Fiction
- Character Chat: character selector (Harry / Ron / Hermione), chat thread, citation side panel showing retrieved source passages
- Fork Explorer: grid of 4 fork cards with title + setup summary; click → streaming alternate scene generation
- Fan Fiction: character multi-select + free-text prompt → streaming scene output
- All generation surfaces stream responses token-by-token

### Vector DB
- Qdrant Cloud (free tier)
- Single collection: `hp_dialogue`
- Metadata filters on `speaker` (string) and `book_number` (integer, range filter for timeline)

### Character Roster (Demo)
- Harry Potter
- Ron Weasley
- Hermione Granger

### Fork Points (Pre-Curated)
1. "What if Harry was Sorted into Slytherin?"
2. "What if Harry let Voldemort kill him in the forest and didn't come back?"
3. "What if Hermione chose to stay behind instead of following Harry and Ron?"
4. "What if Draco accepted Dumbledore's offer at the top of the Astronomy Tower?"

Each fork point stores: `{id, title, setup_summary, fork_chapter, fork_book, character_pov, system_prompt_override}`

## Testing Decisions

A good test validates **external behavior** — what comes out of a module given a specific input — not implementation details like which internal functions were called or how the prompt was assembled.

### Modules to test

**Stage 1 — Speaker attribution parser**
- Test: given a raw excerpt with known dialogue, assert the output JSONL contains the correct speaker, book, chapter, and text fields
- Edge cases: nested quotes, thought-speech (`He thought, "..."`), letters, no attribution (narrator lines should be skipped or marked unknown)

**Stage 3 — Retrieval + timeline filter**
- Test: given a character and a book number, assert retrieved lines belong only to that speaker and books ≤ the specified book
- Test: assert that a query about a Book 7 event returns no results when `book_number <= 3` filter is applied

**Stage 4 — Serving endpoint contract**
- Test: given a valid `{character, user_message, mode: "chat"}` payload, assert response contains `response_text` (non-empty string) and `citations` (array with at least one entry containing `text`, `book`, `chapter`)
- Test: given `mode: "fork"` with a valid `fork_id`, assert response is non-empty and citations are all from books ≤ the fork's `fork_book`
- Do not test prompt internals; test only the output shape and citation validity

**Frontend — Fork Explorer**
- Test: clicking a fork card triggers a streaming response and renders output in the scene panel
- Test: citation panel populates with at least one source entry after a chat response completes

## Out of Scope

- LoRA fine-tuning for voice fidelity (architectural provision exists, not built for demo)
- User accounts, saved conversations, or history persistence
- Books other than Harry Potter
- Characters beyond Harry, Ron, and Hermione
- User-defined fork points beyond the 4 pre-curated ones (fan fiction free-form covers this need)
- Moderation or content filtering
- Mobile-optimized UI
- Multi-turn memory beyond a single conversation session

## Further Notes

- The HP corpus PDF is sourced from: `https://kvongcmehsanalibrary.wordpress.com/wp-content/uploads/2021/07/harrypotter.pdf` — verify copyright compliance before any public deployment; demo use only.
- The Stage 2 burst pattern (`workers=(0, 30)`) is the primary RunPod Flash demonstration moment — the architecture diagram should make this visible to judges.
- The timeline filter (speaker + book ≤ N) is the key technical differentiator from generic RAG chatbots; it should be called out explicitly in the demo narrative.
- Stage 4 uses OpenAI GPT-4o as the generation model (not a self-hosted model) to ensure reliable latency during the live demo. The Flash endpoint acts as the orchestration and retrieval layer wrapping the OpenAI call.
- Demo build order: Stage 1 (parse corpus) → Stage 3 (embed + load Qdrant) → Stage 2 (RunPod Flash GPU persona extraction) → Stage 4 (Flash serving endpoint) → Frontend.
