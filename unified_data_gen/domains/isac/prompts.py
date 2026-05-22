# prompts_isac.py

QA_SYSTEM_PROMPT = "You are an expert in wireless communications and Integrated Sensing and Communication (ISAC)."

RELEVANCE_SYSTEM = "You are a wireless communications and sensing domain expert curating a tutorial dataset."

RELEVANCE_USER = """Decide whether this Tele-Data sample is suitable for a tutorial dataset focused on Integrated Sensing and Communication (ISAC) / Joint Communication and Sensing (JCAS) for 5G-Advanced and 6G.

Include content that is explanatory/instructional about:
- ISAC/JCAS concepts: shared spectrum/hardware, joint waveform design, dual-function radar-communication, sensing-assisted comm, comm-assisted sensing
- Waveforms & signals: OFDM-based sensing, OTFS, FMCW/PMCW comparisons, radar ambiguity function basics, pilot/PRS/SRS reuse ideas
- Beamforming & MIMO: joint precoding, beam management for sensing+comm, monostatic/bistatic/multistatic sensing
- Signal processing: radar parameter estimation (range/Doppler/angle), clutter suppression, CFAR (high level), tracking basics
- Performance metrics & trade-offs: sensing resolution/accuracy vs comm rate/reliability/latency, Cramér–Rao bound discussions (high level), resource allocation
- Standards-oriented or practical system discussion: 3GPP/IEEE aligned sensing in cellular/Wi-Fi contexts, if technical

Exclude (reject) if it is mainly:
- marketing, vendor promotion, product announcements, generic business/market news
- pure radar with no communication aspect (unless it clearly teaches ISAC-relevant comparisons/trade-offs)
- pure communication performance evaluation without sensing component
- unrelated topics (security policy docs, generic IoT apps without ISAC, unrelated networking)

Return ONLY JSON:
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "main_topics": ["..."],
  "notes": "short reason"
}}

SAMPLE_ID: {sid}
CATEGORY: {cat}
META_HINT: {meta_hint}
CONTENT_SNIPPET:
<<<
{snippet}
>>>
"""

GEN_SYSTEM = "You are generating high-quality training examples for a tutorial assistant. Outputs must be self-contained and technically correct."

# GEN_USER = """From the text chunk below, generate training examples in 5 categories (same schema as the existing pipeline):

# 1) concept_qa: conceptual Q/A pairs about ISAC
# 2) procedure_explanation: a short step-by-step explanation of an ISAC workflow (e.g., how sensing is performed in an OFDM system, how range/Doppler is estimated, how resources are allocated)
# 3) role_responsibility: responsibilities/roles of entities (e.g., sensing transmitter vs receiver, BS/gNB vs UE, monostatic vs bistatic nodes, scheduler vs signal processing blocks)
# 4) common_misconception: misconception + clarification
# 5) definition: glossary entry (term + definition + related_terms list)

# Rules:
# - DO NOT reference "this document/section/figure/table/above/below".
# - No citations, no section numbers, no “as shown”.
# - Be concise, tutorial-friendly, and generalizable (avoid overly specific parameter dumps).
# - If the chunk lacks info for a category, return an empty list for that category.
# - For role_responsibility: keep it concrete (who does what) and aligned with ISAC architectures.
# - For procedure_explanation: keep it as a practical workflow, not just a list of terms.

# Return ONLY JSON with this exact shape:
# {{
#   "concept_qa": [
#     {{"prompt": "...", "response": "..."}}
#   ],
#   "procedure_explanation": [
#     {{"prompt": "...", "response": "..."}}
#   ],
#   "role_responsibility": [
#     {{"prompt": "...", "response": "..."}}
#   ],
#   "common_misconception": [
#     {{"prompt": "...", "response": "..."}}
#   ],
#   "definition": [
#     {{"term": "...", "definition": "...", "related_terms": ["...", "..."]}}
#   ]
# }}

# TEXT_CHUNK:
# <<<
# {chunk}
# >>>
# """

GEN_USER = """From the TEXT_CHUNK below, generate high-quality, self-contained training examples in the following categories:

1) concept_qa  
2) procedure_explanation  
3) role_responsibility  
4) common_misconception  
5) definition  

STRICT REQUIREMENTS FOR ALL RESPONSES:

- The response must be fully understandable without access to the source document.
- Do NOT reference “this paper”, “this section”, “above”, “below”, figures, tables, or citations.
- Restate necessary assumptions or definitions explicitly.
- When technical terms are introduced, define them briefly.
- Include at least one mechanism, cause-effect relationship, or technical reasoning step where applicable.
- Avoid vague or generic explanations.
- Be concise but technically complete.
- Use precise telecom terminology.
- Do not invent information not supported by the chunk.

CATEGORY-SPECIFIC REQUIREMENTS:

concept_qa:
- The answer must include a definition + mechanism or implication.
- Avoid one-sentence shallow answers.

procedure_explanation:
- Present as a clear step-by-step workflow.
- Each step should explain WHY it is performed, not just WHAT.

role_responsibility:
- Clearly specify the entity (e.g., gNB, UE, sensing node, scheduler).
- Describe concrete responsibilities and interactions.

common_misconception:
- Clearly state the misconception.
- Provide a correction with explanation of why the misconception is incorrect.

definition:
- Provide a precise technical definition.
- Add 2–4 related terms that are meaningfully connected.

If the chunk lacks sufficient information for a category, return an empty list for that category.

Return ONLY JSON with this exact shape:

{{
  "concept_qa": [
    {{"prompt": "...", "response": "..."}}
  ],
  "procedure_explanation": [
    {{"prompt": "...", "response": "..."}}
  ],
  "role_responsibility": [
    {{"prompt": "...", "response": "..."}}
  ],
  "common_misconception": [
    {{"prompt": "...", "response": "..."}}
  ],
  "definition": [
    {{"term": "...", "definition": "...", "related_terms": ["...", "..."]}}
  ]
}}

TEXT_CHUNK:
<<<
{chunk}
>>>
"""

JUDGE_SYSTEM = "You are a strict dataset quality reviewer for ISAC tutorial data."

JUDGE_USER = """Judge whether the following generated training example should be kept.

Reject if:
- It requires the original source text to understand (context-dependent, refers to 'this paper/section/figure/table').
- It is vague, hand-wavy, or mainly fluff.
- It contains factual errors or mismatched concepts (e.g., confusing radar Doppler with carrier frequency offset).
- It is not about ISAC/JCAS or has no sensing+comm linkage.
- It includes citations, section numbers, or local references.

Return ONLY JSON with this exact shape:
{{
  "keep": true/false,
  "answerable_without_context": true/false,
  "technical_score": 0.0-1.0,
  "clarity_score": 0.0-1.0,
  "issues": ["..."]
}}

EXAMPLE_TO_JUDGE_JSON:
<<<
{example_json}
>>>
"""
