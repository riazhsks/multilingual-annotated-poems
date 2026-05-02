import os
from dotenv import load_dotenv
from openai import OpenAI
import json

PROMPT1 = """
## Fixed Poetic Form Identification

## Introduction

- **YOU ARE** a **LITERARY SCHOLAR AND POETRY ANALYST** with deep expertise in prosody, poetic traditions, and historical poetic forms across cultures and languages.

(Context: "Accurate identification of poetic forms is crucial for literary analysis, education, and digital humanities research.")

## Task Description

- **YOUR TASK IS** to **ANALYZE a poem provided in plain text** and **IDENTIFY its fixed poetic form**, if any.

- The poem will be provided in the variable:

{POEM_TEXT}

(Context: "The poem may belong to one the traditional fixed forms such as sonnet, couplet, ghazal, haiku, limerick, quatrain, sestina, villanelle, or to any other fixed form, or be unfixed.")

(Context: "If you conclude that the poem does not belong to any fixed form, mark its form as **unfixed**")

---

## Action Steps

### Step 0 — Text Normalization

- NORMALIZE whitespace and punctuation.
- SPLIT text into lines using line breaks.
- IF line breaks appear corrupted or merged, INFER likely line boundaries based on punctuation and rhythm.
- REMOVE empty lines.

### Step 1 — Structural Analysis

- **COUNT** the number of lines in the poem.
- **IDENTIFY** stanza structure (couplets, tercets, quatrains, etc.).
- **CHECK** whether there are repeating lines or refrains.
- **DETECT** characteristic structures (e.g., volta in sonnets, repeating couplets in ghazals).

(Context: "Structural features often determine the poetic form.")

---

### Step 2 — Rhyme Scheme Detection

- **EXTRACT** the rhyme scheme of the poem.
- **LABEL** rhyme patterns using letters (e.g. A B B A C C).
- **COMPARE** the detected rhyme scheme with known forms.

(Context: "Rhyme patterns provide strong signals of traditional poetic forms.")

---

### Step 3 — Meter and Line Pattern Recognition

- **ESTIMATE** the dominant meter if possible (e.g., iambic pentameter).
- **IDENTIFY** syllable patterns typical of certain forms (e.g., haiku 5-7-5).

(Context: "Many fixed forms are defined partly by meter or syllable count.")

---

### Step 4 — Form Matching

- **COMPARE** the poem’s structure with known fixed poetic forms such as:

  - Sonnet
  - Couplet
  - Ghazal
  - Haiku
  - Limerick
  - Quatrain
  - Sestina
  - Villanelle
  - Blank verse
  - Common measure
  - Rondel
  - Other classical forms

- **DETERMINE** the most likely form.

- IF key defining features of a form are missing (e.g., correct rhyme scheme, refrain structure, or meter), DO NOT classify the poem as that form.
- PARTIAL matches MUST result in "unfixed".

- For Czech and related traditions, consider forms such as:
  - sapfická strofa
  - tercína
  - hrdinský kuplet
  - stance
- APPLY language-specific criteria when identifying these forms.
- ADAPT analysis to the poem’s language (e.g., Czech, English, German, Hungarian, Spanish etc.), considering language-specific metrical traditions.
---

### Step 5 — Confidence Evaluation

- **ASSESS** how strongly the poem matches the form.
- If no strong match exists, classify it as "unfixed"

---

## Output Format
- YOU MUST RESPOND ONLY IN **JSON** FORMAT.

Provide the result in the following structured **JSON** format:
{{
    "stanza_scheme" : extracted stanza structure (e.g. 4 4 3 3),
    "rhyme_scheme" : extracted rhyme scheme (e.g. A B B A A B B A C D C E D E),
    "metrical_foot" : "iambic | trochaic | dactylic | anapestic | polymetric",
    "metrical_foot_count" : integer or "mixed",
    "line_count" : extracted line count (e.g. 14),
    "total_syllables" : extracted total number of syllables (e.g. 154),
    "average_syllable_count" : extracted average number of syllables per line (e.g. 11.0),
    "proof" : Concise justification of the chosen form. MUST explicitly reference line count, stanza scheme, rhyme scheme, and meter (if known),
    "form" : Name of form or "unfixed",
    "confidence" : integer (0 to 100)
}}
- ENSURE THE **JSON** IS VALID.
---

## Goals and Constraints

- **FOCUS** on **objective structural analysis** rather than interpretation of meaning.
- **ENSURE** the classification is based on **formal poetic criteria**.
- **YOU MUST AVOID** guessing without evidence.

(Context: "Scholarly rigor is essential when identifying poetic forms.")

---

## IMPORTANT

- "This analysis is important for literary research and education—accuracy matters."

- "You have deep expertise in poetic structure and prosody. Use it fully."

- "Careful reasoning will lead to reliable identification of poetic forms."

---

**EXAMPLE of required response**

EXAMPLE 1:

INPUT:
Ve srdci dík a na rtech píseň nesa,blížím se tobě, všehomíra Pane, z milosti tvojí svaté, svrchované
zbavena trýzně duše moje plesá.
I stál jsem sám a v úzkostech se třesa,
tys viděl srdce choré, zbědované, palčivou slzu, kterak přes tvář kane,
a tělo bídné, ano hyne, klesá.
I dal jsi vzejíť slunci slitování,
vložil jsi balsam na hlubokou ránu na setřel slzy všemohoucí dlaní.
Otče náš, láska tvá je nekonečna: prosícím dáváš slova svého manu
a stonásobně žehnáš srdce vděčná.

OUTPUT:
{{
"stanza_scheme" : "4 4 3 3",
"rhyme_scheme" : "A B B A A B B A C D C E D E",
"metrical_foot" : "iambic",
"metrical_foot_count" : 5,
"line_count" : 14,
"total_syllables" : 154,
"average_syllable_count" : 11.0,
"proof" : "This is a sonnet because it has 14 lines, a stanza scheme 4 4 3 3, and a iambic pentameter, which are all typical features of a sonnet.",
"form": "sonnet", 
"confidence" : 95
}}

EXAMPLE 2:

INPUT:
(14-line poem with irregular rhyme and no consistent meter)

OUTPUT:
{{
"stanza_scheme": "4 4 3 3",
"rhyme_scheme": "A B C D E F G H I J K L M N",
"metrical_foot": "polymetric",
"metrical_foot_count": "mixed",
"line_count": 14,
"total_syllables": 150,
"average_syllable_count": 12.3,
"proof": "The poem has 14 lines but lacks a consistent rhyme scheme and meter, so it does not meet the criteria for a sonnet or any other fixed form.",
"form": "unfixed",
"confidence": 90
}}
"""
 


PROMPT2 =  """
## Fixed Poetic Form Identification (Closed Set Classification)

## Introduction

- **YOU ARE** a **LITERARY SCHOLAR AND POETRY ANALYST** with deep expertise in prosody, poetic traditions, and historical poetic forms across cultures and languages.

(Context: "Accurate identification of poetic forms is crucial for literary analysis, education, and digital humanities research.")

---

## Task Description

- **YOUR TASK IS** to **ANALYZE a poem provided in plain text** and **CLASSIFY it into ONE form from a fixed list**, or "unfixed" if no form matches.

- The poem will be provided in the variable:

{POEM_TEXT}

---

## Allowed Forms (STRICT CLOSED SET)

You MUST choose EXACTLY ONE form from this list:

### General / International Forms
- sonnet
- couplet
- quatrain
- ghazal
- haiku
- limerick
- sestina
- villanelle
- blank verse
- common measure
- rondel

---

### German poetic forms
- knittelvers (irregular rhymed German verse, often rhyming couplets with loose meter)
- liedstrophe (strophic song form common in German lyric poetry)
- sonett (German sonnet tradition variant; treat as "sonnet" if English-style, but this label only if explicitly German structure dominates)

---

### Hungarian poetic forms
- dal (Hungarian lyrical song form with regular stanzaic structure)
- felező tizenkettes (Hungarian 12-syllable hemistich verse; often epic/ballad structure)
- bokorrím (Hungarian clustered rhyme structure; often stanza-based rhyming pattern)

---

### Spanish poetic forms
- romance (Spanish narrative ballad in octosyllabic lines with assonant rhyme in even lines)
- copla (Spanish quatrain-based folk form)
- décima (10-line stanza with ABBAACCDDC rhyme scheme)
- redondilla (octosyllabic quatrain with ABBA rhyme)

---

### Czech / Central European forms
- sapfická strofa
- tercína
- hrdinský kuplet
- stance

---

### Fallback
- unfixed

---

## Critical Classification Rules

- You MUST choose ONLY from the allowed list.
- DO NOT invent new forms.
- DO NOT output paraphrases or hybrids.
- IF the poem partially matches a form → classify as "unfixed".
- Structural precision is required (rhyme scheme, stanza structure, meter where applicable).

---

## Action Steps

### Step 0 — Text Normalization
- Normalize whitespace and punctuation.
- Split into lines.
- Infer missing line breaks if necessary.
- Remove empty lines.

---

### Step 1 — Structural Analysis
- Count lines.
- Identify stanza structure.
- Detect refrains or repetition patterns.

---

### Step 2 — Rhyme Scheme Detection
- Extract rhyme scheme (AABB, ABAB, etc.).
- Detect assonance vs full rhyme where relevant (especially Spanish forms).

---

### Step 3 — Meter Recognition
- Estimate dominant meter type.
- Identify syllable patterns if possible.

---

### Step 4 — Form Matching
- Match ONLY against allowed list.
- Require all defining constraints of the form.

---

### Step 5 — Confidence Evaluation
- If uncertainty or partial match → "unfixed"

---

## Output Format
- YOU MUST RESPOND ONLY IN **JSON** FORMAT.

Provide the result in the following structured **JSON** format:
{{
    "stanza_scheme" : extracted stanza structure (e.g. 4 4 3 3),
    "rhyme_scheme" : extracted rhyme scheme (e.g. A B B A A B B A C D C E D E),
    "metrical_foot" : "iambic | trochaic | dactylic | anapestic | polymetric",
    "metrical_foot_count" : integer or "mixed",
    "line_count" : extracted line count (e.g. 14),
    "total_syllables" : extracted total number of syllables (e.g. 154),
    "average_syllable_count" : extracted average number of syllables per line (e.g. 11.0),
    "proof" : Concise justification of the chosen form. MUST explicitly reference line count, stanza scheme, rhyme scheme, and meter (if known),
    "form" : Name of form or "unfixed",
    "confidence" : integer (0 to 100)
}}
- ENSURE THE **JSON** IS VALID.
---

## Proof Requirements
Must explicitly reference:
- line count
- stanza structure
- rhyme scheme
- meter (if known)

---

## Goals and Constraints
- Focus strictly on structure.
- Do not interpret meaning.
- Do not guess.

---

## Example

EXAMPLE 1:

INPUT:
Ve srdci dík a na rtech píseň nesa,blížím se tobě, všehomíra Pane, z milosti tvojí svaté, svrchované
zbavena trýzně duše moje plesá.
I stál jsem sám a v úzkostech se třesa,
tys viděl srdce choré, zbědované, palčivou slzu, kterak přes tvář kane,
a tělo bídné, ano hyne, klesá.
I dal jsi vzejíť slunci slitování,
vložil jsi balsam na hlubokou ránu na setřel slzy všemohoucí dlaní.
Otče náš, láska tvá je nekonečna: prosícím dáváš slova svého manu
a stonásobně žehnáš srdce vděčná.

OUTPUT:
{{
"stanza_scheme" : "4 4 3 3",
"rhyme_scheme" : "A B B A A B B A C D C E D E",
"metrical_foot" : "iambic",
"metrical_foot_count" : 5,
"line_count" : 14,
"total_syllables" : 154,
"average_syllable_count" : 11.0,
"proof" : "This is a sonnet because it has 14 lines, a stanza scheme 4 4 3 3, and a iambic pentameter, which are all typical features of a sonnet.",
"form": "sonnet", 
"confidence" : 95
}}

EXAMPLE 2:

INPUT:
(14-line poem with irregular rhyme and no consistent meter)

OUTPUT:
{{
"stanza_scheme": "4 4 3 3",
"rhyme_scheme": "A B C D E F G H I J K L M N",
"metrical_foot": "polymetric",
"metrical_foot_count": "mixed",
"line_count": 14,
"total_syllables": 150,
"average_syllable_count": 12.3,
"proof": "The poem has 14 lines but lacks a consistent rhyme scheme and meter, so it does not meet the criteria for a sonnet or any other fixed form.",
"form": "unfixed",
"confidence": 90
}}
"""

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
 
FILES = {"../shuffled/shuffled_samples/english_shuffled.jsonl" : "english_predictions.txt",
         "../shuffled/shuffled_samples/spanish_shuffled.jsonl" : "spanish_predictions.txt",
         "../shuffled/shuffled_samples/czech_c3p_shuffled.jsonl" : "czech_c3p_predictions.txt",
         "../shuffled/shuffled_samples/czech_ccv_shuffled.jsonl" : "czech_ccv_predictions.txt",
         "../shuffled/shuffled_samples/german_shuffled.jsonl" : "german_predictions.txt",
         "../shuffled/shuffled_samples/hungarian_shuffled.jsonl" : "hungarian_predictions.txt"}

MODELS = ["openai/gpt-4o", "google/gemini-2.5-pro" ]
PROMPTS =  {"P1" : PROMPT1, "P2" : PROMPT2}


def format_with_stanzas(text, stanza_scheme):
    """Formats the text using the stanza scheme."""
    text_by_stanza = []
    lines = text.split('\n')

    # split text into stanzas
    current_id = 0
    for stanza in stanza_scheme:
        for line in lines[current_id: current_id + stanza]:
            text_by_stanza.append(line)
            text_by_stanza.append('\n')
        current_id += stanza
        text_by_stanza.append('\n')
    text_by_stanza = "".join(text_by_stanza)
    return text_by_stanza.strip()


for input_file, output_file in FILES.items():
    with open(input_file,"r") as file:
        for line in file:
            # Load a poem record and format the text using the stanza scheme
            poem = json.loads(line)
            stanza_scheme = [int(s) for s in poem["stanza_scheme"].split()]
            text_by_stanza = format_with_stanzas(poem["text"], stanza_scheme)

            for model in MODELS:
                model_name = model.split("/")[1]
                for prompt_name, prompt in PROMPTS.items():
                    save_dir = f"{model_name}/{prompt_name}"
                    os.makedirs(save_dir, exist_ok=True)
                    with open(f"{save_dir}/{output_file}","a") as out:
                        # API Call to OpenRouter
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user", 
                                    "content": prompt.format(POEM_TEXT=text_by_stanza)
                                }
                            ],
                            response_format={ "type": "json_object" } 
                        )
                        # Extract and save the LLM response
                        response = completion.choices[0].message.content
                        out.write(response)
