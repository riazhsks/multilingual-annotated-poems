import numpy as np
import json
import urllib.request
import pandas as pd
from collections import Counter
from nltk.tokenize.toktok import ToktokTokenizer
from stop_words import get_stop_words

from common_processing import normalize_text, make_rhyme_scheme_per_stanza, convert_meter_type, get_dominant_feature


def get_primary_data(poem, tokenizer, stop_words):
    """
    Extracts raw poem text and its normalized version.
    """
    body = poem.get("body")
    if not body:
        return None, None

    # Get full text from lines
    lines = [str(line.get('text', '')) for line in body]
    full_text = "\n".join(lines)

    # Normalize text
    tokens = normalize_text(full_text, "cs", tokenizer, stop_words)
    normalized_text = " ".join(tokens) if tokens else None

    return full_text, normalized_text

def get_metadata(poem, tokenizer, stop_words):
    """
    Extracts title of the poem, collection title, their normalized versions, and author name.
    """
    # Helper to extract and normalize specific metadata fields
    def extract_and_normalize(field_name):
        value = poem.get(field_name)
        if not value:
            return None, None
        
        raw_val = str(value).strip()
        tokens = normalize_text(raw_val, "cs", tokenizer, stop_words)
        norm_val = " ".join(tokens) if tokens else None
        return raw_val, norm_val

    title, normalized_title = extract_and_normalize('title')
    collection, normalized_collection = extract_and_normalize('b_title')

    # Format author name in the format "Name Surname"
    raw_author = poem.get('author')
    author = None
    if raw_author:
        raw_author = str(raw_author).strip()
        if "," in raw_author:
            parts = [p.strip() for p in raw_author.split(",")]
            if len(parts) == 2:
                author = f"{parts[1]} {parts[0]}"
            else:
                author = raw_author
        else:
            author = raw_author

    return title, normalized_title, collection, normalized_collection, author


def get_features(poem, meter_map):
    """
    Extracts poetic features: stanzas, rhymes, syllables, and meter.
    Fallback to None for string features, and to 0 for numeric features.
    """
    body = poem.get("body") or []
    schemes = poem.get("schemes") or {}
    
    meter_feet_raw = []
    meter_types_raw = []
    line_syllables = []

    for line in body:
        # Meter extraction
        line_meter = line.get("metre", {})
        for m_type, m_data in line_meter.items():
            meter_types_raw.append(str(m_type))
            foot = m_data.get("foot")
            if foot is not None:
                meter_feet_raw.append(str(foot))

        # Syllable extraction
        syll_count = sum(len(word.get("syllables", [])) for word in line.get("words", []))
        line_syllables.append(syll_count)

    # Stanza scheme extraction

    stanza_scheme_raw = schemes.get("stanza_scheme", None) if schemes else None
    if stanza_scheme_raw:
        stanza_scheme_list = [len(group) for group in stanza_scheme_raw]
    else:
        # Fallback: group by the stanza index in lines
        stanza_indices = [line.get('stanza', 0) for line in body]
        counts = Counter(stanza_indices)
        stanza_scheme_list = [counts[i] for i in sorted(counts.keys())]
    
    stanza_scheme_str = " ".join(map(str, stanza_scheme_list)) if stanza_scheme_list else None

    # Rhyme scheme extraction
    # Replace unlabeled rhymes with unique incrementing IDs
    raw_rhymes = [str(line.get('rhyme', 'None')) for line in body]
    try:
        next_id = max([int(r) for r in raw_rhymes if r.isdigit()] + [0]) + 1
    except ValueError:
        next_id = 1

    final_rhymes = []
    for r in raw_rhymes:
        if r == "None":
            final_rhymes.append(str(next_id))
            next_id += 1
        else:
            final_rhymes.append(r)
    
    rhyme_scheme_str = " ".join(final_rhymes) if final_rhymes else None

    # Stanza rhyme scheme (rhyme grouped by stanza) extraction 
    rhyme_scheme_per_stanza = None
    if rhyme_scheme_str and stanza_scheme_list:
        rhyme_scheme_per_stanza = make_rhyme_scheme_per_stanza(rhyme_scheme_str, stanza_scheme_list)

    # Meter foot count extraction
    meter_foot_count = get_dominant_feature(meter_feet_raw)
    
    # Meter foot types extraction using the external meter_map
    processed_types = []
    for m_t in meter_types_raw:
        mapped = convert_meter_type(m_t.lower(), meter_map)
        if mapped:
            processed_types.append(mapped)

    # Extraction of the dominant meter type
    meter_foot = get_dominant_feature(processed_types, is_meter=True)

    # Syllable counts (total and avg) extraction
    if line_syllables:
        total_syllables = sum(line_syllables)
        number_of_lines = len(line_syllables)
        mean_syllables = float(total_syllables / number_of_lines)
    else:
        total_syllables, number_of_lines, mean_syllables = 0, 0, 0.0

    # Poetic form extraction
    form_val = schemes.get("form")
    form = form_val.lower() if form_val else None

    return (stanza_scheme_str, rhyme_scheme_str, rhyme_scheme_per_stanza, meter_foot, 
            meter_foot_count, number_of_lines, total_syllables, mean_syllables, form)

def get_all_features(url, tokenizer, stop_words, meter_map):
    """
    Fetches poem JSON from API and aggregates all processed features.
    """
    try:
        with urllib.request.urlopen(url) as response:
            if response.getcode() != 200:
                return None
            poem = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Connection error for {url}: {e}")
        return None

    if not poem:
        return None

    text, norm_text = get_primary_data(poem, tokenizer, stop_words)
    title, norm_title, coll, norm_coll, author = get_metadata(poem, tokenizer, stop_words)
    
    (st_sch, rh_sch, st_rh_sch, m_type, m_foot, 
     lines, tot_syll, avg_syll, form) = get_features(poem, meter_map)

    return {
        "title": title,
        "normalized_title": norm_title,
        "author": author,
        "collection": coll,
        "normalized_collection": norm_coll,
        "form": form,
        "stanza_scheme": st_sch,
        "rhyme_scheme": rh_sch,
        "rhyme_scheme_per_stanza": st_rh_sch,
        "metrical_foot": m_type,
        "metrical_foot_count": m_foot,
        "line_count": lines,
        "total_syllables": tot_syll,
        "average_syllable_count": avg_syll,
        "text": text,
        "normalized_text": norm_text
    }

def process_all_poems(output_path, duplicates_list, is_ccv=True):
    """
    Main loop to iterate through poem IDs and write results to JSONL.
    """
    # Range configuration based on dataset
    start, end = (0, 80229) if is_ccv else (100000, 180130)
    duplicates_set = set(duplicates_list) if duplicates_list else set()

    with open(output_path, "a", encoding="utf-8") as out_file:
        for poem_id in range(start, end):
            if poem_id in duplicates_set:
                print(f"Skipping duplicate {poem_id}")
                continue

            url = f"https://quest.ms.mff.cuni.cz/edupo-api/show?poemid={poem_id}&accept=json"
            
            try:
                data = get_all_features(url, tokenizer, stop_words, meter_map)
                if data:
                    out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    print(f"Could not process: {url}")
            except Exception as e:
                print(f"Error at {url}: {e}")


if __name__ == "__main__":
    # Global Initializations
    tokenizer = ToktokTokenizer()
    stop_words = get_stop_words("czech")
    meter_map = {
        "j": "iambic", 
        "t": "trochaic", 
        "d": "dactylic", 
        "a": "amphibrachic",
        "n": "unknown", 
        "x": "logaoedic", 
        "y": "logaoedic-with-anacrusis",
        "penta": "pentameter", 
        "hexa": "hexameter", 
        "alex": "alexandrine",
    }

    # Process CCV Dataset
    ccv_output = "czech_data/czech_poems_processed_ccv.jsonl"
    open(ccv_output, "w", encoding="utf-8").close() # Clear file
    
    # Remove duplicates (obtained from the EduPo project)
    try:
        dup_df = pd.read_csv('czech_data/duplicates_list.csv', sep=';', usecols=[0])
        dups = dup_df['poem_id'].tolist()
    except FileNotFoundError:
        dups = []

    process_all_poems(ccv_output, dups, is_ccv=True)

    # Process C3P Dataset
    c3p_output = "czech_data/czech_poems_processed_c3p.jsonl"
    open(c3p_output, "w", encoding="utf-8").close() # Clear file
    process_all_poems(c3p_output, None, is_ccv=False)


