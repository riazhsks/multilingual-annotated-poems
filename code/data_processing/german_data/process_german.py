# Importantly, espeak-ng has to be installed for rhymetagger to work

import os
import json
import re
import nltk
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from rhymetagger import RhymeTagger

from common_processing import (normalize_text, make_rhyme_scheme_per_stanza, get_dominant_feature, convert_meter_type)

METER_TYPE_MAP = {
    "iambic": 'iambic',
    "alexandrine": 'alexandrine',
    "trochaic": 'trochaic',
    "anapaest": 'anapestic',
    "dactylic": 'dactylic',
    "amphibrach": 'amphibrachic',
    "spondeus": 'spondeus',
    "hexameter": 'hexameter'
}

METER_FEET_MAP = {
    "single": 1, "di": 2, "tri": 3, "tetra": 4, 
    "penta": 5, "hexa": 6, "septa": 7, "octa": 8
}

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def get_meter_foot(meter_raw):
    """Extracts metrical foot type."""
    if not meter_raw:
        return None
    for key, value in METER_TYPE_MAP.items():
        if key in meter_raw.lower():
            return value
    try:
        m_type = meter_raw.split(".")[0].strip().lower()
        return None if m_type == "unknown" else m_type
    except Exception:
        return None

def get_meter_foot_count(meter_raw):
    """Extracts metrical foot count."""
    if not meter_raw:
        return None
    for key, value in METER_FEET_MAP.items():
        if key in meter_raw.lower():
            return value
    return None


def get_primary_data(poem_dict, tokenizer, stop_words):
    """Extracts raw poem text and its normalized version."""
    try:
        poem = poem_dict[list(poem_dict.keys())[0]]["poem"]
        
        text_lines = []
        for stanza_id in poem:
            lines = poem[stanza_id]
            for line_id in lines:
                text_lines.append(lines[line_id]["text"])
        
        full_text = "\n".join(text_lines)
        
        # Normalize text 
        tokens = normalize_text(full_text, "de", tokenizer, stop_words)
        normalized_text = " ".join(tokens) if tokens else None
        
        return full_text, normalized_text
    except Exception:
        return None, None

def get_metadata(poem_dict, tokenizer, stop_words):
    """Extracts and formats title, collection, and author metadata."""
    try:
        metadata = poem_dict[list(poem_dict.keys())[0]]["metadata"]
    except Exception:
        return None, None, None, None, None

    # Process title
    title = metadata.get("title")
    if title in (None, "N.A."):
        title, normalized_title = None, None
    else:
        title = str(title).strip()
        tokens = normalize_text(title, "de", tokenizer, stop_words)
        normalized_title = " ".join(tokens) if tokens else None

    # Process collection
    collection = metadata.get("booktitle")
    if collection in (None, "N.A."):
        collection, normalized_collection = None, None
    else:
        collection = str(collection).strip()
        tokens = normalize_text(collection, "de", tokenizer, stop_words)
        normalized_collection = " ".join(tokens) if tokens else None

    # Process author name to "Name Surname" format
    try:
        author_raw = str(metadata["author"]["name"])
        parts = author_raw.split(",")
        if len(parts) == 2:
            author = f"{parts[1].strip()} {parts[0].strip()}"
        else:
            author = author_raw.strip()
    except Exception:
        author = None
    
    return title, normalized_title, collection, normalized_collection, author

def get_features(poem_dict, tei_tree, rt):
    """Extracts poetic features."""
    poem_lines, stanza_scheme = [], []
    meter_types, meter_feet_sizes = [], []
    
    first_key = list(poem_dict.keys())[0]
    poem_content = poem_dict[first_key]["poem"]

    # Stanza scheme, lines and metrical extraction
    for stanza_id in poem_content:
        lines = poem_content[stanza_id]
        stanza_scheme.append(len(lines))
        for line_id in lines:
            line_data = lines[line_id]
            poem_lines.append(line_data["text"])

       
            meter_type = get_meter_foot(line_data["measure"])
            if meter_type: 
                meter_types.append(meter_type)
                
            m_feet = get_meter_foot_count(line_data["measure"])
            if m_feet:
                if type(m_feet) == int:
                    meter_feet_sizes.append(m_feet)
                else:
                    for foot in m_feet:
                        meter_feet_sizes.append(foot)

    # Rhyme scheme extraction
    try:
        rhymes = rt.tag(lines, output_format=3) if lines else None
        rhyme_scheme = [str(x) for x in rhymes] if rhymes else None
        if rhyme_scheme:
            try:
                max_in_rhyme_scheme = max([int(x) for x in rhyme_scheme if x != "None"]) + 1
            except:
                max_in_rhyme_scheme = 1

            for k in range(len(rhyme_scheme)):
                if rhyme_scheme[k] == "None":
                    rhyme_scheme[k] = str(max_in_rhyme_scheme)
                    max_in_rhyme_scheme += 1
    except KeyError:
        rhyme_scheme_str = None

    stanza_rhyme_scheme = None
    if rhyme_scheme_str and stanza_scheme:
        stanza_rhyme_scheme = make_rhyme_scheme_per_stanza(rhyme_scheme_str, stanza_scheme)

    meter_feet_count = get_dominant_feature(meter_feet_sizes)
    meter_feet = get_dominant_feature(meter_types, is_meter=True)

    stanza_scheme_str = " ".join(map(str, stanza_scheme)) if stanza_scheme else None

    # Syllable counts extraction
    try:
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        syll_node = tei_tree.xpath("//tei:measure[@type='syllables']/text()", namespaces=ns)
        total_syllables = int(syll_node[0]) if syll_node else 0
    except Exception:
        total_syllables = 0
    
    line_count = len(poem_lines)
    avg_syllables = total_syllables / line_count if line_count > 0 else 0.0

    return (stanza_scheme_str, rhyme_scheme_str, stanza_rhyme_scheme, meter_feet, 
            meter_feet_count, line_count, total_syllables, avg_syllables, None)

def process_poem(data, tree, rt, tokenizer, stop_words):
    """Extracts all features for a single poem."""
    text, normalized_text = get_primary_data(data, tokenizer, stop_words)
    title, normalized_title, collection, normalized_collection, author = get_metadata(data, tokenizer, stop_words)

    (stanza_scheme_str, rhyme_str, rhyme_per_stanza, meter_foot, meter_foot_count, 
     num_lines, total_syll, mean_syll, form) = get_features(data, tree, rt)

    return {
        "title": title,
        "normalized_title": normalized_title,
        "author": author,
        "collection": collection,
        "normalized_collection": normalized_collection,
        "form": form,
        "stanza_scheme": stanza_scheme_str,
        "rhyme_scheme": rhyme_str,
        "rhyme_scheme_per_stanza": rhyme_per_stanza,
        "metrical_foot": meter_foot,
        "metrical_foot_count": meter_foot_count,
        "line_count": num_lines,
        "total_syllables": total_syll,
        "average_syllable_count": mean_syll,
        "text": text,
        "normalized_text": normalized_text
    }

def process_all_poems(json_directory, tei_directory, output_path, rt, tokenizer, stop_words):
    """Extracts features for all poems."""
 
    with open(output_path, "a", encoding="utf-8") as out:
        for root, _, files in os.walk(json_directory):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                
                json_path = os.path.join(root, filename)

                tei_filename = filename.rsplit(".", 1)[0] + ".tei.xml"
                tei_path = os.path.join(tei_directory, tei_filename)

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        poem_data = json.load(f)
                    
                    poem_tree = etree.parse(tei_path)
                    record  = process_poem(poem_data, poem_tree, rt, tokenizer, stop_words)

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
   
        
if __name__ == "__main__":
    nltk_data_dir = os.environ.get("NLTK_DATA", ".")
    stop_words = set(stopwords.words('german'))
    tokenizer = ToktokTokenizer()

    rt = RhymeTagger()
    rt.load_model(model='de')

    json_dir = "../DLK/DLK/meterized/json_DLK_v6"
    tei_dir = "../DLK/DLK/tei/tei_plain"
    output_file = "german_data/german_poems_processed.jsonl"

    # Reset output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as _: pass

    process_all_poems(json_dir, tei_dir, output_file, rt, tokenizer, stop_words)