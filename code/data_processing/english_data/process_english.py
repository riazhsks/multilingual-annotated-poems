# Importantly, espeak-ng has to be installed for rhymetagger and prosodic to work

import re
import json
import nltk
import pandas as pd
import numpy as np
import prosodic 
from collections import Counter
from nltk.corpus import stopwords, cmudict
from nltk.tokenize.toktok import ToktokTokenizer
from rhymetagger import RhymeTagger
from g2p_en import G2p

from common_processing import (normalize_text, make_rhyme_scheme_per_stanza, 
                               convert_meter_type, get_meter_foot_and_foot_count, 
                               get_dominant_feature)

# Book extraction patterns
RE_SOURCE_BLOCK = re.compile(r"Source:", re.IGNORECASE)
RE_FROM_PATTERN = re.compile(r'^From\s+(.*?)(?=\(| by |\.)', re.IGNORECASE)
RE_PUB_IN_PATTERN = re.compile(r'Published in\s+(.*?)(?= on | by )', re.IGNORECASE)
RE_FROM_GENERAL = re.compile(r'from\s+(.*?)(?=\.|\s+Copyright| by )', re.IGNORECASE)

# Cleaning patterns
RE_PAREN_YEAR = re.compile(r'\s*\([^)]*\d{4}[^)]*\)$')
RE_CLEAN_WORD = re.compile(r"^[^a-z']+|[^a-z']+$")

# Authors/Metadata
CUTOFF_PHRASES = [
    ", edited", ", ed", " edited by", ", translated", ", as translated", 
    ", vol", " written by", " (edited by"
]


def extract_book_name(text):
    """Extracts the title of a book/collection."""
    if not isinstance(text, str):
        return None

    # Normalization
    text = text.replace('\xa0', ' ').strip()
    if text.startswith(("Notes:", "Translated by")):
        return None

    # Try "Source:" pattern
    if RE_SOURCE_BLOCK.search(text):
        content = RE_SOURCE_BLOCK.sub("", text)
        for line in content.split('\n'):
            clean_line = line.strip()
            if clean_line:
                return clean_line

    # Try Regex-based patterns
    for pattern in [RE_FROM_PATTERN, RE_PUB_IN_PATTERN, RE_FROM_GENERAL]:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
            
    return None

def clean_collection_title(title):
    """Cleans trailing editors, years, and punctuation from book titles."""
    if not title:
        return None
    
    title = RE_PAREN_YEAR.sub('', title)
    
    for phrase in CUTOFF_PHRASES:
        if phrase in title:
            title = title.split(phrase)[0]
            
    return title.strip(" ,:;")

def clean_collections(collections):
    return [clean_collection_title(extract_book_name(item)) for item in collections]

def get_stress_pattern(word, pronouncing_dict, g2p_instance):
    """Retrieves the syllable stress pattern (0=unstressed, 1=stressed)."""
    word = word.lower().replace("’", "'")
    
    # Try CMUdict
    if word in pronouncing_dict:
        phones = pronouncing_dict[word][0]
        # Map secondary stress (2) to unstressed (0) for metrical consistency
        return "".join([p[-1].replace("2", "0") for p in phones if p[-1].isdigit()])
    
    # Fallback to G2P model for archaic/unknown words
    phonemes = g2p_instance(word)
    stresses = []
    for p in phonemes:
        if any(c.isdigit() for c in p):
            digit = p[-1].replace("2", "0")
            stresses.append(digit)
    return "".join(stresses)


def get_primary_data(poem_text, tokenizer, stop_words):
    """Extracts raw text and normalized version."""
    if not poem_text:
        return None, None
    
    tokens = normalize_text(poem_text, "en", tokenizer, stop_words)
    normalized_text = " ".join(tokens) if tokens else None
    return str(poem_text), normalized_text

def get_metadata(i, titles, authors, collections, tokenizer, stop_words):
    """Extracts and normalizes metadata for the i-th poem in a corpus."""
    
    def process_field(arr):
        val = arr[i] if i < len(arr) and pd.notna(arr[i]) else None
        if not val: return None, None
        
        raw_val = str(val).strip()
        tokens = normalize_text(raw_val, "en", tokenizer, stop_words)
        norm_val = " ".join(tokens) if tokens else None
        return raw_val, norm_val

    title, norm_title = process_field(titles)
    coll, norm_coll = process_field(collections)

    # Author name formatting to "Name Surname" format 
    raw_author = authors[i] if i < len(authors) and pd.notna(authors[i]) else None
    author = None
    if raw_author:
        raw_author = str(raw_author).strip()
        if "," in raw_author:
            parts = [p.strip() for p in raw_author.split(",")]
            author = f"{parts[1]} {parts[0]}" if len(parts) == 2 else raw_author
        else:
            author = raw_author

    return title, norm_title, coll, norm_coll, author

def get_features(poem_text, i, forms, pronouncing_dict, g2p_instance, rhyme_tagger, meter_map):
    """Extracts poetic features from poem text."""

    # Stanza and line extraction
    stanzas = [s.strip() for s in re.split(r'\n\s*\n+', poem_text.strip()) if s.strip()]
    stanza_scheme_list = [len(s.split('\n')) for s in stanzas]
    lines = [line.strip() for line in poem_text.split('\n') if line.strip()]

    meter_types, meter_feet_sizes, line_syllables = [], [], []

    for line in lines:
        stress_pattern = ""
        for word in line.split():
            clean_w = RE_CLEAN_WORD.sub("", word.lower().replace("’", "'"))
            if clean_w:
                stress_pattern += get_stress_pattern(clean_w, pronouncing_dict, g2p_instance)
        
        if stress_pattern:
            line_syllables.append(len(stress_pattern))

        # Metrical Analysis
        meter_type, feet = None, None
        try: # A line can be handled by prosodic completely
            p_line = prosodic.Text(line)
            p_line.scan
            poem_line = p_line.lines[0]
            parser = poem_line.parse().best_parse

            meter_type = parser.foot_type
            feet = parser.foot_sizes
        except:
            # Fallback to extraction from stress pattern
            meter_type, feet = get_meter_foot_and_foot_count(stress_pattern)

        if meter_type:
            meter_types.append(convert_meter_type(meter_type.strip().lower(), meter_map))
        if feet:
            if type(feet) == int:
                meter_feet_sizes.append(feet)
            else:
                for foot in feet:
                    meter_feet_sizes.append(foot)

    # Rhyme scheme extraction
    try:
        rhymes = rhyme_tagger.tag(lines, output_format=3) if lines else None
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
        rhyme_str = None

    if rhyme_scheme:
        
        rhyme_str = " ".join(rhyme_scheme)

    stanza_scheme_str = " ".join(map(str, stanza_scheme_list)) if stanza_scheme_list else None
    stanza_rhyme = make_rhyme_scheme_per_stanza(rhyme_str, stanza_scheme_list) if rhyme_str else None

    meter_foot_count = get_dominant_feature([m for m in meter_feet_sizes if m is not None])
    meter_type_set = get_dominant_feature(meter_types, is_meter=True)

    if line_syllables:
        total_syll, mean_syll = sum(line_syllables), float(np.mean(line_syllables))
        num_lines = len(line_syllables)
    else:
        total_syll, mean_syll, num_lines = 0, 0.0, 0

    # Poetic Form
    form = str(forms[i]).strip().lower() if i < len(forms) and pd.notna(forms[i]) else None

    return (stanza_scheme_str, rhyme_str, stanza_rhyme, meter_type_set, 
            meter_foot_count, num_lines, total_syll, mean_syll, form)


def get_all_features(poem_text, i, titles, authors, collections, forms, pronouncing_dict, g2p, rt, meter_map, tokenizer, stop_words):
    text, normalized_text = get_primary_data(poem_text, tokenizer, stop_words)
    title, normalized_title, collection, normalized_collection, author =  get_metadata(i, titles, authors, collections, tokenizer, stop_words)
    (stanza_scheme_str, rhyme_str, 
     rhyme_per_stanza, meter_foot, 
     meter_foot_count, num_lines, 
     total_syll, mean_syll, 
     form) = get_features(poem_text, i, forms, pronouncing_dict, g2p, rt, meter_map)
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



def process_all_poems(url, output, pronouncing_dict, g2p, rt, meter_map,  tokenizer, stop_words):

    df = pd.read_csv(url)

    authors = df['author'].values
    titles = df['poem_title'].values
    texts = df['poem_text'].values
    forms = df['form'].values
    collections = df['poem_source'].values

    collections = clean_collections(collections)

    for i in range (0, len(texts)):
        data = get_all_features(texts[i], i, titles, authors, collections, forms, pronouncing_dict, g2p, rt, meter_map,  tokenizer, stop_words)
        with open(output, "a", encoding="utf-8") as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + "\n")
 

if __name__ == "__main__":
    nltk.download(['averaged_perceptron_tagger_eng', 'stopwords', 'cmudict'], quiet=True)
    stop_words = set(stopwords.words('english'))
    pronouncing_dict = cmudict.dict()
    tokenizer = ToktokTokenizer()
    g2p = G2p()
    rt = RhymeTagger()
    rt.load_model(model='en')

    meter_map = {
        'iambic': 'iambic', 'trochaic': 'trochaic', 'dactylic': 'dactylic',
        'amphibrachic': 'amphibrachic', 'anapestic': 'anapestic', 'unknown': 'unknown',
        'pentameter': 'pentameter', 'hexameter': 'hexameter', 'alexandrine': 'alexandrine'
    }

    url = "https://raw.githubusercontent.com/maria-antoniak/poetry-eval/refs/heads/main/data/poetry-evaluation_public-domain-poems.csv"
    output_filename = "english_data/english_poems_processed.jsonl"
    
    open(output_filename, "w", encoding="utf-8").close()
    process_all_poems(url, output_filename, pronouncing_dict, g2p, rt, meter_map,  tokenizer, stop_words)
