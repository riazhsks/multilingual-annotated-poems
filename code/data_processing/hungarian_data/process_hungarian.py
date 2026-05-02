import os
import json
from collections import Counter
from lxml import etree
from nltk.tokenize.toktok import ToktokTokenizer
from stop_words import get_stop_words
from common_processing import (
    normalize_text, make_rhyme_scheme_per_stanza, convert_meter_type, 
    rhyme_scheme_to_numbers, get_meter_foot_and_foot_count, get_dominant_feature
)


NS_TEI = {"tei": "http://www.tei-c.org/ns/1.0"}
FOOT_LENGTHS = {
    'iambic': 2, 'trochaic': 2,
    'dactylic': 3, 'anapestic': 3,
    'amphibrachic': 3 
}

def get_primary_data(poem_tree):
    """Extracts raw poem text and its normalized version."""
    try:
        # Extract lines from body
        lines_elems = poem_tree.xpath("//text/body/div[@type='poem']//l")
        lines = [" ".join(t.strip() for t in l.xpath(".//text()") if t.strip()) for l in lines_elems]
        text = "\n".join(lines) if lines else None

        # Normalize using Hungarian stopwords
        tokens = normalize_text(text, "hu", tokenizer, stop_words) if text else None
        normalized_text = " ".join(tokens) if tokens else None    
        return text, normalized_text
    except Exception:
        return None, None
    
def get_metadata(poem_tree):
    """Extracts title, collection, normalized versions and author."""
    # Process title
    try:
        title_nodes = poem_tree.xpath("//tei:teiHeader//tei:titleStmt/tei:title[@type='main']/text()", namespaces=NS_TEI)
        title = str(title_nodes[0]).strip() if title_nodes else None
        tokens = normalize_text(title, "hu", tokenizer, stop_words) if title else None
        norm_title = " ".join(tokens) if tokens else None
    except Exception:
        title, norm_title = None, None

    # Process collection
    try:
        coll_path = "//tei:teiHeader//tei:sourceDesc//tei:relatedItem[@type='copyText']//tei:title/text()"
        coll_nodes = poem_tree.xpath(coll_path, namespaces=NS_TEI)
        collection = str(coll_nodes[0]).strip() if coll_nodes else None
        tokens = normalize_text(collection, "hu", tokenizer, stop_words) if collection else None
        norm_coll = " ".join(tokens) if tokens else None
    except Exception:
        collection, norm_coll = None, None

    # Process author to get "Name Surname" format
    try:
        fname = poem_tree.xpath("//tei:teiHeader//tei:author/tei:persName/tei:forename/text()", namespaces=NS_TEI)
        sname = poem_tree.xpath("//tei:teiHeader//tei:author/tei:persName/tei:surname/text()", namespaces=NS_TEI)
        
        name = str(fname[0]).strip() if fname else None
        surname = str(sname[0]).strip() if sname else None
        
        author = " ".join(filter(None, [name, surname])) or None
    except Exception:
        author = None
    
    return title, norm_title, collection, norm_coll, author


def count_feet(syllable_pattern, meter_type):
    """Calculates foot count based on syllable length and meter type."""
    if not meter_type or not syllable_pattern:
        return 0
    
    # Get divisor based on foot type, default to 2
    length = FOOT_LENGTHS.get(meter_type.lower(), 2)
    return round(len(syllable_pattern) / length)


def extract_feet_and_type(poem_tree, base_meter_type):
    """Extract meter fouut and foot counts"""
    lines = poem_tree.xpath("//text/body/div/lg/l")
    meter_types_found, feet_counts = [], []

    for line in lines:
        pattern = line.get("real")
        if not pattern:
            continue
            
        if not base_meter_type:
            # If poem-level meter is missing, infer from line pattern
            m_type, f_count = get_meter_foot_and_foot_count(pattern)
            meter_types_found.append(m_type)
        else:
            # Use poem-level meter to calculate exact feet
            f_count = count_feet(pattern, base_meter_type)

        if f_count != 0:
            feet_counts.append(f_count)

    return feet_counts, meter_types_found

def get_features(poem_tree):
    """Extracts poetic features."""
    root_div = poem_tree.xpath("//text/body/div[@type='poem']")
    root = root_div[0]

    # Stanza scheme extraction
    stanzas = poem_tree.xpath("//text/body/div[@type='poem']/lg")
    s_list = [int(s.get('lg_numLine')) for s in stanzas if s.get('lg_numLine')]
    stanza_scheme = " ".join(map(str, s_list)) if s_list else None

    # Rhyme scheme extraction
    rhyme_attr = root.get('div_rhyme')
    rhyme_scheme = " ".join(rhyme_scheme_to_numbers(rhyme_attr)) if rhyme_attr else None

    st_rhyme_scheme = make_rhyme_scheme_per_stanza(rhyme_scheme, s_list) if (rhyme_scheme and s_list) else None

    # Meter extraction
    m_raw = root.get('met_quan') or root.get('met_qual')
    m_type_set = str(convert_meter_type(m_raw, meter_map)) if m_raw else None
    feet, line_m_types = extract_feet_and_type(poem_tree, m_raw)
    
    if not m_type_set and line_m_types:
        most_common_type = Counter(line_m_types).most_common(1)[0][0]
        m_type_set = str(convert_meter_type(most_common_type, meter_map))

    # Determine dominant Foot Count (Leader must double the runner-up)
    m_feet_set = get_dominant_feature(feet)
    
    # Syllable counts extraction
    line_count = int(root.get('div_numLine', 0))
    total_syll = int(root.get('div_numSyll', 0))
    avg_syll = total_syll / line_count if line_count > 0 else 0.0

    return (stanza_scheme, rhyme_scheme, st_rhyme_scheme, m_type_set, 
            m_feet_set, line_count, total_syll, avg_syll, None)


def get_all_features(filepath):
    """Parse XML and aggregate all features."""
    tree = etree.parse(filepath)
    text, normalized_text = get_primary_data(tree)
    title, normalized_title, collection, normalized_collection, author= get_metadata(tree)
    (stanza_scheme_str, rhyme_str, rhyme_per_stanza, 
     meter_foot, meter_foot_count, num_lines, 
     total_syll, mean_syll, form) = get_features(tree)

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

def process_all_poems(repository, output_file):
    """Extracts poem records and writes results to a JSONL file."""
    with open(output_file, "a", encoding="utf-8") as f:
        for subdir, _, files in os.walk(repository):
            for filename in files:
                if not filename.endswith('.xml'): 
                    continue 
                
                filepath = os.path.join(subdir, filename)
                try:
                    data = get_all_features(filepath)
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    tokenizer = ToktokTokenizer()
    stop_words = get_stop_words('hungarian')
    
    meter_map = {
        'iambic': 'iambic', 
        'trochaic': 'trochaic', 
        'dactylic': 'dactylic', 
        'anapestic': 'anapestic'
    }

    output_file = "hungarian_data/hungarian_poems_processed.jsonl"
    open(output_file, "w", encoding="utf-8").close()

    process_all_poems("../level4", output_file)