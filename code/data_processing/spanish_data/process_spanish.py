import json
import os
import re
import nltk
from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from rhymetagger import RhymeTagger

from common_processing import (normalize_text, make_rhyme_scheme_per_stanza, convert_meter_type, 
    get_meter_foot_and_foot_count, get_dominant_feature, rhyme_scheme_to_numbers)

TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def get_primary_data(poem_tree, tokenizer, stop_words):
    """Extracts raw text and its normalized version."""
    try:
        lines = poem_tree.xpath('//tei:l', namespaces=TEI_NS)
        text_lines = [l.xpath('string(.)').strip() for l in lines if l.xpath('string(.)')]
        
        text = "\n".join(text_lines) if text_lines else None

        tokens = normalize_text(text, "es", tokenizer, stop_words) if text else None
        normalized_text = " ".join(tokens) if tokens else None    
        return text, normalized_text
    except Exception:
        return None, None

def get_features(poem_tree, corpus, rt, meter_map):
    """Extracts poetic features."""
    stanza_scheme, syllable_counts, meter_types= [], [], []
    meter_foot_counts, rhyme_groups, text_lines = [], [], []

    line_groups = poem_tree.xpath('//tei:lg', namespaces=TEI_NS)
    lines = poem_tree.xpath('//tei:l', namespaces=TEI_NS)

    # Stanza extraction
    for lg in line_groups:
        lines_in_stanza = lg.xpath('./tei:l', namespaces=TEI_NS)
        if lines_in_stanza:
            stanza_scheme.append(len(lines_in_stanza))
            if corpus == "DISCO":
                stanza_rhymes = lg.xpath('./tei:l/@rhyme', namespaces=TEI_NS)
                if stanza_rhymes:
                    rhyme_groups.append("".join(stanza_rhymes))

    # Metrical analysis
    for l in lines:
        line_text = l.xpath('string(.)').strip()
        if line_text:
            text_lines.append(line_text)

        met = l.get('met') 
        if met:
            syllable_counts.append(len(met))
            bin_metre = met.replace('+', '1').replace('-', '0')
            m_type, feet = get_meter_foot_and_foot_count(bin_metre)

            if m_type:
                meter_types.append(convert_meter_type(m_type, meter_map))
            if feet:
                if type(feet) == int:
                    meter_foot_counts.append(feet)
                else:
                    for foot in feet:
                        meter_foot_counts.append(foot)

    # Rhyme scheme extraction
    try:
        if corpus == "CSSDO":
            rhymes = rt.tag(text_lines, output_format=3)
            rhyme_list = [str(x) for x in rhymes]
            max_val = max([int(x) for x in rhyme_list if x != "None"], default=0) + 1
            rhyme_scheme = [str(max_val + i) if x == "None" else x for i, x in enumerate(rhyme_list)]
        else:
            raw_rhyme = "|".join(rhyme_groups) if rhyme_groups else None
            rhyme_scheme = rhyme_scheme_to_numbers(raw_rhyme) if raw_rhyme else None
        
        rhyme_str = " ".join(rhyme_scheme) if rhyme_scheme else None
    except Exception:
        rhyme_str = None

    meter_foot_count = get_dominant_feature(meter_foot_counts, is_meter=False)
    meter_type_set = get_dominant_feature(meter_types, is_meter=True)

    stanza_rhyme_scheme = make_rhyme_scheme_per_stanza(rhyme_str, stanza_scheme) if rhyme_str and stanza_scheme else None
    stanza_scheme_str = " ".join(map(str, stanza_scheme)) if stanza_scheme else None

    num_lines = len(text_lines)
    total_syll = sum(syllable_counts)
    mean_syll = total_syll / num_lines if num_lines > 0 else 0.0

    return (stanza_scheme_str, rhyme_str, stanza_rhyme_scheme, meter_type_set, 
            meter_foot_count, num_lines, total_syll, mean_syll, "sonnet")

def get_metadata(poem_tree, corpus, tokenizer, stop_words):
    """Extracts metadata."""
    try:
        # Title extraction
        if corpus == "CSSDO":
            title = poem_tree.find('.//tei:body/tei:head/tei:title', namespaces=TEI_NS).text
        else:
            t_node = poem_tree.xpath('//tei:title[@property="dc:title"]', namespaces=TEI_NS)
            title = t_node[0].text if t_node else None
        
        title = re.sub(r'\s+', ' ', str(title)).strip() if title else None
        tokens = normalize_text(title, "es", tokenizer, stop_words) if title else None
        norm_title = " ".join(tokens) if tokens else None

        # Collection extraction
        if corpus == "CSSDO":
            collection = poem_tree.find('.//tei:sourceDesc/tei:bibl/tei:publisher', namespaces=TEI_NS).text
        else:
            c_node = poem_tree.xpath('//tei:bibl', namespaces=TEI_NS)
            collection = c_node[0].text if c_node else None
        
        collection = re.sub(r'\s+', ' ', str(collection)).strip() if collection else None
        tokens = normalize_text(collection, "es", tokenizer, stop_words) if title else None
        norm_collection = " ".join(tokens) if tokens else None
        
        # Author formatting in format "Name Surname"
        if corpus == "CSSDO":      
            author = poem_tree.find('.//tei:sourceDesc/tei:bibl/tei:author', namespaces=TEI_NS).text
        else:
            a_node = poem_tree.xpath('//tei:author[@property="dc:creator"]', namespaces=TEI_NS)
            author = a_node[0].text if a_node else None
        
        if author:
            author = re.sub(r'\s+', ' ', str(author)).strip()
            if "," in author:
                parts = author.split(",")
                author = f"{parts[1].strip()} {parts[0].strip()}"
        
        return title, norm_title, collection, norm_collection, author
    except Exception:
        return None, None, None, None, None

def process_poem(filepath, corpus, rt, meter_map, tokenizer, stop_words):
    """Extracts all features for a single poem."""
    poem_tree = etree.parse(filepath)
    
    text, normalized_text = get_primary_data(poem_tree, tokenizer, stop_words)
    (stanza_scheme, rhyme_scheme, stanza_rhyme, meter_foot, 
     meter_foot_count, num_lines, total_syll, mean_syll, form) = get_features(poem_tree, corpus, rt, meter_map)
    title, normalized_title, collection, normalized_collection, author = get_metadata(poem_tree, corpus, tokenizer, stop_words)

    return {
        "title": title,
        "normalized_title": normalized_title,
        "author": author,
        "collection": collection,
        "normalized_collection": normalized_collection,
        "form": form,
        "stanza_scheme": stanza_scheme,
        "rhyme_scheme": rhyme_scheme,
        "rhyme_scheme_per_stanza": stanza_rhyme,
        "metrical_foot": meter_foot,
        "metrical_foot_count": meter_foot_count,
        "line_count": num_lines,
        "total_syllables": total_syll,
        "average_syllable_count": mean_syll,
        "text": text,
        "normalized_text": normalized_text
    }

def delete_duplicates(input_file, output_file):
    """Removes duplicates."""
    seen = set()
    unique_count, duplicate_count = 0, 0
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            fingerprint = data.get("normalized_text", "").strip()
            if fingerprint and fingerprint not in seen:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                seen.add(fingerprint)
                unique_count += 1
            else:
                duplicate_count += 1
    print(f"Unique: {unique_count}, Removed: {duplicate_count}")

def process_all_poems(directory, output, corpus_type, rt, meter_map, tokenizer, stop_words):
    """Extracts features for all poems."""
    with open(output, "a", encoding="utf-8") as out:
        for root, _, files in os.walk(directory):
            if corpus_type == "DISCO" and "per-sonnet" not in root:
                continue
            for filename in files:
                if filename.endswith(".xml"):
                    try:
                        data = process_poem(os.path.join(root, filename), corpus_type, rt, meter_map, tokenizer, stop_words)
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Error {filename}: {e}")

if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('spanish'))
    tokenizer = ToktokTokenizer()
    rt = RhymeTagger()
    rt.load_model(model='es')

    meter_map = {'iambic': 'iambic', 'trochaic': 'trochaic', 'dactylic': 'dactylic',
                 'amphibrachic': 'amphibrachic', 'anapestic': 'anapestic'}

    raw_output = "spanish_data/spanish_raw.jsonl"
    clean_output = "spanish_data/spanish_poems_processed.jsonl"
    
    open(raw_output, "w").close() # Clear file

    process_all_poems("../CorpusSonetosSigloDeOro", raw_output, "CSSDO", rt, meter_map, tokenizer, stop_words)
    process_all_poems("../disco/tei", raw_output, "DISCO", rt, meter_map, tokenizer, stop_words)

    delete_duplicates(raw_output, clean_output)