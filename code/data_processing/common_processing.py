import re
import simplemma
from collections import Counter

NON_ALPHANUM_PATTERN = re.compile(r'[^\w\s\u00C0-\u017F]')
ROMAN_NUMERAL_PATTERN = re.compile(r'\b(?:[ivxlcdm]{1,10})\b', re.IGNORECASE)

def normalize_text(text, lang, tokenizer, stop_words):
    """
    Removes non-alphanum characters and roman numerals, tokenizes, lemmatizes text and removes stop words.
    """
    if text is None or text == "":
        return []

    # Clean the string before tokenization
    clean_text = str(text).lower()
    clean_text = NON_ALPHANUM_PATTERN.sub('', clean_text)
    clean_text = ROMAN_NUMERAL_PATTERN.sub('', clean_text)

    # Convert to tokens and lemmatize
    raw_tokens = tokenizer.tokenize(clean_text)
    lemmatized = [simplemma.lemmatize(t, lang) for t in raw_tokens]

    # Filter out empty strings and stop words
    return [t for t in lemmatized if t and t not in stop_words]

def make_rhyme_scheme_per_stanza(rhyme_scheme, stanza_scheme):
    """
    Groups a rhyme scheme by stanza blocks, e.g., "A A B B", "2 2" -> "AA BB"
    """
    if not rhyme_scheme or not stanza_scheme:
        return None

    rhyme_list = rhyme_scheme.split()
    stanza_blocks = []
    pointer = 0

    for length in stanza_scheme:
        stanza_segment = rhyme_list[pointer : pointer + length]
        stanza_blocks.append("".join(stanza_segment))
        pointer += length

    return " ".join(stanza_blocks).strip()


def convert_meter_type(meter_name, meter_mapping):
    """
    Maps not-unified meter names using a corpus-specific map with a fallback to the original meter value.
    """
    # Check if we have a predefined name for this meter
    if meter_name in meter_mapping:
        mapped_value = meter_mapping[meter_name]
        return None if mapped_value == "unknown" else mapped_value

    # Default: return a cleaned version of the input
    return meter_name.strip().lower()

def rhyme_scheme_to_numbers(rhyme_string):
    """
    Converts letter rhymes (AABB) to numeric (1122).
    """
    if not rhyme_string:
        return []

    stanzas = rhyme_string.split('|')
    encoded_output = []

    for stanza in stanzas:
        symbol_to_id = {}
        next_id = 1
        
        for char in stanza:
            if char not in symbol_to_id:
                symbol_to_id[char] = next_id
                next_id += 1
            encoded_output.append(str(symbol_to_id[char]))
                
    return encoded_output
    
def get_meter_foot_and_foot_count(stress_pattern):
    if not stress_pattern or len(stress_pattern) < 2:
        return None, None
        
    syllables = len(stress_pattern)

    patterns = {
        'anapestic': '001',
        'dactylic': '100',
        'iambic': '01',
        'trochaic': '10',
        'amphibrachic': '010' 
    }
    
    scores = {}
    for name, unit in patterns.items():
        template = (unit * (syllables // len(unit) + 1))[:syllables]
        
        # Calculate how many positions match
        matches = sum(1 for a, b in zip(stress_pattern, template) if a == b)
        scores[name] = matches / syllables

    best_meter = max(scores, key=scores.get)
    
    if scores[best_meter] > 0.7:
        foot_len = 3 if best_meter in ['anapestic', 'dactylic', 'amphibrachic'] else 2
        return best_meter, round(syllables / foot_len)
    
    return None, None


def get_dominant_feature(feature_list, is_meter=False):
    """
    Helper to find the dominant feature. 
    If the most common item is at least 2x more frequent than the runner-up, 
    it returns that item. Otherwise, returns 'mixed' for meter_foot_count or 'polymetric' for meter_foot.
    """
    if not feature_list:
        return None

    counts = Counter(feature_list)
    most_common = counts.most_common(2)

    if not most_common:
        return None

    # Only one type of feature exists
    if len(most_common) == 1:
        return most_common[0][0]

    # Check the 2x frequency rule for dominance
    leader_label, leader_count = most_common[0]
    _, runner_up_count = most_common[1]

    if leader_count >= (runner_up_count * 2):
        return str(leader_label)
    
    return "polymetric" if is_meter else "mixed"
