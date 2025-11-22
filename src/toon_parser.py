# toon_parser.py
import re
import logging
import csv
from io import StringIO

logger = logging.getLogger(__name__)

def parse_toon_line(line_def, data_line):
    """
    Parses a single TOON data line based on headers.
    Handles CSV-style quoting for text fields.
    """
    if not data_line or data_line.isspace():
        return {}

    try:
        # Use csv module to handle quoted strings
        reader = csv.reader(StringIO(data_line), skipinitialspace=True)
        try:
            values = next(reader)
        except StopIteration:
            values = []
        
        # CLEANUP: Strip parentheses if the model output numbers as (9) instead of 9
        values = [v.strip().replace('(', '').replace(')', '') for v in values]

        headers = line_def.get('headers', [])
        
        # Ensure values match headers length if possible, or pad
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        elif len(values) > len(headers):
            values = values[:len(headers)]

        return dict(zip(headers, values))
    except Exception as e:
        logger.error(f"Error parsing TOON line '{data_line}': {e}")
        return {}

def fuzzy_extract_scores(text: str) -> dict:
    """
    Fallback method. Scans text for key metrics followed near-immediately by a number.
    """
    scores = {
        'visual': '0', 'audio': '0', 'source': '0', 'logic': '0', 'emotion': '0',
        'video_audio': '0', 'video_caption': '0', 'audio_caption': '0'
    }
    
    # Mappings: Regex Pattern -> Score Key
    mappings = [
        ('visual', 'visual'),
        ('audio', 'audio'),
        ('source', 'source'),
        ('logic', 'logic'),
        ('emotion', 'emotion'),
        (r'video.*?audio', 'video_audio'),
        (r'video.*?caption', 'video_caption'),
        (r'audio.*?caption', 'audio_caption')
    ]

    for pattern_str, key in mappings:
        # Look for pattern, optional separators (: or - or space), then a number 1-10 or 100.
        # Handles cases like: "Visual: 9", "Visual - 9", "Visual (9)"
        pattern = re.compile(fr'(?i){pattern_str}\s*[:=\-\s\(]+\s*(\b10\b|[0-9])')
        match = pattern.search(text)
        if match:
            scores[key] = match.group(1)
    
    return scores

def parse_veracity_toon(text: str) -> dict:
    """
    Parses the Veracity Vector TOON output into a standardized dictionary.
    Handles "Simple", "Reasoning", and new "Modalities" blocks.
    Robust against Markdown formatting artifacts.
    """
    if not text:
        return {}

    # 1. Cleanup
    text = re.sub(r'```\w*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    parsed_sections = {}

    # 2. Relaxed Regex for TOON Block Headers
    # Matches: key : type [ count ] { headers } :
    # Made [count] optional to handle model hallucinations
    block_pattern = re.compile(
        r'^\s*([a-zA-Z0-9_]+)\s*:\s*(?:\w+\s*)?(?:\[\s*(\d+)\s*\])?\s*\{\s*(.*?)\s*\}\s*:\s*', 
        re.MULTILINE
    )
    
    matches = list(block_pattern.finditer(text))
    
    for i, match in enumerate(matches):
        key = match.group(1).lower()
        # Default to 1 if count is missing
        count = int(match.group(2)) if match.group(2) else 1
        headers_str = match.group(3)
        headers = [h.strip().lower() for h in headers_str.split(',')]
        
        start_idx = match.end()
        # End at next match or end of text
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
        block_content = text[start_idx:end_idx].strip()
        
        lines = [line.strip() for line in block_content.splitlines() if line.strip()]
        
        data_items = []
        for line in lines[:count]:
            item = parse_toon_line({'key': key, 'headers': headers}, line)
            data_items.append(item)
            
        if count == 1 and data_items:
            parsed_sections[key] = data_items[0]
        else:
            parsed_sections[key] = data_items

    # --- Flatten logic to standardized structure ---
    flat_result = {
        'veracity_vectors': {
            'visual_integrity_score': '0',
            'audio_integrity_score': '0',
            'source_credibility_score': '0',
            'logical_consistency_score': '0',
            'emotional_manipulation_score': '0'
        },
        'modalities': {
            'video_audio_score': '0',
            'video_caption_score': '0',
            'audio_caption_score': '0'
        },
        'video_context_summary': '',
        'factuality_factors': {},
        'disinformation_analysis': {},
        'final_assessment': {}
    }
    
    got_vectors = False
    got_modalities = False

    # 1. Process 'vectors'
    vectors_data = parsed_sections.get('vectors', [])
    if isinstance(vectors_data, dict): # Simple schema
        v = vectors_data
        if any(val and val != '0' for val in v.values()):
            flat_result['veracity_vectors']['visual_integrity_score'] = v.get('visual', '0')
            flat_result['veracity_vectors']['audio_integrity_score'] = v.get('audio', '0')
            flat_result['veracity_vectors']['source_credibility_score'] = v.get('source', '0')
            flat_result['veracity_vectors']['logical_consistency_score'] = v.get('logic', '0')
            flat_result['veracity_vectors']['emotional_manipulation_score'] = v.get('emotion', '0')
            got_vectors = True
    elif isinstance(vectors_data, list): # Reasoning schema
        for item in vectors_data:
            cat = item.get('category', '').lower()
            score = item.get('score', '0')
            if score and score != '0': 
                got_vectors = True
            if 'visual' in cat: flat_result['veracity_vectors']['visual_integrity_score'] = score
            elif 'audio' in cat: flat_result['veracity_vectors']['audio_integrity_score'] = score
            elif 'source' in cat: flat_result['veracity_vectors']['source_credibility_score'] = score
            elif 'logic' in cat: flat_result['veracity_vectors']['logical_consistency_score'] = score
            elif 'emotion' in cat: flat_result['veracity_vectors']['emotional_manipulation_score'] = score

    # 2. Process 'modalities'
    modalities_data = parsed_sections.get('modalities', [])
    if isinstance(modalities_data, dict): # Simple schema
        m = modalities_data
        if 'video_audio_score' in m: flat_result['modalities']['video_audio_score'] = m['video_audio_score']
        if 'video_caption_score' in m: flat_result['modalities']['video_caption_score'] = m['video_caption_score']
        if 'audio_caption_score' in m: flat_result['modalities']['audio_caption_score'] = m['audio_caption_score']
        got_modalities = True
    elif isinstance(modalities_data, list): # Reasoning schema
        for item in modalities_data:
            cat = item.get('category', '').lower().replace(' ', '')
            score = item.get('score', '0')
            if score and score != '0':
                got_modalities = True
            # Check against condensed category names
            if 'videoaudio' in cat: flat_result['modalities']['video_audio_score'] = score
            elif 'videocaption' in cat: flat_result['modalities']['video_caption_score'] = score
            elif 'audiocaption' in cat: flat_result['modalities']['audio_caption_score'] = score

    # --- FUZZY FALLBACK ---
    if not got_vectors or not got_modalities:
        fuzzy_scores = fuzzy_extract_scores(text)
        
        if not got_vectors:
            # Only warn if we genuinely had to use fuzzy because structural failed
            logger.warning("Vectors missing in TOON. Applying Fuzzy Fallback.")
            flat_result['veracity_vectors']['visual_integrity_score'] = fuzzy_scores['visual']
            flat_result['veracity_vectors']['audio_integrity_score'] = fuzzy_scores['audio']
            flat_result['veracity_vectors']['source_credibility_score'] = fuzzy_scores['source']
            flat_result['veracity_vectors']['logical_consistency_score'] = fuzzy_scores['logic']
            flat_result['veracity_vectors']['emotional_manipulation_score'] = fuzzy_scores['emotion']
            
        if not got_modalities:
            logger.warning("Modalities missing in TOON. Applying Fuzzy Fallback.")
            flat_result['modalities']['video_audio_score'] = fuzzy_scores['video_audio']
            flat_result['modalities']['video_caption_score'] = fuzzy_scores['video_caption']
            flat_result['modalities']['audio_caption_score'] = fuzzy_scores['audio_caption']

    # 3. Factuality
    f = parsed_sections.get('factuality', {})
    if isinstance(f, list): f = f[0] if f else {}
    flat_result['factuality_factors'] = {
        'claim_accuracy': f.get('accuracy', 'Unverifiable'),
        'evidence_gap': f.get('gap', ''),
        'grounding_check': f.get('grounding', '')
    }

    # 4. Disinfo
    d = parsed_sections.get('disinfo', {})
    if isinstance(d, list): d = d[0] if d else {}
    flat_result['disinformation_analysis'] = {
        'classification': d.get('class', 'None'),
        'intent': d.get('intent', 'None'),
        'threat_vector': d.get('threat', 'None')
    }

    # 5. Final Assessment
    fn = parsed_sections.get('final', {})
    if isinstance(fn, list): fn = fn[0] if fn else {}
    flat_result['final_assessment'] = {
        'veracity_score_total': fn.get('score', '0'),
        'reasoning': fn.get('reasoning', '')
    }

    # 6. Summary
    s = parsed_sections.get('summary', {})
    if isinstance(s, list): s = s[0] if s else {}
    flat_result['video_context_summary'] = s.get('text', '')

    flat_result['raw_parsed_structure'] = parsed_sections
    
    return flat_result