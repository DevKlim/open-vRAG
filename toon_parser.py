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
    # Pre-processing: Clean the data_line if it contains the schema definition again
    # The logs showed values like: "scores{visual, audio...}: 9, 9..."
    # We strip anything looking like "word[...]{...}:" or "word{...}:" from the start
    
    # Regex to remove "type{headers}:" or "type[count]{headers}:" prefix
    # This matches "scores{a,b}:" or "scores[5]{a,b}:" followed by optional space
    clean_pattern = re.compile(r'^\s*\w+(?:\[\d+\])?\{.*?\}:\s*')
    data_line = clean_pattern.sub('', data_line)

    try:
        # Use csv module to handle quoted strings (e.g., "This is a summary, with commas")
        reader = csv.reader(StringIO(data_line), skipinitialspace=True)
        try:
            values = next(reader)
        except StopIteration:
            values = []
        
        headers = line_def['headers']
        
        # Basic validation
        if len(values) != len(headers):
            logger.warning(f"TOON Mismatch: Expected {len(headers)} values for {line_def['key']}, got {len(values)}. Headers: {headers}, Values: {values}")
            # Attempt to map what we have, or pad
            if len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            else:
                values = values[:len(headers)]

        return dict(zip(headers, values))
    except Exception as e:
        logger.error(f"Error parsing TOON line '{data_line}': {e}")
        return {}

def parse_veracity_toon(text: str) -> dict:
    """
    Parses the full Veracity Vector TOON output into a flat dictionary 
    compatible with the project's CSV structure.
    
    This function uses robust Regex to handle model hallucinations, specifically
    handling 'function call' style outputs like `scores(...)`.
    """
    # Clean text: remove markdown code blocks if present
    text = re.sub(r'```\w*\n', '', text)
    text = re.sub(r'```', '', text)
    
    matches = []

    # 1. Standard TOON Definition Regex
    # Matches: Start of line(optional space) -> Key -> : -> type[count]{headers}: -> Newline -> Value
    standard_pattern = re.compile(r'^\s*(\w+):\s*\w+\[(\d+)\]\{(.*?)\}:\s*\n(.+)', re.MULTILINE)
    matches.extend(standard_pattern.findall(text))

    # 2. Relaxed "Key: Value" Regex (for models ignoring strict TOON but providing Keys)
    # Key can be summary, vectors, factuality, disinfo, final
    if not matches:
         relaxed_pattern = re.compile(r'^\s*(summary|vectors|factuality|disinfo|final)(?:.*?:|:\s*)(.+)$', re.MULTILINE)
         relaxed_matches = relaxed_pattern.findall(text)
         for key, value_line in relaxed_matches:
             # Map keys to default headers
             if key == 'summary': headers_str = 'text'
             elif key == 'vectors': headers_str = 'visual,audio,source,logic,emotion'
             elif key == 'factuality': headers_str = 'accuracy,gap,grounding'
             elif key == 'disinfo': headers_str = 'class,intent,threat'
             elif key == 'final': headers_str = 'score,reasoning'
             else: continue
             matches.append((key, '0', headers_str, value_line))

    # 3. Function Call / CSV-ish Regex (Handle logs like `scores(9,9,8,7,7)`)
    # This handles the specific "grounding" error observed in logs
    if not matches:
        # Map discovered function-like names to internal keys & headers
        func_map = {
            'scores': ('vectors', 'visual,audio,source,logic,emotion'),
            'factors': ('factuality', 'accuracy,gap,grounding'),
            'analysis': ('disinfo', 'class,intent,threat'),
            'assessment': ('final', 'score,reasoning'),
            'text': ('summary', 'text') # sometimes models output text("...")
        }
        
        # Regex to catch name(arg1, arg2...)
        # Note: This is a greedy regex inside parentheses, relying on the fact that line breaks usually separate these
        func_pattern = re.compile(r'^\s*(\w+)\((.+)\)', re.MULTILINE)
        func_matches = func_pattern.findall(text)
        
        for func_name, content in func_matches:
            if func_name in func_map:
                key, headers_str = func_map[func_name]
                matches.append((key, '0', headers_str, content))

    temp_storage = {}

    for key, count, headers_str, value_line in matches:
        headers = [h.strip() for h in headers_str.split(',')]
        data_map = parse_toon_line({'key': key, 'headers': headers}, value_line.strip())
        temp_storage[key] = data_map

    # Flatten the structure for the CSV
    flat_result = {}
    
    # Mapping TOON keys to CSV columns
    
    # 1. Vectors
    if 'vectors' in temp_storage:
        v = temp_storage['vectors']
        flat_result['veracity_vectors'] = {
            'visual_integrity_score': v.get('visual', '0'),
            'audio_integrity_score': v.get('audio', '0'),
            'source_credibility_score': v.get('source', '0'),
            'logical_consistency_score': v.get('logic', '0'),
            'emotional_manipulation_score': v.get('emotion', '0')
        }

    # 2. Factuality
    if 'factuality' in temp_storage:
        f = temp_storage['factuality']
        flat_result['factuality_factors'] = {
            'claim_accuracy': f.get('accuracy', 'Unverifiable'),
            'evidence_gap': f.get('gap', ''),
            'grounding_check': f.get('grounding', '')
        }

    # 3. Disinfo
    if 'disinfo' in temp_storage:
        d = temp_storage['disinfo']
        flat_result['disinformation_analysis'] = {
            'classification': d.get('class', 'None'),
            'intent': d.get('intent', 'None'),
            'threat_vector': d.get('threat', 'None')
        }

    # 4. Final
    if 'final' in temp_storage:
        fn = temp_storage['final']
        flat_result['final_assessment'] = {
            'veracity_score_total': fn.get('score', '0'),
            'reasoning': fn.get('reasoning', '')
        }
        
    # 5. Context (Summary)
    if 'summary' in temp_storage:
        flat_result['video_context_summary'] = temp_storage['summary'].get('text', '')
    elif 'context' in temp_storage:
         flat_result['video_context_summary'] = temp_storage['context'].get('text', '')
    
    # Fallback if parsing completely failed but text exists
    if not flat_result and text:
        logger.warning("TOON parsing yielded empty results. Returning raw text in summary.")
        # Clean up raw text (remove thinking tags if they exist to save space)
        clean_reasoning = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
        if not clean_reasoning: clean_reasoning = text[:1000]
        
        flat_result['final_assessment'] = {'reasoning': clean_reasoning}

    return flat_result