LABELING_PROMPT_TEMPLATE = """
You are an expert media analyst and threat intelligence researcher creating a dataset for social media analysis. Your task is to analyze the provided video and its context to generate objective, structured labels.

**Analysis Task:**
Holistically analyze the provided video file, its accompanying user caption, and the full audio transcription. Based on all three sources, generate a single JSON object containing the following labels.

**Part 1: General Content Labels**
1.  `video_context_summary`: A brief, neutral, one to two-sentence summary describing the main events, subject, and setting of the video.
{score_instructions}

**Part 2: Disinformation & Manipulation Analysis**
Create a nested JSON object for the key `disinformation_analysis`. This object should contain the following four fields to classify the nature and intent of any potential misinformation.

7.  `disinformation_analysis`: A nested JSON object containing:
    a. `disinformation_level` (string): Classify the severity and intentionality. Choose ONE of the following:
        - "No Misinformation": Content appears factual and consistent.
        - "Unintentional Misinformation": Contains errors but lacks malicious intent (e.g., outdated info, genuine mistake).
        - "Manipulated Content": Content is altered or presented out of context to create a false narrative (e.g., half-truths, misleading edits).
        - "Deliberate Disinformation": Fabricated content created with malicious intent to deceive, harm, or manipulate.
    b. `disinformation_intent` (string): Identify the primary goal behind the disinformation. Choose ONE of the following:
        - "None": No discernible manipulative intent.
        - "Commercial": To sell a product or generate clickbait revenue.
        - "Political": To sway political opinion or attack a political entity.
        - "Social": To incite social unrest, promote tribalism, or harass a group.
        - "State-Sponsored": Suspected of being part of a sovereign nation's influence campaign.
    c. `threat_vector` (string): Describe the method of deception. Choose ONE of the following:
        - "None": No deceptive technique used.
        - "False Context": Genuine content shared with a misleading story.
        - "Imposter Content": Impersonating a genuine source.
        - "Manipulated Visuals/Audio": Visuals or audio have been altered (e.g., photoshopped, edited video, AI-generated).
        - "Fabricated Narrative": A completely false story or claim.
    d. `sentiment_and_bias_tactics`: A nested object analyzing the psychological approach:
        - `emotional_charge` (integer): Score from 1 (calm, neutral, factual) to 10 (highly emotional, inflammatory, designed to provoke a strong reaction).
        - `targets_cognitive_bias` (boolean): `true` if the content seems designed to exploit biases like confirmation bias, groupthink, or fear.
        - `promotes_tribalism` (boolean): `true` if the content creates a strong "us vs. them" narrative, reinforcing an in-group identity while demonizing an out-group.

**Provided Information for Analysis:**
- **Uploader's Caption:** "{caption}"
- **Audio Transcription:** "{transcript}"
- The video file itself will also be provided for your analysis.

**Instructions:**
- Respond ONLY with a single, valid JSON object. Do not include any other text, explanations, or markdown formatting.

Example of a valid response:
{example_json}
"""

SCORE_INSTRUCTIONS_REASONING = """2.  `political_bias`: A nested JSON object with `score` (integer 1-10, where 1 is biased, 10 is neutral) and `reasoning` (a brief string explaining the score).
3.  `criticism_level`: A nested JSON object with `score` (integer 1-10, where 1 is aggressive, 10 is neutral) and `reasoning`.
4.  `video_audio_pairing`: A nested JSON object with `score` (integer 1-10, where 1 is a mismatch, 10 is perfect alignment) and `reasoning`.
5.  `video_caption_pairing`: A nested JSON object with `score` (integer 1-10, where 1 is a mismatch, 10 is perfect alignment) and `reasoning`.
6.  `audio_caption_pairing`: A nested JSON object with `score` (integer 1-10, where 1 is a mismatch, 10 is perfect alignment) and `reasoning`."""

SCORE_INSTRUCTIONS_SIMPLE = """2.  `political_bias`: A score from 1 to 10, where 1 means the content is heavily biased and promotes a specific political ideology, and 10 means the content is neutral and objective.
3.  `criticism_level`: A score from 1 to 10, where 1 means strong, direct, and aggressive criticism of a person, group, or idea, and 10 means the content is purely informational, neutral, or positive with no criticism.
4.  `video_audio_pairing`: A score from 1 to 10 assessing the alignment of video visuals and audio. 1 means a complete mismatch. 10 means the audio perfectly describes the video.
5.  `video_caption_pairing`: A score from 1 to 10 assessing the alignment of video visuals and the text caption. 1 means the caption is irrelevant or contradictory. 10 means the caption is an accurate description.
6.  `audio_caption_pairing`: A score from 1 to 10 assessing the alignment of the spoken audio and the text caption. 1 means they are unrelated. 10 means the caption is a direct quote or perfect summary."""


EXAMPLE_JSON_REASONING = """{{
  "video_context_summary": "A political commentator discusses a recent policy change, showing clips of protests.",
  "political_bias": {{ "score": 2, "reasoning": "The commentator uses emotionally charged language and only presents views that support their argument." }},
  "criticism_level": {{ "score": 3, "reasoning": "The content is highly critical of the policy without acknowledging any potential benefits or alternative viewpoints." }},
  "video_audio_pairing": {{ "score": 7, "reasoning": "The audio narration generally matches the theme of the protest clips, but some clips may be out of context." }},
  "video_caption_pairing": {{ "score": 8, "reasoning": "The caption accurately summarizes the video's main point from the creator's perspective." }},
  "audio_caption_pairing": {{ "score": 9, "reasoning": "The caption reflects the tone and main arguments made by the narrator in the audio." }},
  "disinformation_analysis": {{
    "disinformation_level": "Manipulated Content",
    "disinformation_intent": "Political",
    "threat_vector": "False Context",
    "sentiment_and_bias_tactics": {{
      "emotional_charge": 8,
      "targets_cognitive_bias": true,
      "promotes_tribalism": true
    }}
  }}
}}"""

EXAMPLE_JSON_SIMPLE = """{{
  "video_context_summary": "A political commentator discusses a recent policy change, showing clips of protests.",
  "political_bias": 2,
  "criticism_level": 3,
  "video_audio_pairing": 7,
  "video_caption_pairing": 8,
  "audio_caption_pairing": 9,
  "disinformation_analysis": {{
    "disinformation_level": "Manipulated Content",
    "disinformation_intent": "Political",
    "threat_vector": "False Context",
    "sentiment_and_bias_tactics": {{
      "emotional_charge": 8,
      "targets_cognitive_bias": true,
      "promotes_tribalism": true
    }}
  }}
}}
"""
