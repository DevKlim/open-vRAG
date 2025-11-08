LABELING_PROMPT_TEMPLATE = """
You are an expert media analyst creating a dataset for social media research. Your task is to analyze the provided video and its context to generate objective, structured labels.

**Analysis Task:**
Holistically analyze the provided video file, its accompanying user caption, and the full audio transcription. Based on all three sources, generate a single JSON object containing the following seven labels.

**Label Definitions:**
1.  `video_context_summary`: A brief, neutral, one to two-sentence summary describing the main events, subject, and setting of the video.
2.  `political_bias`: A score from 1 (content is heavily biased, promoting a specific political ideology) to 10 (content is neutral and objective).
3.  `is_misleading`: A boolean value (`true` or `false`). `true` if the content (video, audio, or caption) appears to be intentionally deceptive, factually incorrect, or presented out of context to mislead the viewer.
4.  `criticism_level`: A score from 1 (strong, direct, and aggressive criticism of a person, group, or idea) to 10 (no criticism; content is purely informational, neutral, or positive).
5.  `video_audio_pairing`: A score assessing the relationship between the video's visuals and its audio track (the spoken words). A score of 1 means a complete mismatch (e.g., audio about cooking over video of a football game). A score of 10 means the audio perfectly describes or narrates the events in the video.
6.  `video_caption_pairing`: A score assessing the relationship between the video's visuals and the uploader's text caption. A score of 1 means the caption is irrelevant or contradictory to the video. A score of 10 means the caption is a direct and accurate description of the video.
7.  `audio_caption_pairing`: A score assessing the relationship between the audio track (spoken words) and the uploader's text caption. A score of 1 means they are unrelated. A score of 10 means the caption is a direct quote from the audio or a perfect summary of what was said.

**Provided Information for Analysis:**
- **Uploader's Caption:** "{caption}"
- **Audio Transcription:** "{transcript}"
- The video file itself will also be provided for your analysis.

**Instructions:**
- Evaluate all provided materials holistically.
- Respond ONLY with a single, valid JSON object containing the seven keys defined above. Do not include any other text, explanations, or markdown formatting.

Example of a valid response:
{{
  "video_context_summary": "A chef is demonstrating how to properly chop an onion in a well-lit kitchen, explaining each step of the process clearly.",
  "political_bias": 8,
  "is_misleading": false,
  "criticism_level": 9,
  "video_audio_pairing": 10,
  "video_caption_pairing": 7,
  "audio_caption_pairing": 2
}}
"""
