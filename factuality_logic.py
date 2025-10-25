import os
import re
import logging
import asyncio
from pathlib import Path
import inference_logic # Use direct import instead of relative

logger = logging.getLogger(__name__)

#  Prompts for Different Analysis Steps 

PROMPT_VISUAL_ARTIFACTS = (
    "Analyze the video for visual inconsistencies, artifacts, or evidence of digital manipulation. "
    "Pay close attention to the following:\n"
    "- **AI-Generated Content:** Look for common AI artifacts like waxy skin, strange physics, distorted hands or text, and unnatural movements.\n"
    "- **Editing & Cuts:** Identify abrupt cuts, jumps, or transitions that might be misleading.\n"
    "- **Watermarks/Logos:** Note any watermarks, especially those from AI tools or social media platforms that might indicate the video's origin or that it has been re-posted.\n"
    "- **Overall Authenticity:** Provide a summary of whether the video appears authentic or potentially manipulated.\n\n"
    "Present your findings in a structured manner."
)

PROMPT_CONTENT_ANALYSIS = (
    "Based on the video's visual content and the provided transcript, perform a content and credibility analysis. Address the following points:\n"
    "- **Bias and Objectivity:** Is the language neutral or does it show political, commercial, or other forms of bias? Explain your reasoning.\n"
    "- **Clickbait Assessment:** Does the content seem exaggerated or designed to provoke an emotional response for views (i.e., 'clickbait')? Look at the language and visual style.\n"
    "- **Sourcing and Credibility:** Does the video cite any sources for its claims? Does it appear to be from a credible source (e.g., established news organization, expert)?\n"
    "- **Sponsored Content:** Are there any signs that this video is a paid promotion or sponsored content? Look for explicit disclosures or product placements.\n\n"
    "Transcript for context:\n\n{transcript}\n\n\n"
    "Provide a detailed analysis for each point."
)

PROMPT_AUDIO_ANALYSIS = (
    "Analyze the video's audio track for anomalies that could suggest editing or manipulation. While you cannot process the audio directly, use the visual context of the video to infer potential audio issues. Consider:\n"
    "- **Audio-Visual Sync:** Do lip movements match the speech in the transcript? Are sounds synchronized with on-screen actions?\n"
    "- **Abrupt Changes:** Are there moments where the background noise or audio tone changes suddenly, suggesting a cut or edit?\n"
    "- **Transcript Consistency:** Does the flow of the transcript seem natural, or are there non-sequiturs and awkward pauses that might indicate parts have been removed?\n\n"
    "Transcript for context:\n\n{transcript}\n\n\n"
    "Based on your visual analysis and the transcript, report any suspected audio manipulations."
)

def parse_vtt(file_path: str) -> str:
    """Parses a .vtt subtitle file and returns the clean text content."""
    try:
        if not os.path.exists(file_path):
            return "Transcript file not found."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out VTT metadata (timestamps, styling, etc.)
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit():
                # Remove VTT tags like <c> or <v>
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line and (not text_lines or clean_line != text_lines[-1]):
                     text_lines.append(clean_line)
        
        return "\n".join(text_lines) if text_lines else "No speech found in transcript."
    except Exception as e:
        logger.error(f"Error parsing VTT file {file_path}: {e}")
        return f"Error reading transcript: {e}"

async def run_factuality_pipeline(paths: dict, checks: dict, generation_config: dict):
    """
    Asynchronously runs a pipeline of factuality checks on the video.
    Yields progress messages and analysis results.
    """
    video_path = paths.get("video")
    transcript_path = paths.get("transcript")
    audio_path = paths.get("audio")

    if not video_path:
        yield "ERROR: Video path not found. Cannot start analysis.\n\n"
        return

    #  1. Extract and Display Transcript 
    yield " Step 1: Processing Transcript \n"
    transcript = "No transcript was downloaded."
    if transcript_path:
        transcript = parse_vtt(transcript_path)
    else:
        yield "No transcript file was found for this video.\n"
    
    yield f"Extracted Transcript:\n\n{transcript}\n\n\n"
    await asyncio.sleep(0.1) # Allow UI to update

    analysis_steps = []
    if checks.get("visuals"):
        analysis_steps.append(("Visual Artifacts", PROMPT_VISUAL_ARTIFACTS))
    if checks.get("content"):
        analysis_steps.append(("Content & Credibility", PROMPT_CONTENT_ANALYSIS.format(transcript=transcript)))
    if checks.get("audio"):
        # The prompt for audio analysis is contextual based on visuals and transcript
        analysis_steps.append(("Audio Anomaly Detection", PROMPT_AUDIO_ANALYSIS.format(transcript=transcript)))

    #  2. Run Selected Analyses 
    for i, (title, prompt) in enumerate(analysis_steps):
        yield f" Step {i + 2}: Running {title} Analysis \n"
        yield f"Prompt for this step:\n\n{prompt}\n\n\n"
        await asyncio.sleep(0.1) # UI update

        try:
            # Re-populating config since inference_step might modify it
            current_gen_config = generation_config.copy()

            #  FIX 
            # Pop pipeline-specific parameters that are not used by model.generate()
            # This prevents the "model_kwargs are not used by the model" error.
            sampling_fps = current_gen_config.pop("sampling_fps", 2.0)
            current_gen_config.pop("num_perceptions", None) # Safely remove this key
            #  END FIX 
            
            if current_gen_config.get("temperature", 0.0) > 0.0:
                current_gen_config["do_sample"] = True
                
            ans = inference_logic.inference_step(
                video_path=video_path,
                prompt=prompt,
                generation_kwargs=current_gen_config, # This dict is now clean
                sampling_fps=sampling_fps,
                pred_glue=None  # Not used in factuality checks
            )
            yield f"Model Analysis:\n\n{ans}\n\n\n"
        except Exception as e:
            error_message = f"An error occurred during the '{title}' analysis step: {e}"
            logger.error(error_message, exc_info=True)
            yield f"ERROR: {error_message}\n\n"
            break # Stop pipeline on error
    
    yield "Factuality Analysis Complete \n"
