# Manual Labeling Guide: Factuality & Verification Rubric

## 1. Core Scoring Philosophy

**The Golden Rule:** 
*   **Score 1 =** Fabricated, Malicious, Contradictory, or Harmful.
*   **Score 10 =** Authentic, Verifiable, Consistent, and Neutral.
*   **Score 5 =** The "Unknown" or "Generic" Baseline. If there is no evidence of authenticity *but also* no evidence of manipulation, score a 5.

**Determinism Instructions:** 
When analyzing content, start at **5**. 
- Deduct points for specific evidence of manipulation or low quality. 
- Add points for specific evidence of verification or high quality. 
- **Do not guess.** If a feature is ambiguous, stay in the 4-6 range.

---

## 2. Veracity Vectors (1-10 Scale)

### A. Visual Integrity
**Question:** *Is the visual footage real, unedited, and accurately representing the time/place it claims to?*

| Score | Label | Detailed Criteria & Examples |
| :--- | :--- | :--- |
| **1-2** | **Fabricated / Deepfake** | **Criteria:** Visuals are fully AI-generated or manipulated pixel-by-pixel to show events that never occurred.<br>**Example (1):** High-quality deepfake of a politician confessing to a crime they didn't commit.<br>**Example (2):** Obvious AI generation (e.g., extra fingers, morphing textures) presented as real footage. |
| **3-4** | **Deceptive Manipulation** | **Criteria:** Real footage that has been spatially or temporally altered to change the narrative.<br>**Example (3):** Speeding up a video to make a person look like they are panicking.<br>**Example (4):** Cropping out the person who actually started a fight to make the other party look like the aggressor. |
| **5-6** | **Recontextualized / Stock** | **Criteria:** The pixels are real and unedited, *but* they do not depict the specific event claimed, or are generic.<br>**Difference (5 vs 6):**<br>**5:** "False Context" - Using old footage for a new event (e.g., 2015 protest footage labeled "Live 2025").<br>**6:** "Generic" - Using stock B-roll (e.g., generic "hacker typing" clip) to illustrate a story without claiming it's specific evidence. |
| **7-8** | **Processed / Standard** | **Criteria:** Real footage with standard post-production that does not mislead.<br>**Example (7):** Heavy filters, text overlays, music videos, or meme formats where the underlying video is real.<br>**Example (8):** Standard news editing (jump cuts, B-roll transitions) that condenses time accurately. |
| **9-10** | **Forensic / Raw** | **Criteria:** High-assurance authenticity.<br>**Difference (9 vs 10):**<br>**9:** Continuous, clear shot without cuts, but lacking metadata verification.<br>**10:** Verified raw footage with metadata, timestamp, or cryptographic signature confirming location and time. |

---

### B. Audio Integrity
**Question:** *Is the audio track authentic, synchronous, and free from voice cloning or deceptive editing?*

| Score | Label | Detailed Criteria & Examples |
| :--- | :--- | :--- |
| **1-2** | **Synthetic / Cloned** | **Criteria:** Audio is AI-generated (Voice Cloning) or heavily spliced (Sentence Mixing) to fabricate speech.<br>**Example (1):** AI Voice Clone of a celebrity endorsing a scam.<br>**Example (2):** "Sentence mixing" editing where words are cut from different speeches to form a new, controversial sentence. |
| **3-4** | **Mismatched / Dubbed** | **Criteria:** Real video with a completely fake or unrelated audio track added to mislead.<br>**Example (3):** Adding aggressive crowd booing sounds to a video of a silent room.<br>**Example (4):** A "Bad Lip Reading" style dub presented as a real translation of a foreign leader. |
| **5-6** | **Generic / TTS** | **Criteria:** Audio is artificial but not necessarily malicious.<br>**Difference (5 vs 6):**<br>**5:** Overwhelming background music that obscures the actual event audio.<br>**6:** Standard AI Text-to-Speech (e.g., TikTok Voice) reading the caption. It's "fake" speech, but serves a functional purpose. |
| **7-8** | **Processed / Narrated** | **Criteria:** Audio is processed for clarity or includes a human narrator.<br>**Example (7):** Noise reduction or EQ that makes voices sound slightly unnatural but clearer.<br>**Example (8):** Professional voiceover (news anchor) describing the video content accurately. |
| **9-10** | **Natural / Synchronous** | **Criteria:** Authentic sound from the event.<br>**Difference (9 vs 10):**<br>**9:** Clear ambient sound (street noise, wind) consistent with the video.<br>**10:** Perfect lip-sync of a speaker on camera, with consistent acoustic reverb matching the visual environment. |

---

### C. Source Credibility
**Question:** *Who posted this? Do they have a history of reliability?*

| Score | Label | Detailed Criteria & Examples |
| :--- | :--- | :--- |
| **1-2** | **Malicious / Blacklisted** | **Criteria:** Source is a known disinformation actor, imposter, or satire site presented as fact.<br>**Example (1):** An account mimicking "BBC News" (Imposter).<br>**Example (2):** A known state-sponsored propaganda bot account. |
| **3-4** | **Hyper-Partisan / Fringe** | **Criteria:** Source has a clear agenda that overrides factual reporting.<br>**Example (3):** Conspiracy theory blogs or channels that regularly fail fact-checks.<br>**Example (4):** Extreme partisan activists who post inflammatory, unverified content. |
| **5-6** | **Unknown / Individual** | **Criteria:** No verification available.<br>**Difference (5 vs 6):**<br>**5:** Anonymous account created recently with low follower count (Unknown).<br>**6:** Established individual user or "Citizen Journalist" with a history of normal posting, but no institutional backing. |
| **7-8** | **Mainstream / Verified** | **Criteria:** Established institutions with editorial standards.<br>**Example (7):** Verified opinion commentators or editorial columns (Subjective but identifiable).<br>**Example (8):** Standard news networks (CNN, Fox, BBC) reporting hard news. |
| **9-10** | **Primary / Institutional** | **Criteria:** The origin of the information.<br>**Difference (9 vs 10):**<br>**9:** Academic institutions, Government official channels, or Experts in the specific field.<br>**10:** Direct Evidence Provider (e.g., NASA video feed, CSPAN unedited feed, or the person who actually filmed the video). |

---

### D. Logical Consistency
**Question:** *Does the content make sense? Is the argument sound?*

| Score | Label | Detailed Criteria & Examples |
| :--- | :--- | :--- |
| **1-2** | **Incoherent / Fallacious** | **Criteria:** Argument relies entirely on logical fallacies or internal contradictions.<br>**Example (1):** "Word Salad" – incoherent rambling.<br>**Example (2):** Direct contradiction (e.g., "The fire was caused by cold weather"). |
| **3-4** | **Speculative / Weak** | **Criteria:** Jumps to conclusions without evidence.<br>**Example (3):** Conspiracy logic: "They don't want you to know X, therefore Y is true."<br>**Example (4):** Correlation/Causation fallacy: "Event A happened before Event B, so A caused B" (without proof). |
| **5-6** | **Anecdotal / Opinion** | **Criteria:** Logic applies to personal experience but isn't a universal fact.<br>**Difference (5 vs 6):**<br>**5:** Presenting a single anecdote as a statistical trend ("It happened to me, so it happens to everyone").<br>**6:** Stated clearly as subjective opinion ("I feel that...") which is logically valid for the speaker but not a fact. |
| **7-8** | **Structured / Plausible** | **Criteria:** A coherent argument is presented.<br>**Example (7):** A plausible theory that fits the known facts but lacks definitive proof.<br>**Example (8):** A structured legal or political argument with premises and a conclusion. |
| **9-10** | **Rigorous / Deductive** | **Criteria:** Scientific or Fact-based reasoning.<br>**Difference (9 vs 10):**<br>**9:** Cited sources and data to back up claims.<br>**10:** Deductively valid argument where the conclusion inevitably follows from proven premises. |

---

### E. Emotional Manipulation
**Question:** *Is the content trying to force a reaction, or is it neutral? (High Score = Neutral)*

| Score | Label | Detailed Criteria & Examples |
| :--- | :--- | :--- |
| **1-2** | **Incitement / Extremist** | **Criteria:** Dangerous rhetoric designed to trigger violence or hatred.<br>**Example (1):** Calls for violence, dehumanization ("vermin", "cancer"), or gore designed to traumatize.<br>**Example (2):** Hate speech or extreme vitriol. |
| **3-4** | **Inflammatory / Fear** | **Criteria:** "Rage Bait" or "Fear Mongering".<br>**Example (3):** "They are coming for your children!" (Panic induction).<br>**Example (4):** Aggressive ridicule or mockery designed to humiliate. |
| **5-6** | **Dramatic / Engaging** | **Criteria:** Emotional but safe.<br>**Difference (5 vs 6):**<br>**5:** "Clickbait" style hype ("You won't believe this!").<br>**6:** Storytelling/Vlog style. Uses music and humor to entertain. |
| **7-8** | **Professional / Calm** | **Criteria:** Generally measured tone.<br>**Example (7):** An impassioned but respectful speech.<br>**Example (8):** Standard news delivery (serious but not hysterical). |
| **9-10** | **Clinical / Objective** | **Criteria:** Zero emotional loading.<br>**Difference (9 vs 10):**<br>**9:** Instructional or Educational tone.<br>**10:** Pure Data / Raw Feed (e.g., a weather radar loop, a surveillance tape). |

---

## 3. Modality Alignment Rubric (1-10 Scale)

These scores measure how well the different components of the post match each other.

### A. Video-Audio Alignment
*   **1-2 (Mismatch):** Audio is completely unrelated to video (e.g., heavy metal music over a silent meditation video).
*   **3-4 (Dissonant):** Audio mood contradicts video mood (e.g., laughing track over a tragic accident video).
*   **5-6 (Thematic):** Audio matches the general vibe (e.g., generic "protest noise" over a specific protest video).
*   **7-8 (Descriptive):** Audio narrates or discusses the specific events shown on screen.
*   **9-10 (Synchronous):** Audio is the direct sound recording of the video event (lips move, words are heard).

### B. Video-Caption Alignment
*   **1-2 (Lie):** Caption claims the video shows X, but the video clearly shows Y. (e.g., Caption: "Massive fire!", Video: A small candle).
*   **3-4 (Misleading):** Caption exaggerates what is shown or implies context that isn't there.
*   **5-6 (Tangent):** Caption discusses a related topic but doesn't describe the video. (e.g., Video: A cat, Caption: "I had a bad day at work").
*   **7-8 (Accurate):** Caption accurately describes the visual content.
*   **9-10 (Grounded):** Caption provides specific, verifiable details visible in the video (e.g., "The red car turns left at 0:15").

### C. Audio-Caption Alignment
*   **1-2 (Contradiction):** Caption says "He said Yes", Audio says "No".
*   **5-6 (Summary):** Caption summarizes the gist of a long speech.
*   **9-10 (Transcript):** Caption is a word-for-word transcript of the audio.

---

## 4. Disinformation Classification Guide

**1. Classification:**
*   **None:** Factually accurate or harmless opinion.
*   **Misinformation:** False, but likely accidental (e.g., sharing an outdated chart by mistake).
*   **Disinformation:** False and clearly intentional (e.g., a deepfake, or a coordinated lie).
*   **Satire:** Humor/Parody. *Note: If satire is labeled "Real" by the user, it becomes Misinformation.*

**2. Intent:**
*   **Political:** To change votes, policy, or public opinion on governance.
*   **Commercial:** To sell a product, get clicks for ad revenue, or promote a scam.
*   **Social:** To create tribalism ("Us vs Them"), promote racism, or social division.
*   **None:** No apparent agenda (e.g., a prank).

**3. Threat Vector (How are they lying?):**
*   **False Context:** The video is real, but the caption is a lie (Time/Place/Event).
*   **Manipulated AV:** The video/audio files themselves are edited or AI-generated.
*   **Fabricated Narrative:** The entire story is made up (even if visuals are unrelated/stock).
*   **Imposter:** Pretending to be a trusted source (fake logos).