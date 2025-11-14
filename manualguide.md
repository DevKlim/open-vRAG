## Manual Labeling Guideline for Video Disinformation Analysis

### 1. Introduction

**Purpose:** This document provides a comprehensive guide for manually labeling social media videos. The goal is to create a high-quality, consistent "ground truth" dataset for training and evaluating misinformation detection models.

**Core Task:** As a labeler, you will watch a video and analyze its accompanying text caption and audio transcript. You will then fill out a structured JSON form with objective labels based on the criteria outlined below.

**Guiding Principles:**
*   **Objectivity First:** Set aside all personal beliefs and political opinions. Your analysis must be based solely on the evidence present in the provided materials (video, audio, caption).
*   **Holistic Analysis:** Do not judge the content based on just one element. A misleading caption might accompany a factual video, or vice-versa. Consider all three components together before making a final judgment.
*   **When in Doubt, Be Conservative:** If the evidence for a severe classification (e.g., "Deliberate Disinformation") is not clear and convincing, default to a less severe but still accurate label (e.g., "Manipulated Content" or "Unintentional Misinformation").

---

### 2. General Content Labels

These labels describe the general characteristics of the content.

#### `video_context_summary` (String)
*   **Objective:** To provide a brief, neutral summary of the video's content.
*   **Instructions:** In 1-2 sentences, describe the main events, subjects, and setting. Focus on what is literally happening. Avoid interpretation, emotion, or judgment.
*   **Good Example:** "A person is assembling a piece of flat-pack furniture in a living room while following paper instructions."
*   **Bad Example:** "A frustrated man is struggling to build a cheap table, showing how terrible this company's products are."

#### `political_bias` (Integer, 1-10)
*   **Objective:** To measure the extent to which the content promotes a specific political viewpoint or party.

| Score  | Description                      | Example Scenario                                                                                         |
| :----- | :------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **1-2**  | **Blatant Propaganda**           | Uses derogatory language, demonizes opponents, presents information with no nuance. Clearly an attack piece. |
| **3-4**  | **Strongly Biased**              | Presents only one side of an argument, uses loaded language and emotional appeals to favor one viewpoint.      |
| **5-6**  | **Moderately Biased**            | Primarily focuses on one perspective but may briefly acknowledge another. The framing clearly favors one side. |
| **7-8**  | **Mostly Neutral / Slight Lean** | Factual reporting that contains slightly leading questions or framing that suggests a preferred viewpoint.    |
| **9-10** | **Objective & Balanced**         | Presents multiple viewpoints fairly, uses neutral language, and avoids emotional manipulation.             |

#### `criticism_level` (Integer, 1-10)
*   **Objective:** To measure how critical or aggressive the content is towards a person, group, or idea.

| Score  | Description                   | Example Scenario                                                                      |
| :----- | :---------------------------- | :------------------------------------------------------------------------------------ |
| **1-2**  | **Aggressive / Hostile**      | Uses insults, mockery, or overtly hostile language. Aims to attack rather than argue.   |
| **3-4**  | **Strongly Critical**         | Directly and repeatedly points out flaws or failures without resorting to insults.       |
| **5-6**  | **Moderately Critical**       | Expresses disagreement or points out negative aspects in a measured tone.               |
| **7-8**  | **Slightly Critical / Nuanced** | Raises questions or presents a balanced view that includes some negative points.        |
| **9-10** | **Neutral or Positive**       | Content is purely informational, educational, positive, or contains no criticism at all. |

#### `video_audio_pairing` / `video_caption_pairing` / `audio_caption_pairing` (Integer, 1-10)
*   **Objective:** To score the alignment and consistency between the different modalities (visuals, sound, text).

| Score  | Description                     | Example for `video_audio_pairing`                                                         |
| :----- | :------------------------------ | :---------------------------------------------------------------------------------------- |
| **1-2**  | **Complete Mismatch**           | Audio is about cooking, while the video shows a car race. They are completely unrelated.    |
| **3-4**  | **Loosely Related / Mismatching** | Audio discusses a political protest in general, but the video shows a specific, unrelated event. |
| **5-6**  | **Thematically Aligned**        | The audio and video are about the same general topic (e.g., climate change) but don't directly correspond. |
| **7-8**  | **Closely Aligned**             | The audio describes events that are very similar to what is happening in the video.        |
| **9-10** | **Perfectly Synchronized**      | The audio is a direct narration of or commentary on the specific actions shown in the video. |

---

### 3. Disinformation & Manipulation Analysis (`disinformation_analysis`)

This section requires you to act as a threat analyst. You must classify the nature, intent, and tactics of any manipulative content.

#### `disinformation_level` (String - Choose ONE)
*   **Objective:** To classify the severity and intentionality of false information.
    *   **"No Misinformation":** Content appears factual, consistent, and well-intentioned. All claims are verifiable or presented as opinion.
    *   **"Unintentional Misinformation":** Content contains factual errors, but the intent does not appear malicious. *Example: Sharing an old news story believing it is current; making a genuine mistake in statistics.*
    *   **"Manipulated Content":** Genuine content that is edited, cropped, or re-contextualized to mislead. This includes half-truths and misleading juxtapositions. *Example: Using a real photo of a protest from 2015 with a caption claiming it happened yesterday.*
    *   **"Deliberate Disinformation":** Content that is partially or completely fabricated with clear malicious intent to deceive. *Example: A deepfake video, a completely invented news story, or AI-generated images presented as real.*

#### `disinformation_intent` (String - Choose ONE)
*   **Objective:** To identify the primary motivation behind the manipulation.
    *   **"None":** No manipulative goal is apparent (use for "No Misinformation" or "Unintentional Misinformation").
    *   **"Commercial":** To make money, either by selling a fraudulent product or service, or by generating outrage clicks for ad revenue.
    *   **"Political":** To influence a political outcome, damage a political opponent, or promote a specific ideology.
    *   **"Social":** To stoke social division, harass an individual or group, or promote social tribalism (e.g., "us vs. them").
    *   **"State-Sponsored":** The content aligns with the known geopolitical objectives and propaganda tactics of a specific nation-state targeting another country or population.

#### `threat_vector` (String - Choose ONE)
*   **Objective:** To identify the specific technique used to deceive the audience.
    *   **"None":** No deceptive technique is used.
    *   **"False Context":** Real, genuine content is presented with a misleading caption, headline, or narration to change its meaning.
    *   **"Imposter Content":** The content falsely claims to be from a reputable source (e.g., using a fake "BBC News" logo).
    *   **"Manipulated Visuals/Audio":** The pixels or sounds have been altered. This includes photoshopping, misleading video edits (splicing unrelated clips together), or using AI to change what someone is saying.
    *   **"Fabricated Narrative":** The entire story is made up. This applies to content where the core claim is a complete lie, even if the visuals are real but unrelated.

#### `sentiment_and_bias_tactics` (Nested Object)

##### `emotional_charge` (Integer, 1-10)
*   **Objective:** To score the emotional intensity and volatility of the content.

| Score  | Description                  | Example Scenario                                                                      |
| :----- | :--------------------------- | :------------------------------------------------------------------------------------ |
| **1-2**  | **Calm & Factual**           | A calm, academic lecture or a technical "how-to" guide.                                 |
| **3-4**  | **Low Emotional Content**    | A standard news report with a neutral tone.                                           |
| **5-6**  | **Moderate Emotional Appeal**| An opinion piece or documentary that uses music and storytelling to evoke some emotion. |
| **7-8**  | **Highly Emotional**         | Uses emotionally charged language, sad/angry music, and focuses on personal suffering or outrage. |
| **9-10** | **Inflammatory / Enraging**  | Designed to provoke immediate, strong reactions of anger, fear, or hatred. Uses aggressive language and visuals. |

##### `targets_cognitive_bias` (Boolean)
*   **Objective:** To identify if the content seems designed to exploit common psychological biases.
*   **Label as `true` if:** The content strongly appeals to **confirmation bias** (only showing evidence that supports one view), **fear mongering** (exaggerating threats), or **bandwagon effects** (claiming "everyone knows" or "people are saying").
*   **Label as `false` if:** The content presents information in a straightforward manner without obvious psychological manipulation.

##### `promotes_tribalism` (Boolean)
*   **Objective:** To identify if the content creates or reinforces a strong "us vs. them" dynamic.
*   **Label as `true` if:** The content clearly defines an "in-group" (e.g., "patriots," "our people") and an "out-group" (e.g., "the elites," "foreigners," "the other party"), blaming the out-group for problems and fostering in-group loyalty.
*   **Label as `false` if:** The content discusses issues without framing them as a conflict between distinct, opposing groups of people.

---

### 4. Complete Worked Example

**Scenario:**
*   **Video:** Shows clips of a peaceful daytime protest, but they are rapidly intercut with unrelated clips of nighttime riots and fires. Scary, high-tempo music plays.
*   **Audio:** A narrator with a grave voice claims, "The city is under siege by violent mobs funded by our political opponents. They want to destroy our way of life."
*   **Caption:** "WAKE UP!!! This is what [Politician X] is letting happen to our country! #StopTheViolence #CivilWar"

**Correct JSON Label:**
```json
{
  "video_context_summary": "The video shows edited clips of a protest, intercut with separate clips of riots and fires, accompanied by a narrator.",
  "political_bias": 1,
  "criticism_level": 2,
  "video_audio_pairing": 9,
  "video_caption_pairing": 10,
  "audio_caption_pairing": 10,
  "disinformation_analysis": {
    "disinformation_level": "Deliberate Disinformation",
    "disinformation_intent": "Political",
    "threat_vector": "Manipulated Visuals/Audio",
    "sentiment_and_bias_tactics": {
      "emotional_charge": 10,
      "targets_cognitive_bias": true,
      "promotes_tribalism": true
    }
  }
}
```
**Rationale:**
*   `political_bias` is **1** because it's blatant propaganda against a political figure.
*   `criticism_level` is **2** as it's aggressive and hostile ("destroy our way of life").
*   The pairings are high (**9-10**) because the audio, video edits, and caption are all telling the *same false story*, even if the components are mismatched with reality.
*   `disinformation_level` is **"Deliberate Disinformation"** because splicing unrelated clips is a malicious fabrication.
*   `disinformation_intent` is **"Political"** as it targets a specific politician.
*   `threat_vector` is **"Manipulated Visuals/Audio"** due to the misleading editing.
*   `emotional_charge` is **10** because it uses fear-mongering tactics ("under siege," "civil war").
*   `targets_cognitive_bias` is **true** as it confirms the bias of viewers who already fear the opposition.
*   `promotes_tribalism` is **true** by creating an "us" ("our way of life") vs. "them" ("violent mobs," "political opponents").