# FYLs-G2P: A 1.8M Parameter G2P Engine with Context Awareness and OOV Phonics (That Can Be Deployed on Almost Any Device)

## ⚡ Introduction

Most G2P (Grapheme-to-Phoneme) solutions are either massive end-to-end models that hallucinate, or simple dictionary lookups that fail at context. 

**FYLs-G2P** is a hybrid high-performance engine (~1.8M params) that bridges this gap. It doesn't just "remember" words; it **understands** them through:
1.  **Contextual POS Tagger (ONNX)**: Resolves heteronyms (e.g., *present* vs *present*) based on syntax.
2.  **Neural OOV Inference (BiGRU)**: A Seq2Seq model that predicts phonemes for unseen words using learned English phonotactics.
3.  **Weighted Graph Mapping (`XPOSAlternative`)**: A unique algorithm that dynamically bridges the gap between predicted POS tags and available dictionary entries.

**Total size:** ~1.8M Params. | **Target:** Edge devices & Real-time TTS.

---

## 🚀 Key Features

### 1. Robust OOV & Morphological Intelligence
The neural fallback isn't just a guesser. It captures **morphology** (plurals, tenses) and **compound word phonetics**. 
*   *Example:* Even if the dictionary only has "lead" (/lid/), the model can infer that in `leadcolored`, it should be pronounced as /lɛd/ (the metal) based on the learned representation of compounds.

### 2. Context-Aware Homograph Disambiguation
Correctly distinguishes between nouns, verbs, and adjectives for the same spelling (e.g., *record*, *object*, *desert*) using real-time syntactic analysis.

### 3. "Tag Distance" Fuzzy Matching
When the POS Tagger and Lexicon tags don't align perfectly, our **Dijkstra-based mapping** finds the linguistically closest phonetic candidate instead of falling back to a random default.

---

## 🧪 Performance Demo: The "Homograph & OOV" Torture Test

This sentence tests both syntactic disambiguation AND neural prediction of non-standard compound words.

**Input Text:**
> "Since there was no **present** like the **present**, he decided to **present** the project to the **lead** singer, who was wearing a **leadcolored** suit in the **desert**, even though his friends might **desert** him."

**Output IPA:**
> sˈɪns ðɛɹ wʌz nˈO **pɹˈɛzᵊnt** lˈIk ði **pɹˈɛzᵊnt** , hi dəsˈIdᵻd tu **pɹizˈɛnt** ði pɹˈɑʤˌɛkt tu ði **lˈid** sˈɪŋəɹ , hˌu wʌz wˈɛɹɪŋ ɐ **lˈɛdkˌʌləɹd** sˈut ɪn ði **dˈɛzəɹt** , ˈivən ðˌO hɪz fɹˈɛndz mˌIt **dəzˈɜɹt** hˌɪm .

### 🔍 OOV Analysis (The fallback engine at work)

| Word | Predicted IPA | Why it's impressive |
| :--- | :--- | :--- |
| **leadcolored** | `lˈɛdkˌʌləɹd` | **Correctly identified the /lɛd/ (metal) pronunciation** in a compound context, despite being a non-standard OOV word. |
| **friends** | `fɹˈɛndz` | Automatically handled the **voiced plural suffix** (/z/ after /d/) without needing an explicit dictionary entry. |

---
