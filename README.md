This repository provides the core implementation of our hierarchical two-stage cross-lingual transfer framework for extremely low-resource text-to-speech (TTS).

The framework progressively transfers knowledge from a pretrained multilingual TTS model to linguistically related intermediate languages and finally to an extremely low-resource target language.

The implementation is built on top of CosyVoice3. Only the language modeling module is adapted during both stages, while the speech generation modules remain frozen.

## Method Overview

The proposed framework consists of two adaptation stages:

```text
Pretrained Multilingual TTS Model
              |
              v
Stage 1: Intermediate-Language Adaptation
              |
              v
     Stage-1 Checkpoints
              |
              v
Top-3 Checkpoint Selection by Dev Loss
              |
              v
       Checkpoint Averaging
              |
              v
          llm_avg3.pt
              |
              v
Stage 2: Target-Language Adaptation
              |
              v
        Final TTS Model
