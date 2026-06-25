# Reasoning-Response Disconnect in LLMs

## Research Question
Do large language models internally represent correct reasoning 
that fails to surface in their generated output — and can we 
causally intervene to fix this?

## Core Hypothesis
Correct reasoning features are present and decodable in early-to-mid 
residual stream layers but are suppressed or overridden in final MLP 
and unembedding layers, producing responses that contradict the model's 
own internal state.

## Approach
1. **Locate the disconnect** — Layer-wise logit lens and linear probing 
   across Qwen, Mistral, and Llama model families on SimpleQA, 
   TruthfulQA, and StrategyQA.
2. **Attribute causally** — Activation patching to isolate whether 
   suppression originates in MLP blocks or attention heads
3. **Extract truth features** — SAELens to identify sparse features 
   encoding factual correctness vs hallucination
4. **Intervene** — Activation steering to amplify truth features at 
   corrupting layers without retraining

## Status
🔬 Active research.

## Related Work
- Orgad et al., "LLMs Know More Than They Show" (ICLR 2025) - https://arxiv.org/pdf/2410.02707 
- Burns et al., "Discovering Latent Knowledge in Large Language Models Without Supervision" (2024) - https://arxiv.org/pdf/2212.03827
