---
name: deep-research-verifier
description: Deeply researches top academic papers, technical articles, YouTube teardowns, and GitHub repositories, and cross-verifies theoretical formulations, algorithms, and loss/reward math directly against codebase implementations. Use before designing architecture, implementing algorithms, fine-tuning agents, or validating code fidelity against research literature.
---

# Deep Research Verifier

## Goal
Systematically ground every architectural design, loss function, reward formulation, and evaluation benchmark in peer-reviewed research literature, canonical GitHub implementations, and technical articles, while rigorously verifying that codebase implementations strictly match theoretical guarantees and empirical findings without discrepancies.

## Instructions
1. **Multi-Source Literature & Reference Sweep:**
   - Query and analyze peer-reviewed papers from top AI conferences (NeurIPS, ICML, ICLR, ACL, EMNLP, arXiv).
   - Inspect authoritative GitHub repositories (e.g., [Lexsi-Labs/AgentTune](https://github.com/Lexsi-Labs/AgentTune), [Lexsi-Labs/AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), [huggingface/trl](https://github.com/huggingface/trl), Berkeley Gorilla/BFCL, tau-bench).
   - Review technical articles, engineering postmortems, and conference/presentation teardowns (YouTube technical lectures, lab blogs).
2. **Mathematical & Algorithmic Extraction:**
   - Extract exact mathematical formulations: loss functions (SFT cross-entropy with masking, DPO policy/reference margins, GRPO group relative advantage estimation, KL divergence penalties), reward decomposition, and trajectory discount structures.
   - Document edge cases, failure conditions, and known numerical instabilities identified in prior art.
3. **Theory-to-Code Cross-Verification:**
   - Cross-examine codebase implementation line-by-line against published equations and reference code:
     - Are loss terms weighted and signed correctly?
     - Are sequence masks applied strictly to assistant tool-call tokens (ignoring user and observation tokens during loss computation)?
     - Are advantage estimates normalized over group rollouts without leakage?
     - Is the efficiency penalty applied *strictly conditional* on verified success?
4. **Discrepancy & Gap Diagnosis:**
   - Identify any deviation between reference paper assumptions and the actual code execution path.
   - Detect subtle bugs: prompt template mismatches, special token omission, unscaled rewards, silent NaN/Inf propagation, or reward-hacking loopholes.
5. **Evidence Dossier & Architecture Sync:**
   - Log findings into `references/research_notes.md` with full bibliographic citations (title, authors, conference/year, stable URL, arXiv ID).
   - Ensure verified workflows, interface contracts, and state machines are documented in `system_design.md`.

## Examples
- **Input:** "Verify our GRPO efficiency reward implementation against DeepSeekMath / Skalse et al."
- **Output:** Comparative table matching paper equations $\mathcal{J}_{\text{GRPO}}(\theta)$ against `src/training/rewards.py`, verifying masking, advantage normalization, degenerate policy unit tests, and empirical references.

## Constraints
- Never cite papers or repositories without verifying their exact titles, authors, and URLs.
- Never approve code where the theoretical guarantee contradicts the runtime logic.
- Always require unit tests reproducing both nominal behavior and degenerate edge cases.

