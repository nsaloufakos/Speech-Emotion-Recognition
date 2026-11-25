# P20 project requirements audit

Deadline: **Friday, November 27, 2026 at 11:59 p.m. Toronto time**, confirmed by the user from onQ.

Status checked against supplied course documents, proposal commitments and local result files. The project is not submission-complete. This checklist is planning evidence, not final report prose.

## Course objectives and deliverables

| Scope | Requirement | Status | Evidence and next action |
|---|---|---|---|
| course objective | Fine-tune HuBERT-base with a classification head; report UAR and WAR | **missing results** | src/models/finetuned_hubert_linear.py exists; frozen-head evaluations do not meet backbone fine-tuning objective (capstone-guide.pdf, PDF page 26) |
| course objective | Compare MFCC-BiLSTM, wav2vec2 features and HuBERT features | **partial; Mac MFCC stopped for CUDA rerun** | HuBERT and wav2vec2 matched original/full/capped runs complete; MFCC completion must be checked separately. Historical original/legacy-normalization baselines also exist (capstone-guide.pdf, PDF page 26) |
| course objective | Layer-wise HuBERT probing with a linear classifier per Transformer layer | **complete on original audio** | 12/12 original-audio layer evaluations; 24 folds and three seeds required (capstone-guide.pdf, PDF page 26) |
| course objective | Speaker-independent LOSO evaluation | **complete for baseline experiments** | output/splits.csv: 24 actors; 21 train / 2 validation / 1 test actor; batch speaker calibration assumption must be declared (capstone-guide.pdf, PDF page 26) |
| course objective | Propose and evaluate speaker normalization OR augmentation | **complete through normalization** | analysis/pitch_validation/PRAAT_MEDIAN_EXPERIMENT.md and F0_AUDIT.md (capstone-guide.pdf, PDF page 26) |
| additional proposal commitment | Fine-tune wav2vec2 as well as HuBERT | **missing results** | src/models/finetuned_wav2vec_linear.py exists; no completed result found (proposal.tex, Proposed Extension) |
| additional proposal commitment | Repeat layer probing under normalization | **missing results** | 0 normalized-layer evaluation files found (proposal.tex, Evaluation Plan) |
| additional proposal commitment | Pitch, time-stretch and noise augmentation ablations, applied only to training recordings | **missing evaluated ablations** | Generation code exists; no completed augmentation-combination comparison found. Normalization already satisfies the course OR objective; document a scope change if this proposal item is omitted (proposal.tex, Early Novel Contribution and timeline; capstone-guide.pdf timeline page 27) |
| final deliverable | 6-8 main pages plus appendix; final PDF and LaTeX source | **not found** | No completed final-report .tex/.pdf found in the repository; provided final-template.tex is still a template (capstone-guide.pdf, PDF page 35; final-template.tex) |
| report content | Thematic related work; explicit distinction from prior methods; published baseline context | **partial** | Proposal lists six papers. Guide asks >=5; final template asks >=7 and >=2 published baselines. Published-score protocol matching remains outstanding; 7-class fixed-split results are not directly comparable to 8-class LOSO (capstone-guide.pdf, PDF pages 4/35; final-template.tex; midterm(1).tex) |
| report content | Ablation evidence for proposed design choices | **partial** | Full vs cap isolates limiting shift size; estimator/mask/mean-vs-median changes remain bundled. Add fixed-mask/statistic ablations and a zero-shift resynthesis control for stronger causal claims (final-template.tex, Ablation Study) |
| report content | Qualitative predictions, confusion matrices, >=3 failure categories with examples and mitigations | **partial** | Pitch diagnostics, listening exports and prediction/confusion data exist; integrate model errors and at least three evidence-based failure categories into the final report (final-template.tex, Qualitative Analysis and Failure Analysis) |
| report content | Architecture diagram or pseudocode, equations, loss, training setup, scope changes and limitations | **needs final integration** | Methods and configs exist. Rename current method as actor-level median normalization; disclose batch adaptation; correct the variance statement (Hz variance scales, semitone variance is ideally preserved) (final-template.tex, Method/Experiments/Discussion; proposal.tex) |
| final deliverable | Actual team contributions and percentages; accurate AI disclosure | **team input required** | Final contributions must reflect actual work. AI has assisted code generation, execution, diagnostics and interpretation; do not describe that as only grammar/ideas assistance (capstone-guide.pdf, PDF pages 2/35; final-template.tex; CISC473-F26.pdf, PDF page 11) |
| course reproducibility requirement | One-command reproduction, useful README, versioned dependencies, dataset instructions and checkpoint/retraining instructions | **incomplete** | README.md contains only the title; requirements.txt is unpinned; current Praat and caches use temporary/local setup. Many commands and hashes exist but are not packaged as one documented pipeline (capstone-guide.pdf, PDF page 3; final-template.tex, Reproducibility Checklist) |
| submission requirement | Public GitHub repository linked in deliverables; source/code zip | **public access and final packaging unverified** | Remote URL is present. No successful external public-access verification or final submission archive was established (capstone-guide.pdf, PDF page 2; final-template.tex) |
| final deliverable | 20-minute video (18-22 accepted), MP4 and slides PDF, all members on camera and >=3 minutes each | **not found locally** | No final slides/video found; references and AI disclosure required; content after 22 minutes not assessed; GroupX_Presentation.mp4 and GroupX_Slides.pdf (presentation_handout_compact.pdf, PDF page 1) |
| optional publication-quality bonus | Novelty distinction, >=2 published SOTA comparisons with statistical evidence, design-choice ablations, submission evidence | **not established** | Current normalization study is exploratory; three-seed SD alone does not establish significance; no venue/arXiv submission performed (capstone-guide.pdf, PDF page 5) |

## Source conflicts and corrections

- The guide applies to CISC 473; supplied LaTeX template headers say CISC 867 and must be adapted if used.
- Presentation handout says 20% of project / 4% of course; guide overview says 10% of project. Use the handout's explicit format and confirm grading weight on onQ/instructor.
- User confirmed the onQ deadline: November 27, 2026 at 11:59 p.m. America/Toronto. Guide only gives relative Week-10 deadlines.
- Guide permits literature search/debugging/writing feedback but requires student analysis and experiment implementation/execution. Syllabus page 11 contains both permission and prohibition wording under a select-one heading. Clarify permitted AI-assisted code/automation; disclose actual assistance and personally review/reproduce/understand work.
- Attached parent-directory midterm.tex is largely a template; repository midterm(1).tex contains the completed narrative. Local presence does not prove submission.
- Both mock-exam PDFs are practice exams and add no project deliverables.
- SpeechFormer bibliography needs correction: original 2022 paper is arXiv:2203.03812; 2302.14638 is SpeechFormer++ (2023), and guide link 2203.16741 is an unrelated physics paper. Verify primary sources rather than copying template references.

## Immediate priorities

1. Finish and verify the matched three-model original/full/capped matrix.
2. Complete HuBERT backbone fine-tuning or obtain an explicit approved scope change; a frozen classifier head is not fine-tuning.
3. Complete the component/control ablations; decide whether proposal-only fine-tuning, augmentation and normalized probing commitments will be completed or explained as scope changes.
4. Build final tables, figures and three concrete failure categories; write the final report in the team's own words with accurate method/protocol changes and references.
5. Package reproducibility, clarify the policy questions, and record the 18-22 minute video with all three members.

## Deadline plan

- November 24: finish the matched three-model normalization comparison and verify the resulting tables.
- November 25: prioritize the required HuBERT fine-tuning result and the most informative component/control ablations; settle proposal scope changes with the team/instructor.
- November 26: consolidate figures and failure examples, write the final report in the team's own words, prepare slides, and record a rehearsal.
- November 27: test reproduction and packaging, proofread references/disclosures, record and verify the final video, and upload with time to spare before 11:59 p.m.
- After submission: broader cross-corpus and publication experiments, unless all required work is already secure.
