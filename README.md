<img src="assets/armor_logo.png" alt="ARMOR logo" width="230" align="left">

# ARMOR: Adaptive Regularized Mixture Optimization for Retrievers

**Low-resource domain adaptation for retrieval-augmented generation.**

<br clear="left">

---

## Overview

ARMOR is a retriever-centric adaptation method for low-resource domain RAG. In many specialized domains, only a small amount of supervision is available, while the document corpus is fixed and expensive to re-index. ARMOR targets this setting by keeping the generator and document index fixed and adapting the query encoder, concentrating limited supervision on the component that controls which evidence is shown to the model.

The method combines two complementary retriever objectives:

- **RAG likelihood**, which rewards retrieved documents that improve answer generation.
- **InfoNCE contrastive learning**, which improves the semantic retrieval geometry.

ARMOR balances these objectives through learnable temperatures and regularizes the adapted query encoder toward the frozen base query encoder, helping preserve compatibility with the existing document embedding space.

<p align="center">
  <img src="assets/intro_objective_comparison.png" alt="Retriever adaptation motivation" width="850">
</p>

<p align="center">
  <em>Retriever-side query-encoder adaptation provides a strong low-resource adaptation path compared with generator-side tuning and other baselines.</em>
</p>

## Method

In the ARMOR setup, documents are embedded once using a base dense retriever and stored in a fixed index. During adaptation, only the query encoder is updated.

At a high level, ARMOR optimizes:

```text
ARMOR loss = RAG likelihood + InfoNCE + query distillation
```

where the RAG and InfoNCE terms use separate learned temperatures. These temperatures control how sharply each objective shapes the query encoder during training, while query distillation discourages the adapted query encoder from drifting too far away from the base retrieval space.

## Results

ARMOR is evaluated on two complementary benchmarks. **Tele-Eval** measures open-ended QA quality and retrieval fidelity on domain-aligned evaluation splits, while **Tele-QnA** measures multiple-choice accuracy in a broader out-of-corpus transfer setting. The baselines compare closed-book generation, frozen RAG, single-objective query-encoder fine-tuning, static mixed-objective fine-tuning, and ARMOR.

<p align="center">
  <img src="assets/tele_eval_table1.png" alt="Tele-Eval Table 1 results" width="900">
</p>

<p align="center">
  <em>Tele-Eval compares open-ended QA score and retrieval recall across ISAC, JCC, and SAGIN. ARMOR gives the best average answer score on ISAC and JCC, while RAG QE FT is strongest on SAGIN.</em>
</p>

Tele-Eval evaluates whether the adapted retriever improves generation when the benchmark is aligned with the domain corpus used for training. The results show that query-encoder adaptation is consistently competitive with frozen Base RAG, and ARMOR provides the strongest overall tradeoff across answer quality and retrieval recall in the ISAC and JCC splits.

<p align="center">
  <img src="assets/tele_qna_table2.png" alt="Tele-QnA Table 2 results" width="520">
</p>

<p align="center">
  <em>Tele-QnA compares multiple-choice accuracy across the same domain categories, using a harder transfer benchmark that is not drawn from the adaptation corpus.</em>
</p>

Tele-QnA measures whether the adapted retriever transfers beyond the training-aligned corpus. The results are more mixed: frozen Base RAG remains strongest on ISAC, ARMOR ties the best JCC accuracy, and ARMOR obtains the best SAGIN accuracy. This split is useful because it separates in-corpus specialization from broader domain transfer.

### Generator Scale

The generator-scale comparison evaluates Base Gen, Base RAG, and ARMOR across different generator backbones. This experiment asks how much retriever optimization helps when the generator itself becomes stronger.

<p align="center">
  <img src="assets/model_performance_comparison.png" alt="ARMOR results across generator backbones" width="760">
</p>

<p align="center">
  <em>ARMOR improves over Base RAG across generator backbones, with the largest gains for smaller generators that rely more heavily on retrieved evidence.</em>
</p>

### Training Dynamics

The training-dynamics figure tracks the learned retrieval temperature, learned contrastive temperature, and query-distillation regularization during ARMOR optimization. It illustrates how the adaptive objective changes over training rather than using a fixed objective balance throughout.

<p align="center">
  <img src="assets/training_dynamics.png" alt="ARMOR adaptive temperature and regularization dynamics" width="850">
</p>

The retrieval temperature sharpens during training, indicating that the retriever increasingly focuses on high-utility evidence. Query distillation helps constrain this adaptation so the tuned query encoder remains compatible with the frozen document embedding space.

## Repository Structure

TODO: Add a concise overview of the code layout.

Suggested sections:

- `unified_data_gen/`: data filtering, indexing, QA generation, and alignment.
- `retriever_training/`: RAG, InfoNCE, mixed-objective, ARMOR, RAFT, and SFT training scripts.
- `evaluation/`: Tele-Eval and TeleQnA evaluation scripts.

## Setup

TODO: Add environment creation and dependency installation instructions.

Suggested items:

- Python version
- PyTorch / CUDA requirements
- FAISS installation
- vLLM installation
- Hugging Face model access
- OpenAI API key usage for judging/filtering, if applicable

## Running Experiments

TODO: Add canonical command examples.

For now, use `retriever_training/train_isac_all_methods.sh` as the canonical example script for launching the main methods:

```bash
cd retriever_training

# Examples:
bash train_isac_all_methods.sh rag
bash train_isac_all_methods.sh contriever
bash train_isac_all_methods.sh mix_static
bash train_isac_all_methods.sh mix_adaptive
```

TODO: Document expected data paths, checkpoints, and output directories before these commands are fully reproducible.

## Citation

TODO: Add citation information when the paper entry is ready.

## License

TODO: Add license information.
