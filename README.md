<p align="center">
  <img src="assets/armor_logo.png" alt="ARMOR logo" width="220">
</p>

<h1 align="center">ARMOR</h1>

<p align="center">
  <strong>Adaptive Regularized Mixture Optimization for Retrievers</strong>
</p>

<p align="center">
  Low-resource domain adaptation for retrieval-augmented generation.
</p>

---

## Overview

ARMOR is a retriever-centric adaptation method for low-resource domain RAG. Instead of fine-tuning the generator, ARMOR keeps the generator and document index fixed and adapts the query encoder, concentrating limited supervision on the component that controls which evidence is shown to the model.

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

Across the paper's low-resource domain RAG experiments, ARMOR improves over frozen Base RAG, with particularly visible gains when the generator has less capacity and relies more heavily on retrieved evidence.

<p align="center">
  <img src="assets/model_performance_comparison.png" alt="ARMOR results across generator backbones" width="760">
</p>

<p align="center">
  <em>ARMOR improves RAG performance across generator backbones, with the largest gains for smaller generators.</em>
</p>

ARMOR's learned temperatures also show meaningful training dynamics: the retrieval objective sharpens during optimization, while query-distillation regularization helps constrain drift from the frozen base embedding space.

<p align="center">
  <img src="assets/training_dynamics.png" alt="ARMOR adaptive temperature and regularization dynamics" width="850">
</p>

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
