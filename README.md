# Medical Diagnosis Model Fine-tuning (Lora)

This repository implements a two-stage approach for medical diagnosis prediction using large language models (LLMs). The system first generates synthetic diagnostic data with a 32B parameter model, then fine-tunes a more efficient 7B parameter model using LoRA (Low-Rank Adaptation).

## Key Features

- **Two-Stage Architecture**:
  - **Data Generation**: Uses QwQ-32B-AWQ model for synthetic medical case analysis
  - **Model Fine-tuning**: Adapts Qwen2.5-7B-Instruct model with LoRA for efficient specialization
- **Medical Diagnosis Capabilities**:
  - Multi-disease prediction with supporting reasoning
  - Comprehensive analysis of patient histories and clinical findings
  - Structured JSON output for easy integration

## Workflow Overview

### Stage 1: Synthetic Data Generation (`qwq32b_reasoning.py`)
- **Input**: Raw medical cases (`*.jsonl`)
- **Process**:
  1. Initializes QwQ-32B-AWQ model with vLLM
  2. Generates diagnostic predictions with clinical reasoning
  3. Outputs structured JSON with diseases and rationale
- **Output**:
  - `qwq_vllm_v1.json`: Full structured results
  - `*.txt` files: Raw model outputs

### Stage 2: Model Fine-tuning (`qwq7b_lora_finetuning.py`)
- **Key Components**:
  - LoRA Configuration (r=16, alpha=32)
  - Custom medical prompt template
  - 4-bit quantization support
- **Training**:
  - 3 epochs with linear learning rate scheduler
  - Batch size 2 with gradient accumulation
  - AdamW 8-bit optimizer

### Stage 3: Efficient Inference (`qwq7b_reasoning.py`)
- **Features**:
  - Dual LoRA adapters for disease/reason prediction
  - Context-aware clinical analysis
  - Input validation and empty result detection
- **Output**: `new_data_lora_vllm_v1.json` with formatted predictions

<!-- ## Installation

```bash
git clone [your-repository-url]
cd [repository-name]

# Install dependencies
pip install -r requirements.txt -->