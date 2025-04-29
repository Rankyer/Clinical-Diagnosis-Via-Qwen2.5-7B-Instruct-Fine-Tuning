import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import pandas as pd

# === Step 1: Model Configuration ===
# Define model parameters
MODEL_PATH = "./md/Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 5120  # Maximum input sequence length
DTYPE = None  # Automatically detect data type (e.g., float16 or bfloat16)
LOAD_IN_4BIT = False  # Whether to use 4-bit quantization to reduce memory usage

# Load the pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# === Step 2: Apply LoRA (Low-Rank Adaptation) ===
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Low-rank dimension
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,  # Dropout rate for LoRA
    bias="none",  # Bias type for LoRA
    use_gradient_checkpointing="unsloth",  # Gradient checkpointing for memory efficiency
    use_rslora=False,  # Disable rank-stabilized LoRA
    loftq_config=None,  # Optional LoftQ configuration
)

# === Step 3: Define Prompt Template ===
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Add EOS token to end of the prompt
EOS_TOKEN = tokenizer.eos_token

# Function to format prompts for training
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# === Step 4: Load and Preprocess Training Data ===
# Load training data from JSON file
DATA_PATH = "./data/SFT_train_data.json"
data = pd.read_json(DATA_PATH, lines=True)

# Add a fixed instruction for all training examples
data["instruction"] = (
    "你是一位临床经验丰富的医疗专业人员，面对每条病例时，请依据所提供的信息对病人情况进行逐步详细的分析，并据此判断出可能的诊断结果及其诊断依据。"
    "请特别注意结合患者的主诉、现病史、既往史、个人史、过敏史、婚育史及任何相关的体格检查和辅助检查结果。"
)

# Convert the pandas DataFrame to a Hugging Face Dataset
train_data = Dataset.from_pandas(data)

# Apply the formatting function to create prompts
dataset = train_data.map(formatting_prompts_func, batched=True)

# === Step 5: Fine-Tune the Model ===
# Configure training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Batch size per GPU
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate a larger batch size
    warmup_steps=100,               # Warmup steps for learning rate scheduler
    num_train_epochs=3,             # Number of training epochs
    learning_rate=2e-4,             # Learning rate
    fp16=not is_bfloat16_supported(),  # Use fp16 if bfloat16 is not supported
    bf16=is_bfloat16_supported(),      # Use bfloat16 if supported by the hardware
    logging_steps=100,              # Log training progress every 100 steps
    optim="adamw_8bit",             # Optimizer: AdamW with 8-bit precision
    weight_decay=0.01,              # Weight decay for regularization
    lr_scheduler_type="linear",     # Learning rate scheduler type
    output_dir="outputs_only_2",    # Directory for saving training outputs
    report_to="none",               # Disable reporting to external platforms
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",       # Field name in the dataset containing the formatted text
    max_seq_length=MAX_SEQ_LENGTH,  # Maximum sequence length for training
    dataset_num_proc=2,             # Number of processes for dataset preprocessing
    packing=False,                  # Disable sequence packing
    args=training_args,
)

# === Step 6: Train the Model ===
trainer_stats = trainer.train()