import pandas as pd
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# === Step 1: Data Loading ===
# Load the training data (unlabeled cases)
DATA_PATH = "./data/20250312115015_camp_data_step_2_without_answer.jsonl"
train_data = pd.read_json(DATA_PATH, lines=True)

# === Step 2: Model and LoRA Configuration ===
MODEL_DIR = "./md/Qwen2.5-7B-Instruct"
LORA_PATH_1 = "./checkpoint1"
LORA_PATH_2 = "./checkpoint2"

# System prompt: Instruction for the model
SYSTEM_PROMPT = '''
你是一位临床经验丰富的医疗专业人员，面对每条病例时，请依据所提供的信息对病人情况进行逐步详细的分析，并据此判断出可能的诊断结果及其诊断依据。请特别注意结合患者的主诉、现病史、既往史、个人史、过敏史、婚育史及任何相关的体格检查和辅助检查结果。
示例输出：
病例：
性别: 女
年龄: 45
主诉: 皮炎失眠高尿酸血症
现病史: 给予中药治疗。无发热、干咳、乏力、嗅（味）觉减退、鼻塞、流涕、咽痛、结膜炎、肌痛、腹泻
既往史: 
个人史: 无发病前14天内有病例报告社区的旅行史或居住史；无发病前14天内与新型冠状病毒感染的患者或无症状感染者有接触史；无发病前14天内曾接触过来自有病例报告社区的发热或有呼吸道症状的患者；无聚集性发病；近14天内无进口冷链食品接触史。
过敏史: 
婚育史: 
流行病史: 
体格检查: T36.5℃，P78次/分，R20次/分，BP142/78mmHg。心肺听诊未见异常，腹平软，无压痛。
辅助检查: 
输出：
{"diseases":"1. 高尿酸血症；2. 皮炎；3. 失眠；4. 原发性高血压（1级）","reason":"1. 主诉中提到高尿酸血症，提示患者血尿酸水平升高；\\n2. 主诉中提到皮炎，提示存在皮肤炎症，可能为过敏性或接触性皮炎；\\n3. 主诉中提到失眠，提示存在睡眠障碍，可能与精神压力或其他疾病有关；\\n4. 体格检查中血压142/78mmHg，收缩压≥140mmHg，提示存在1级高血压；\\n5. 无发热、感染等症状，生命体征基本正常，心肺腹部检查未见明显异常。"}
'''

# Prompt format for user-provided content
PROMPT_TEMPLATE = '''
{}
'''

# Initialize the model
llm = LLM(
    model_dir=MODEL_DIR,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True,
    enable_lora=True,
)

# === Step 3: Data Preprocessing ===
# Apply the prompt template to create input messages
def apply_template(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(row["feature_content"])},
    ]
    return llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Add processed messages to the dataset
train_data["messages"] = train_data.apply(apply_template, axis=1)

# === Step 4: Generate Results with LoRA ===
# Generate diseases predictions with LoRA 1
responses_diseases = llm.generate(
    train_data["messages"],
    SamplingParams(n=1, skip_special_tokens=False, max_tokens=5120),
    use_tqdm=True,
    lora_request=LoRARequest("lora1", 1, LORA_PATH_1),
)

# Generate reasons predictions with LoRA 2
responses_reasons = llm.generate(
    train_data["messages"],
    SamplingParams(n=1, skip_special_tokens=False, max_tokens=5120),
    use_tqdm=True,
    lora_request=LoRARequest("lora2", 1, LORA_PATH_2),
)

# === Step 5: Save Results ===
# Save the generated results to a JSON file
OUTPUT_FILE = "new_data_lora_vllm_v1.json"

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for idx, (response_d, response_r) in enumerate(zip(responses_diseases, responses_reasons)):
        # Extract text outputs
        diseases = response_d.outputs[0].text
        reason = response_r.outputs[0].text

        # Construct the result dictionary
        submit_data = {
            "id": idx,
            "reason": reason,
            "diseases": diseases,
            "feature_content": train_data.loc[idx]["feature_content"],
        }

        # Write to file
        json.dump(submit_data, f, ensure_ascii=False)
        f.write("\n")

# === Step 6: Validate Results ===
# Load and check the saved results for empty fields
data = pd.read_json(OUTPUT_FILE, lines=True)
empty_diseases = data[data["diseases"] == ""]
print(f"Number of empty disease results: {empty_diseases.shape[0]}")