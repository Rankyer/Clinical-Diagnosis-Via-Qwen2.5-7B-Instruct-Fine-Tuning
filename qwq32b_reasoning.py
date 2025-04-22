import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import vllm
from vllm.lora.request import LoRARequest

# === Configuration ===
MODEL_NAME = "./md/Qwen/QwQ-32B-AWQ"
DATA_PATH = "./data/20250208181531_camp_data_step_1_without_answer.jsonl"
OUTPUT_JSON = "qwq_vllm_v1.json"
RESULT_TXT = "qwq_result.txt"
OLD_RESULT_TXT = "old_qwq_result.txt"

# SYSTEM_PROMPT = '''
# You are an experienced clinical professional. For each case, analyze the patient's condition step by step based on the provided information and determine possible diagnoses with reasoning.
# Output format (in JSON):
# {"diseases":"diagnosis result","reason":"diagnosis reasoning"}
# '''
SYSTEM_PROMPT = '''
你是一位临床经验丰富的医疗专业人员，面对每条病例时，请依据所提供的信息对病人情况进行逐步详细的分析，并据此判断出可能的诊断结果及其诊断依据。
请特别注意结合患者的主诉、现病史、既往史、个人史、过敏史、婚育史及任何相关的体格检查和辅助检查结果。

输出格式要求如下（以 JSON 形式输出）：
去掉JSON中的特殊符号，保持简洁清晰的文本结构
{"diseases":"诊断结果","reason":"诊断依据"}
'''

PROMPTS_TEMPLATE = '''
{}
'''

# === Initialize the LLM ===
llm = vllm.LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=2800,
    disable_log_stats=True,
)
tokenizer = llm.get_tokenizer()

# === Load the data ===
data = pd.read_json(DATA_PATH, lines=True)

def apply_template(row):
    """
    Construct the LLM input for each case.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPTS_TEMPLATE.format(row["feature_content"])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_response(response_text):
    """
    Parse the response to extract JSON content.
    """
    try:
        response_text = response_text.split("</think>")[-1].split("\n\n")[-1]
        if "```json" in response_text:
            response_text = response_text.replace("```", "").replace("json", "")
        return json.loads(response_text)
    except Exception:
        return {"diseases": "", "reason": ""}

def format_output(index, parsed_data, feature_content):
    """
    Format the output into the required JSON structure.
    """
    parsed_data.setdefault("diseases", "")
    parsed_data.setdefault("reason", "")
    return {
        "id": index,
        "diseases": parsed_data["diseases"],
        "reason": parsed_data["reason"],
        "feature_content": feature_content,
    }

# Prepare LLM input messages
data["messages"] = data.apply(apply_template, axis=1)

# Generate responses
responses = llm.generate(
    data["messages"],
    vllm.SamplingParams(n=1, skip_special_tokens=False, max_tokens=5120),
    use_tqdm=True,
)

# Save results
with open(OUTPUT_JSON, "w", encoding="utf-8") as output_file, \
     open(RESULT_TXT, "w", encoding="utf-8") as result_file, \
     open(OLD_RESULT_TXT, "w", encoding="utf-8") as old_result_file:

    for index, response in enumerate(responses):
        response_text = response.outputs[0].text
        old_result_file.write(response_text)

        parsed_data = parse_response(response_text)
        formatted_output = format_output(index, parsed_data, data.loc[index]["feature_content"])

        json.dump(formatted_output, output_file, ensure_ascii=False)
        output_file.write("\n")

        result_file.write(parsed_data.get("reason", "") + "\n")

print("Processing completed. Results saved.")