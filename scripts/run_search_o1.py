# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from google_search import (
    google_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from evaluate import (
    run_evaluation, 
    extract_answer
)
from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction,   # !!!
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--google_subscription_key',
        type=str,
        required=True,
        help="google Search API subscription key."
    )

    parser.add_argument(
        '--google_endpoint',
        type=str,
        # default="https://api.bing.microsoft.com/v7.0/search",
        default="https://google.serper.dev/search",
        help="google Search API endpoint."
    )

    return parser.parse_args()

def main():
    print("[DEBUG] Starting main()")
    args = parse_args()
    print(f"[DEBUG] Arguments: {args}")

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    google_subscription_key = args.google_subscription_key
    google_endpoint = args.google_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key
    
    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 5
        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
            MAX_SEARCH_LIMIT = 10
            MAX_TURN = 15
        top_k = 10
        max_doc_len = 3000
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0

    # Data paths based on dataset
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    print('-----------------------')
    print(f'Using {dataset_name} {split} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths  建立缓存机制：search_cache.json (保存搜索查询到 Bing 结果的映射) 和 url_cache.json (保存 URL 到网页内容的映射)，避免重复请求。
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}  # URL 内容缓存，避免重复抓取，后续格式化文档时从中读取

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    print("[DEBUG] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("[DEBUG] Tokenizer 加载完毕")

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.search_o1'
            if dataset_name == 'gpqa' and (MAX_SEARCH_LIMIT != 5 or top_k != 10):
                output_dir = f'./outputs/runs.analysis/{dataset_name}.qwq.search_o1.{MAX_SEARCH_LIMIT}.{top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.search_o1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.search_o1'
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] Output directory: {output_dir}")

    # Initialize the LLM
    print("[DEBUG] 初始化 vLLM LLM.......")
    llm = LLM(
        model=model_path,
        # tensor_parallel_size=torch.cuda.device_count(),
        tensor_parallel_size=2,  
        gpu_memory_utilization=0.7,
    )
    print("[DEBUG] vLLM LLM 部署完毕")

    # ---------------------- Data Loading ----------------------
    print(f"[DEBUG] 从 {data_path} 加载数据集")
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
    print(f"[DEBUG] 从数据集加载 {len(filtered_data)} 条数据")

    # ---------------------- Batch Generation Function ----------------------
    # 批量生成推理链（reasoning chain）的函数，主要用于将网页内容转换为结构化的推理过程。
    def generate_webpage_to_reasonchain_batch(
        original_questions: List[str],  # 原始问题列表
        prev_reasonings: List[str],  # 之前的推理步骤（用于多步推理的上下文）
        search_queries: List[str],  # 搜索查询列表
        documents: List[str],  # 检索到的网页文档内容
        dataset_name: str,
        batch_output_records: List[Dict],  # 收集输出记录的列表（用于调试/日志）# New parameter to collect outputs
        max_tokens: int = 32768,
        coherent: bool = False,
    ) -> List[str]:
        """
        输入：原始问题列表、之前的推理步骤、搜索查询、文档列表。
        调用 LLM 为每个文档生成信息提取或推理更新。返回提取的信息列表。模型从网页中提取的精简有用信息，格式为 **Final Information**
        """
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 original_questions = {json.dumps(original_questions, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 prev_reasonings ={json.dumps(prev_reasonings, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 search_queries ={json.dumps(search_queries, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 documents= {json.dumps(documents, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 dataset_name ={json.dumps(dataset_name, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 batch_output_records= {json.dumps(batch_output_records, indent=4, ensure_ascii=False)} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 max_tokens= {max_tokens} ")
        print(f"[DEBUG] generate_webpage_to_reasonchain_batch()里面的 coherent ={json.dumps(coherent, indent=4, ensure_ascii=False)} ")
        user_prompts = [ # 将每个（前序推理、搜索查询、文档）组合转换为指令格式。
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]
        print(f"run_search_o1.py里面的  generate_webpage_to_reasonchain_batch()里面的  user_prompts:{user_prompts}")

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        # prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]
        # 开思考模式
        prompts = [tokenizer.apply_chat_template(
            [p], tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True}
        ) for p in prompts]

        output = llm.generate(  # 调用vLLM批量生成
            prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        )
        print("[DEBUG] llm.generate for webpage analysis completed")

        raw_outputs = [out.outputs[0].text for out in output]  # 原始生成文本
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]   # 从原始输出中提取结构化信息，mode='infogen'表示提取"**Final Information**"标记后的内容

        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e  # 提取后的精简信息（仅Final Information部分）
            })

        print(f"[DEBUG] generate_webpage_to_reasonchain_batch() 最终处理后的 extracted_infos= {json.dumps(extracted_infos, indent=4, ensure_ascii=False)} ")
        return extracted_infos

    # ---------------------- Preparation of Input Prompts ----------------------
    print("[DEBUG] 为所有items构建prompts...")
    input_list = []
    for item in filtered_data:
        question = item['Question']

        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:  # 走这个
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        # 开3.5思考
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True}  # 开启思考模式
        )
        input_list.append(prompt)

    if subset_num != -1:  # 截取数据集的前 subset_num 个样本，用于快速测试或调试。
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]
    print(f"[DEBUG] Total prompts prepared: {len(input_list)}")

    # Initialize active sequences
    active_sequences = [{
        'item': item,  # 原始数据项
        'prompt': prompt,  # 当前累积的提示字符串 (初始为指令+问题)
        'output': '',  # 模型生成的累积输出
        'finished': False,  # 是否完成
        'history': [],  # 历史消息列表 
        'search_count': 0,  # 已执行的搜索次数
        'executed_search_queries': set(),  # 已搜索过的查询集合
    } for item, prompt in zip(filtered_data, input_list)]
    print("[DEBUG] Active sequences initialized")

    # ---------------------- Set Max Tokens ----------------------
    if 'qwq' in model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            max_tokens = 32768
        else:
            max_tokens = 20480
    elif 'qwen3.5' in model_path.lower() or 'qwen3_5' in model_path.lower():
        max_tokens = 20480
    else:
        max_tokens = 8192
    print(f"[DEBUG] Max generation tokens set to {max_tokens}")

    # ---------------------- Generation Function ----------------------
    # 对一批序列调用 vLLM 生成，停止条件包含 END_SEARCH_QUERY。
    def run_generation(sequences: List[Dict], max_tokens: int) -> List:
        prompts = [s['prompt'] for s in sequences]
        print(f"[DEBUG] run_generation()里面的:  len(prompts): {len(prompts)}, max_tokens={max_tokens}")
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            repetition_penalty=repetition_penalty,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        output_list = llm.generate(prompts, sampling_params=sampling_params)
        print("[DEBUG] run_generation()里面的: llm.generate completed")
        return output_list

    # 正则提取两个标记之间的内容，用于获取模型输出的搜索查询。
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        print(f"run_search_o1.py里面的 extract_between() 入参text：{json.dumps(text, indent=4, ensure_ascii=False)}")
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            print(f"run_search_o1.py里面的 extract_between() 处理后返回：{json.dumps(matches[-1].strip(), indent=4, ensure_ascii=False)}")
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        解析模型输出的推理步骤 (格式 Step 1: ...)，根据新生成的分析结果【替换或删除】特定步骤，用于让模型在搜索后更新自己的推理链。处理 DELETE THIS STEP 标记。
          -  替换特定步骤的内容
          -  删除标记为 DELETE THIS STEP 的步骤
          -  保持步骤顺序

        Parameters:
        - origin_str (str): 原始推理步骤.
        - replace_str (str): 新的分析结果.

        Returns:
        - str: 更新后的推理步骤.
        """

        def parse_steps(text):  # 步骤解析器
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
                {
                1: "分析...\n补充内容1\n补充内容2",
                2: "搜索...\n结果分析",
                3: "结论"
            }
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")  # 匹配格式：Step 1: 、Step 12:  等，捕获步骤编号。
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)
        print(f"run_search_o1.py里面的 replace_recent_steps()里面的  parse_steps()里面的 origin_str：{json.dumps(origin_str, indent=4, ensure_ascii=False)}")
        print(f"run_search_o1.py里面的 replace_recent_steps()里面的  parse_steps()里面的 replace_str：{json.dumps(replace_str, indent=4, ensure_ascii=False)}")
        print(f"run_search_o1.py里面的 replace_recent_steps()里面的  parse_steps()里面的 origin_steps：{json.dumps(origin_steps, indent=4, ensure_ascii=False)}")
        print(f"run_search_o1.py里面的 replace_recent_steps()里面的  parse_steps()里面的 replace_steps：{json.dumps(replace_steps, indent=4, ensure_ascii=False)}")

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # 删除该步骤（如果存在）
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # 覆盖或新增该步骤
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        print(f"run_search_o1.py里面的 replace_recent_steps()里面的  处理后的 steps：{json.dumps(steps, indent=4, ensure_ascii=False)}")
        return new_reasoning_steps

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    print("[DEBUG] 启动主交互 循环...")
    print(f"run_search_o1.py里面的 active_sequences ：{active_sequences}")

    # Main loop until all sequences are finished or maximum turns reached
    while True:  # 主循环 (直到所有序列完成或达到最大轮次 MAX_TURN)
        # Identify sequences that need generation
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]
        print(f"[DEBUG] Main loop: turn={turn}, sequences_needing_generation={sequences_needing_generation}")

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"有 len(sequences_needing_generation)={len(sequences_needing_generation)} 的sequences需要生成...")
            outputs = run_generation(sequences_needing_generation, max_tokens) # 对未完成的序列，调用 LLM 生成，使用停止标记 END_SEARCH_QUERY (即 "<|end_search_query|>") 和 EOS。返回生成的文本。
            print(f"生成完成，处理输出outputs={outputs}…")

            # Initialize batch variables
            batch_relevant_info = []  # Google 搜索结果的前 top_k 条，包含 URL、标题、摘要
            batch_original_questions = []
            batch_prev_reasonings = []  # 收集截断后的历史推理步骤，作为 generate_webpage_to_reasonchain_batch 的 prev_reasonings 参数
            batch_search_queries = []  # 收集搜索查询，作为 generate_webpage_to_reasonchain_batch 的 search_queries 参数
            batch_documents = []  # 收集格式化后的网页文档，作为 generate_webpage_to_reasonchain_batch 的 documents 参数输入
            batch_sequences = []  # 收集需要批量处理的序列，决定哪些序列会进入 generate_webpage_to_reasonchain_batch 进行网页分析

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()  # 所有需要抓取的 URL 集合，决定 fetch_page_content 要抓取哪些网页，结果存入 url_cache
            url_snippets = {}
            url_sequence_map = {}  # 将URL映射到需要它的序列列表

            # Process each sequence and collect URLs
            # 将生成文本追加到 seq['prompt'], seq['output'], seq['history']
            for seq, out in zip(sequences_needing_generation, outputs):
                print(f"run_search_o1.py里面的 main主循环 里面的for seq, out in zip(sequences_needing_generation, outputs): \nseq={seq}\nout={out}")
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)# 使用正则提取 BEGIN_SEARCH_QUERY 和 END_SEARCH_QUERY 之间的搜索查询字符串

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):  # 情况 A：存在搜索查询且输出以 END_SEARCH_QUERY 结尾
                    print(f"run_search_o1.py里面的 main() 检测到了以 END_SEARCH_QUERY 结尾 seq['output'].rstrip(): {seq['output'].rstrip()}，\n检测到了的search_query 是: '{search_query}")
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']: # 检查是否达到最大搜索次数或重复查询,若可搜索:
                        print(f"进入  if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']")
                        print(f"执行search Query 查询，search_query是: {search_query}")
                        # Execute search, use cache if available
                        if search_query in search_cache and search_cache[search_query]:  # 调用 bing_web_search (带缓存)
                            results = search_cache[search_query]  
                            print(f"使用缓存的搜索结果进行查询: \"{search_query}\"")
                        else:
                            try:
                                print(f"没有缓存，使用google搜索: \"{search_query}\"")
                                results = google_web_search(search_query, google_subscription_key, google_endpoint, market='en-US', language='en')
                                search_cache[search_query] = results
                            except Exception as e:
                                print(f"搜索查询时出错 '{search_query}': {e}")
                                # search_cache[search_query] = {}  # 异常时不写入缓存
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:top_k]  # 调用 extract_relevant_info 提取前 top_k 个结果
                        print(f"run_search_o1.py里面的 extract_relevant_info()处理后，截取top_k个，得到的relevant_info：{json.dumps(relevant_info, indent=4, ensure_ascii=False)}")
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached  收集所有需要抓取的 URL (过滤已缓存的)
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]  # 过滤掉缓存后真正需要抓取的 URL
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]
                        print(f"[DEBUG] URLs to fetch: {len(urls_to_fetch_filtered)} (cached: {len(urls_to_fetch)-len(urls_to_fetch_filtered)})")

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = ""  # 截断后的推理历史（保留前1步 + 后4步 + 含搜索标签的步骤），控制 prompt 长度
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:  # 不截断
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:  #步骤大于5， 进入智能截断模式
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                # 保留条件：第一步（初始分析，上下文基础） 或 最后4步（最近的推理，与当前决策最相关） 或 包含搜索标签的步骤（BEGIN_SEARCH_QUERY 记录了"问过什么"，避免重复搜索；BEGIN_SEARCH_RESULT 记录了"获得了什么信息"，核心证据）
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    # # 省略中间步骤，但只添加一次 "..."，避免重复
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')  # 去掉末尾多余的换行符。

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)  # 集合，防止重复搜索同一查询

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        print(f"搜索达到上限seq['search_count']：{seq['search_count']}")
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        print(f"您已经搜索了这个查询。请参考以往的结果。")
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")

                else:  # 没有搜索查询或未以结束标签结尾 → 标记序列为 finished
                    # If no search query needs to be executed, mark the sequence as finished
                    print(f"run_search_o1.py里面的 没有搜索查询或未以结束标签结尾 → 标记序列为 finished")
                    seq['finished'] = True  # 	布尔标志，控制主循环是否继续生成
                    print("Sequence marked as complete.")

            # 批量抓取网页内容  。Batch fetch all URLs at once to optimize speed 
            if all_urls_to_fetch:
                print(f"[DEBUG] 开始爬取 {len(all_urls_to_fetch)}个 URLs...")
                try:
                    fetched_contents = fetch_page_content(  # 并发获取每个 URL 的文本 (使用 Jina 或 BeautifulSoup)
                        list(all_urls_to_fetch),
                        use_jina=use_jina,
                        jina_api_key=jina_api_key, # 这里没有使用extract_snippet_with_context，因为没传snippet参数
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly。如果打开注释——破坏缓存通用性：url_cache[url] 存储的是针对当前特定 snippet 提取出的几千字上下文。下次别的查询搜到同一个 URL，从缓存读出来的就是错的/无关的文本。还有就是会重复执行extract_snippet_with_context 
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error 爬取全文出错: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():  # 更新 url_cache
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing。将搜索结果格式化为模型可读的文档，用于批量送入 LLM 进行推理
            # batch_relevant_info（多组搜索结果）
                # │
                # ├── 第1组搜索结果（多个网页）
                # │       ├── 网页1 → 提取上下文 → 格式化
                # │       ├── 网页2 → 提取上下文 → 格式化
                # │       └── ... → 拼接为 formatted_documents
                # │
                # ├── 第2组搜索结果（多个网页）
                # │       └── ... → 拼接为 formatted_documents
                # │
                # └── batch_documents（最终列表，每个元素是一组格式化文档）
            for relevant_info in batch_relevant_info:  # batch_relevant_info批量查询的搜索结果，每个元素是一组网页列表。relevant_info[0] 是第1个查询的10个网页，relevant_info[1] 是第2个查询的10个网页
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','') # 搜索结果的 snippet 常包含高亮标签 <b>...</b>。去掉后得到纯文本摘要
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len) # 根据 snippet 提取相关上下文 (限制长度 max_doc_len)
                    print(f"run_search_o1.py里面的 relevant_info循环 里面的 extract_snippet_with_context结果：success={success}，filtered_context={json.dumps(filtered_context, indent=4, ensure_ascii=False)}")
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len*2] # # 退化为前 max_doc_len*2 个字符

                    doc_info['context'] = context  # 给原始字典新增/覆盖 context 字段，供后续使用
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents) # 最终 batch_documents 的长度 = 查询批次大小，每个元素是一组格式化的网页文档字符串。
            print(f"run_search_o1.py里面的 relevant_info循环 处理后的 batch_documents 结果：batch_documents={json.dumps(batch_documents, indent=4, ensure_ascii=False)}")

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"用generate_webpage_to_reasonchain_batch 处理 len(batch_sequences)={len(batch_sequences)}个 sequences ...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(  # 将所有序列的 (原始问题、之前的推理步骤、搜索查询、格式化文档) 打包，调用 generate_webpage_to_reasonchain_batch 得到分析结果。
                    original_questions=batch_original_questions, # webpage_analyses 是与输入等长的列表，每个元素是 LLM 对对应网页内容的分析文本。
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=dataset_name,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    max_tokens=max_tokens,
                )
                print("Batch generation completed, assigning outputs to sequences...")
                print(f"将所有序列的 (原始问题、之前的推理步骤、搜索查询、格式化文档) 打包，调用 generate_webpage_to_reasonchain_batch 得到分析结果---webpage_analyses:{webpage_analyses}")

                # 将分析结果以 BEGIN_SEARCH_RESULT ... END_SEARCH_RESULT 的格式追加到对应序列的 prompt 中。
                for seq, analysis in zip(batch_sequences, webpage_analyses):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

        # Check if all sequences are finished
        # 检查退出条件：如果所有 active_sequences 的 finished 都为 True，或者 turn >= MAX_TURN，则退出循环。
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            print("[DEBUG] All sequences finished. Exiting loop.")
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break
            print(f"[DEBUG] {len(unfinished)} sequences still active, continuing to next turn...")

    total_time = time.time() - start_time
    print(f"[DEBUG] 总的执行时间: {total_time:.2f} seconds")

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # 结束后的处理
    # Define output JSON file path
    t = time.localtime()
    # 保存 batch_output_records (包含每次生成网页分析的原始输出和提取信息) 到 JSON 文件。
    batch_output_file = os.path.join(output_dir, f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)
    print(f"Batch outputs saved to {batch_output_file}")

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    # Run evaluation。  调用 run_evaluation 评估最终输出 (从 seq['output'] 提取答案，计算准确率等)。
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：filtered_data：{json.dumps(filtered_data, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：input_list：{json.dumps(input_list, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：output_list：{json.dumps(output_list, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：dataset_name：{json.dumps(dataset_name, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：output_dir：{json.dumps(output_dir, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：total_time：{json.dumps(total_time, indent=4, ensure_ascii=False)}")
    print(f"run_search_o1.py里面的 调用 run_evaluation 评估最终输出 的参数：split：{json.dumps(split, indent=4, ensure_ascii=False)}")

    run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split)

    # ---------------------- Update Search and URL Cache。 将本次运行产生的搜索缓存和 URL 缓存更新保存到本地文件----------------------
    print('更新搜索和 URL缓存…')
    # Load existing caches or initialize empty dictionaries
    # 加载已有的搜索缓存
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    # 加载已有的URL 缓存
    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    # 合并缓存。两个字典合并。如果本次运行和磁盘缓存有相同的搜索查询，但返回结果不同（比如网页内容更新了），旧缓存会覆盖新缓存。这可能导致数据不一致。
    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()  # 将合并后的缓存写回磁盘

    print("完成！！！！")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()


# # 问题 → 构建 prompt → 模型生成
#                      ↓
#             生成 <|begin_search_query|>query<|end_search_query|>
#                      ↓
#             执行 Bing 搜索 (缓存) → 提取 top_k 结果
#                      ↓
#             批量抓取网页内容 (缓存)
#                      ↓
#             为每个网页提取 snippet context → 构建文档 JSON
#                      ↓
#             模型分析文档 (generate_webpage_to_reasonchain_batch) → 追加分析结果
#                      ↓
#             继续循环 (模型可能再次输出搜索查询或直接给出答案)
#                      ↓
#             达到停止条件 → 结束 → 评估