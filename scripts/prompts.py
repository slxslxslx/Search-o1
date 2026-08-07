

def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_gpqa_search_o1_instruction_ZH(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网页搜索能力的推理助手，旨在通过主动搜索帮助用户准确回答问题。你可以使用以下特殊工具：\n\n"
        "- **执行搜索**：请输出 `<|begin_search_query|>你的搜索查询内容<|end_search_query|>`。\n"
        "  随后，系统将搜索并分析相关网页，并以 `<|begin_search_result|>...搜索结果...<|end_search_result|>` 的格式向你提供提炼后的有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。最大搜索尝试次数限制为 {MAX_SEARCH_LIMIT} 次。\n\n"
        "一旦你获得了所需的所有信息，请继续你的推理过程。\n\n"
        "示例：\n"
        "问题：“pp III 中微子的能量范围是多少？”\n"
        "助手思考步骤：\n"
        "- 我需要查阅关于 pp III 中微子（pp III neutrinos）能量范围的详细科学数据。\n\n"
        "助手：\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "（系统返回相关网页的处理后信息）\n\n"
        "助手结合新信息继续推理……\n\n"
        "请严格遵守以下规则：\n"
        "1. 必须精确使用 `<|begin_search_query|>` 发起网页搜索请求，并严格以 `<|end_search_query|>` 结束，不要添加任何多余字符。\n"
        "2. 搜索完成后，请务必结合获取到的信息继续你的推理和解答。\n\n"
    )

def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_math_search_o1_instruction_ZH(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网页搜索能力的推理助手，旨在通过主动搜索帮助用户准确回答数学问题。你可以使用以下特殊工具：\n\n"
        "- **执行搜索**：请输出 `<|begin_search_query|>你的搜索查询内容<|end_search_query|>`。\n"
        "  随后，系统将搜索并分析相关网页，并以 `<|begin_search_result|>...搜索结果...<|end_search_result|>` 的格式向你提供提炼后的有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。最大搜索尝试次数限制为 {MAX_SEARCH_LIMIT} 次。\n\n"
        "一旦你获得了所需的所有信息，请停止搜索并继续你的推理过程。\n\n"
        "示例：\n"
        "问题：“如何计算 e^(x^2) dx 的积分？”\n"
        "助手思考步骤：\n"
        "- 这是一个非初等积分问题，我需要查阅关于 e^(x^2) 积分的特殊数学技巧或相关函数（如误差函数）。\n\n"
        "助手：\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "（系统返回相关网页的处理后信息）\n\n"
        "助手结合新信息继续推理……\n\n"
        "请严格遵守以下规则：\n"
        "1. 必须精确使用 `<|begin_search_query|>` 发起网页搜索请求，并严格以 `<|end_search_query|>` 结束，不要添加任何多余字符。\n"
        "2. 搜索完成后，请务必结合获取到的信息继续你的数学推导和解答。\n\n"
    )

def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_code_search_o1_instruction_ZH(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网页搜索能力的推理助手，旨在通过主动搜索帮助用户准确回答算法与代码问题。你可以使用以下特殊工具：\n\n"
        "- **执行搜索**：请输出 `<|begin_search_query|>你的搜索查询内容<|end_search_query|>`。\n"
        "  随后，系统将搜索并分析相关网页，并以 `<|begin_search_result|>...搜索结果...<|end_search_result|>` 的格式向你提供提炼后的有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。最大搜索尝试次数限制为 {MAX_SEARCH_LIMIT} 次。\n\n"
        "一旦你获得了所需的所有信息，请停止搜索并继续你的推理过程。\n\n"
        "示例：\n"
        "问题：“在给定的树中，求包含所有指定顶点的斯坦纳树（Steiner tree）的最小顶点数。”\n"
        "助手思考步骤：\n"
        "- 这是一个图论/算法问题。我需要查阅“树结构上的斯坦纳树问题”的定义，以及如何计算在给定树中包含所有指定终端顶点所需的最小顶点数的算法。\n\n"
        "助手：\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "（系统返回相关网页的处理后信息）\n\n"
        "助手结合新信息继续推理……\n\n"
        "请严格遵守以下规则：\n"
        "1. 必须精确使用 `<|begin_search_query|>` 发起网页搜索请求，并严格以 `<|end_search_query|>` 结束，不要添加任何多余字符。\n"
        "2. 搜索完成后，请务必结合获取到的信息继续你的算法设计和代码解答。\n\n"
    )

# 生成一个结构化提示词（Prompt），用于指导大语言模型（LLM）分析网页内容并整合到推理链中。将"网页搜索 → 信息筛选 → 推理链更新"封装为标准化提示词的函数，是构建自主研究型 Agent（如 ReAct、Reflexion 等架构）的关键组件。
# 这是一个Agent 系统中的"工具调用指令模板"
# 参数：
    # prev_reasoning	之前的推理步骤（已有的思考过程）
    # search_query	当前搜索查询（需要解答的具体问题）
    # document	搜索到的网页内容（原始素材）
# 生成的提示词结构：
# 任务指令
# ├── 分析网页（要求逐页审查）
# ├── 提取相关信息（筛选与查询相关的事实）
# └── 输出格式（两种分支）
#     ├── 有帮助 → 输出 "**Final Information**" + 内容
#     └── 无帮助 → 输出 "**Final Information**" + "No helpful information found."
def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

def get_webpage_to_reasonchain_instruction_ZH(prev_reasoning, search_query, document):
    return f"""**任务指令：**

    你的任务是根据以下输入阅读并分析网页：**先前的推理步骤**、**当前搜索查询**和**搜索到的网页**。你的目标是从**搜索到的网页**中提取与**当前搜索查询**相关的有用信息，并将这些信息无缝整合到**先前的推理步骤**中，以继续针对原始问题进行推理。

    **指导原则：**

    1. **分析搜索到的网页：**
    - 仔细查看每个搜索到的网页的内容。
    - 识别与**当前搜索查询**相关的事实信息，这些信息可以帮助推进原始问题的推理过程。

    2. **提取相关信息：**
    - 从搜索到的网页中选择能够直接推进**先前推理步骤**的信息。
    - 确保提取的信息准确且相关。

    3. **输出格式：**
    - **如果网页提供了对当前搜索查询有帮助的信息：** 以下面的格式呈现信息，以 `**Final Information**` 开头。
    
    **Final Information**
    
    [有帮助的信息]
    
    - **如果网页没有提供对当前搜索查询有帮助的信息：** 输出以下文本。
    
    **Final Information**
    
    No helpful information found.

    **输入：**
    - **先前的推理步骤：**  
    {prev_reasoning}

    - **当前搜索查询：**  
    {search_query}

    - **搜索到的网页：**  
    {document}

    现在你应该分析每个网页，并基于当前搜索查询"{search_query}"和先前的推理步骤，找出有帮助的信息。
"""

def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_singleqa_search_o1_instruction_ZH(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以执行网页搜索以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 执行搜索：编写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "随后，系统将搜索并分析相关网页，然后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以重复搜索过程多次。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦你获得了所需的全部信息，请继续你的推理过程。\n\n"
        "示例：\n"
        "问题：\"谁获得了第一个诺贝尔物理学奖？\"\n"
        "助手的思考步骤：\n"
        "- 我需要查明谁被授予了第一个诺贝尔物理学奖。\n\n"
        "助手：\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "（系统从相关网页返回处理后的信息）\n\n"
        "助手继续结合新信息进行推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网页搜索请求，并以 <|end_search_query|> 结束。\n"
        "- 完成搜索后，继续你的推理过程。\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_multiqa_search_o1_instruction_ZH(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以执行网页搜索以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 执行搜索：编写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "随后，系统将搜索并分析相关网页，然后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以重复搜索过程多次。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦你获得了所需的全部信息，请继续你的推理过程。\n\n"
        "示例：\n"
        "问题：\"Alice David 是某款电子游戏中 Lara Croft 的配音演员，这款游戏由哪家公司开发？\"\n"
        "助手的思考步骤：\n"
        "- 我需要查明谁为电子游戏中的 Lara Croft 配音。\n"
        "- 然后，我需要确定那款电子游戏是由哪家公司开发的。\n\n"
        "助手：\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "（系统从相关网页返回处理后的信息）\n\n"
        "助手思考：搜索结果表明，Alice David 是某款特定电子游戏中 Lara Croft 的配音演员。现在，我需要查明那款游戏是由哪家公司开发的。\n\n"
        "助手：\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "（系统从相关网页返回处理后的信息）\n\n"
        "助手继续结合新信息进行推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网页搜索请求，并以 <|end_search_query|> 结束。\n"
        "- 完成搜索后，继续你的推理过程。\n\n"
    )
    
def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

def get_singleqa_rag_agent_instruction_ZH(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "你是一个具备网络搜索和网页内容获取能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会调用网络搜索 API，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式返回搜索结果。\n"
        "  搜索结果将包含网页列表，包括标题、URL 和摘要（但不包含完整内容）。\n\n"
        "- 收到搜索结果后，如果你需要一个或多个特定 URL 的更详细信息，请写 <|begin_url|> url1, url2, ... <|end_url|>。\n"
        "  系统将获取这些 URL 的完整页面内容，并以 <|begin_full_page|> ...完整页面内容... <|end_full_page|> 的格式返回给你。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        f"你可以获取最多 {MAX_URL_FETCH} 个 URL 的详细信息。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"谁获得了第一届诺贝尔物理学奖？\"\n"
        "助手思考步骤：\n"
        "- 我需要找出谁获得了第一届诺贝尔物理学奖。\n\n"
        "助手：\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提到了几个 URL。我想获取其中一个的完整详细信息。\n\n"
        "助手：\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "（系统返回完整页面内容）\n\n"
        "助手：\n"
        "<|begin_full_page|> ...完整页面内容... <|end_full_page|>\n\n"
        "现在助手已有足够信息，可以继续推理。\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 使用 <|begin_url|> 请求完整页面内容，并以 <|end_url|> 结束。\n"
        "- 信息获取完成后，继续你的推理。\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention the company that developed the video game featuring Alice David as Lara Croft.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

def get_multiqa_rag_agent_instruction_ZH(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "你是一个具备网络搜索和网页内容获取能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会调用网络搜索 API，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式返回搜索结果。\n"
        "  搜索结果将包含网页列表，包括标题、URL 和摘要（但不包含完整内容）。\n\n"
        "- 收到搜索结果后，如果你需要一个或多个特定 URL 的更详细信息，请写 <|begin_url|> url1, url2, ... <|end_url|>。\n"
        "  系统将获取这些 URL 的完整页面内容，并以 <|begin_full_page|> ...完整页面内容... <|end_full_page|> 的格式返回给你。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        f"你可以获取最多 {MAX_URL_FETCH} 个 URL 的详细信息。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"Alice David 是哪家公司开发的视频游戏中 Lara Croft 的配音演员？\"\n"
        "助手思考步骤：\n"
        "- 我需要找出谁为视频游戏中的 Lara Croft 配音。\n"
        "- 然后，我需要确定是哪家公司开发了那款视频游戏。\n\n"
        "助手：\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提供了 Lara Croft 的配音演员名字。我需要确认 Alice David 是否是其中之一。\n\n"
        "助手：\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果表明 Alice David 是某款特定视频游戏中 Lara Croft 的配音。现在我需要找出是哪家公司开发了那款游戏。\n\n"
        "助手：\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提到了由 Alice David 为 Lara Croft 配音的视频游戏的开发公司。\n\n"
        "助手：\n"
        "<|begin_url|>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html<|end_url|>\n\n" 
        "（系统返回完整页面内容）\n\n"
        "助手：\n"
        "<|begin_full_page|> ...完整页面内容... <|end_full_page|>\n\n"
        "现在助手已有足够信息，可以继续推理。\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 使用 <|begin_url|> 请求完整页面内容，并以 <|end_url|> 结束。\n"
        "- 信息获取完成后，继续你的推理。\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

def get_gpqa_rag_agent_instruction_ZH(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "你是一个具备网络搜索和网页内容获取能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会调用网络搜索 API，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式返回搜索结果。\n"
        "  搜索结果将包含网页列表，包括标题、URL 和摘要（但不包含完整内容）。\n\n"
        "- 收到搜索结果后，如果你需要一个或多个特定 URL 的更详细信息，请写 <|begin_url|> url1, url2, ... <|end_url|>。\n"
        "  系统将获取这些 URL 的完整页面内容，并以 <|begin_full_page|> ...完整页面内容... <|end_full_page|> 的格式返回给你。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        f"你可以获取最多 {MAX_URL_FETCH} 个 URL 的详细信息。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"pp III 中微子的能量范围是多少？\"\n"
        "助手思考步骤：\n"
        "- 我可能需要查阅关于 pp III 中微子的详细信息。\n\n"
        "助手：\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提到了一些 URL。我想获取其中一个的完整详细信息。\n\n"
        "助手：\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "（系统返回完整页面内容）\n\n"
        "助手：\n"
        "<|begin_full_page|> ...完整页面内容... <|end_full_page|>\n\n"
        "现在助手已有足够信息，可以继续推理。\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 使用 <|begin_url|> 请求完整页面内容，并以 <|end_url|> 结束。\n"
        "- 信息获取完成后，继续你的推理。\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

def get_math_rag_agent_instruction_ZH(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "你是一个具备网络搜索和网页内容获取能力的推理助手，可以帮助你准确回答用户的数学相关问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会调用网络搜索 API，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式返回搜索结果。\n"
        "  搜索结果将包含网页列表，包括标题、URL 和摘要（但不包含完整内容）。\n\n"
        "- 收到搜索结果后，如果你需要一个或多个特定 URL 的更详细信息，请写 <|begin_url|> url1, url2, ... <|end_url|>。\n"
        "  系统将获取这些 URL 的完整页面内容，并以 <|begin_full_page|> ...完整页面内容... <|end_full_page|> 的格式返回给你。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        f"你可以获取最多 {MAX_URL_FETCH} 个 URL 的详细信息。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"如何计算 e^(x^2) dx 的积分？\"\n"
        "助手思考步骤：\n"
        "- 我可能需要查阅积分 e^(x^2) 的方法。\n\n"
        "助手：\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提到了一些 URL。我想获取其中一个的完整详细信息。\n\n"
        "助手：\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "（系统返回完整页面内容）\n\n"
        "助手：\n"
        "<|begin_full_page|> ...完整页面内容... <|end_full_page|>\n\n"
        "现在助手已有足够信息，可以继续推理。\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 使用 <|begin_url|> 请求完整页面内容，并以 <|end_url|> 结束。\n"
        "- 信息获取完成后，继续你的推理。\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do I implement a binary search algorithm in Python?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

def get_code_rag_agent_instruction_ZH(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "你是一个具备网络搜索和网页内容获取能力的推理助手，可以帮助你准确回答用户的编程相关问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会调用网络搜索 API，并以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式返回搜索结果。\n"
        "  搜索结果将包含网页列表，包括标题、URL 和摘要（但不包含完整内容）。\n\n"
        "- 收到搜索结果后，如果你需要一个或多个特定 URL 的更详细信息，请写 <|begin_url|> url1, url2, ... <|end_url|>。\n"
        "  系统将获取这些 URL 的完整页面内容，并以 <|begin_full_page|> ...完整页面内容... <|end_full_page|> 的格式返回给你。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n"
        f"你可以获取最多 {MAX_URL_FETCH} 个 URL 的详细信息。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"如何在 Python 中实现二分搜索算法？\"\n"
        "助手思考步骤：\n"
        "- 我可能需要查阅 Python 中二分搜索的实现细节。\n\n"
        "助手：\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "（系统返回搜索结果）\n\n"
        "助手：\n"
        "<|begin_search_result|> ...不包含完整页面的搜索结果... <|end_search_result|>\n\n"
        "助手思考：搜索结果提到了一些 URL。我想获取其中一个的完整详细信息。\n\n"
        "助手：\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "（系统返回完整页面内容）\n\n"
        "助手：\n"
        "<|begin_full_page|> ...完整页面内容... <|end_full_page|>\n\n"
        "现在助手已有足够信息，可以继续推理。\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 使用 <|begin_url|> 请求完整页面内容，并以 <|end_url|> 结束。\n"
        "- 信息获取完成后，继续你的推理。\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )

def get_naive_rag_instruction_ZH(question, documents):
    return (
        "你是一个知识渊博的助手，使用提供的文档来回答用户的问题。\n\n"
        "问题：\n"
        f"{question}\n"
        "文档：\n"
        f"{documents}\n"
    )


def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_openqa_ZH(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下问题。'
            '你应该以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下问题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt


def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math_ZH(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下数学问题。'
            '你应该以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下数学问题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt


def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt


def get_task_instruction_multi_choice_ZH(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下选择题。'
            '你应该以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。\n\n'
            f'问题：\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            '请回答以下选择题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。你的最终选择应该是字母 A、B、C 或 D 中的一个，不要包含任何答案内容。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下选择题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

def get_task_instruction_code_ZH(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '生成一个正确的 Python 程序，使其通过给定问题的所有测试。'
            '你应该在使用三重反引号的 Python 代码块中提供最终代码（```python\n'
            'YOUR_CODE\n'
            '```）。\n\n'
            f'问题标题：{question_title}\n\n'
            f'问题陈述：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '你将收到一个问题（问题规范），并生成一个符合规范且通过所有测试的正确 Python 程序。'
            f'你应该一步一步思考来解决它。\n\n问题：\n{question}\n\n'
            '从标准输入读取输入，解决问题并将答案写入标准输出（不要直接在样本输入上测试）。将你的代码用以下分隔符括起来。\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt