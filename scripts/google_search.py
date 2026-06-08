import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)
proxies = {"http": "socks5h://127.0.0.1:1824", "https": "socks5h://127.0.0.1:1824"}

# 移除输入文本中的所有标点符号
def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

# 计算两个单词集合（真实集合与预测集合）之间的 F1 分数，用于衡量集合的重叠程度。若交集为 0 则返回 0.0，否则计算精确率、召回率并返回调和平均数
def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

# 从完整文本中定位最匹配给定摘要（snippet）的句子，并提取该句子前后各 context_chars 个字符作为上下文。返回布尔值表示是否成功，以及提取到的上下文文本。若找不到匹配句子，则返回完整文本的前 context_chars * 2 个字符。
def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): 从网页抓取的完整文本内容
        snippet (str): 搜索引擎返回的摘要片段（通常几十到几百字）
        context_chars (int): 在匹配句子前后各截取多少字符，默认 2500

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的len(full_text)：{len(full_text)}")
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的snippet：{snippet}")
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的context_chars：{context_chars}")
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)  # 去掉标点
        snippet_words = set(snippet.split())   # 拆成单词集合
        # 原始 snippet: "Citibank was founded in 1812, in New York."
        # 处理后: {'citibank', 'was', 'founded', 'in', '1812', 'new', 'york'}

        best_sentence = None  # 当前找到的最佳匹配句子
        best_f1 = 0.2  # 最佳匹配的 F1 分数阈值（低于 0.2 不算匹配）。只有当句子与摘要的单词重叠度 F1 分数 > 0.2 时，才认为是有效匹配。这过滤掉完全不相关的句子。

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # 用 NLTK 的 sent_tokenize 把长文本拆成句子列表。

        # 遍历句子，找最佳匹配
        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:   # # 如果找到了匹配的句子
            print(f"google_search.py里面的 extract_snippet_with_context() 找到匹配snippet的句子！！！best_sentence：{best_sentence}")
            # # 找到最佳句子在原文中的位置
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            # # 前后各扩展 context_chars 个字符
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            # 截取上下文
            context = full_text[start_index:end_index]
            print(f"google_search.py里面的 extract_snippet_with_context() 找到匹配snippet的句子！！！best_sentence：{best_sentence}\n返回的content：{context}")
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            print(f"google_search.py里面的 extract_snippet_with_context() 找不到匹配snippet的句子")
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

# 从给定的 URL（网页或 PDF）中提取文本内容。
# 若 use_jina=True 则通过 Jina AI 服务获取 Markdown 格式内容并清理；否则用 BeautifulSoup 解析 HTML 并提取纯文本。
# 如果提供了 snippet 参数，会调用 extract_snippet_with_context 获取与摘要相关的上下文；否则返回前 8000 个字符。
# 对 PDF 文件会调用 extract_pdf_text 专门处理。该函数处理各种网络异常并返回错误信息。
def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            # # response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # response = requests.get(f'https://r.jina.ai/{url}' , headers=jina_headers, proxies=proxies).text
            try:
                resp = requests.get(
                    f'https://r.jina.ai/{url}',
                    headers=jina_headers,
                    proxies=proxies,
                    timeout=300  # 加个超时，别挂死
                )
                resp.raise_for_status()  # 4xx/5xx 会抛异常
                response_text = resp.text
            except requests.exceptions.Timeout:
                return f"Error: Jina request timed out for {url}"
            except requests.exceptions.HTTPError as e:
                return f"Error: Jina HTTP error {e.response.status_code} for {url}"
            except requests.exceptions.ConnectionError:
                return f"Error: Jina connection failed for {url}"
            except Exception as e:
                return f"Error: Jina unexpected error for {url}: {str(e)}"
            
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            # text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
            text = re.sub(pattern, "", response_text).replace('---', '-').replace('===', '=').replace('   ',' ').replace('   ',' ')
        
        
        else:
            try:
                response = session.get(url, timeout=50, proxies=proxies)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return f"Error: Request timed out for {url}"
            except requests.exceptions.HTTPError as e:
                return f"Error: HTTP {e.response.status_code} for {url}"
            except requests.exceptions.ConnectionError:
                return f"Error: Connection error for {url}"
            except Exception as e:
                return f"Error: Unexpected error for {url}: {str(e)}"
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        print(f"google_search.py里面的 extract_text_from_url()里面的 url:{url}fetch结果：{len(text)}")
        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                print(f"google_search.py里面的 extract_text_from_url()里面的 fetch全文，抽取后的结果：{context}")
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# 使用线程池（最大并发数 max_workers）并发地从多个 URL 获取内容。
# 支持为每个 URL 单独提供对应的摘要（snippets 字典）。返回字典，键为 URL，值为提取到的文本或错误信息，并用 tqdm 显示进度条，同时每次请求后休眠 0.2 秒做简单限流。
def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    failed_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                # results[url] = data
                # ========== 新增：检查是否抓取失败 ==========
                if data and not data.startswith("Error"):
                    results[url] = data
                else:
                    failed_urls.append(url)
                    print(f"[Warning] Failed to fetch {url}: {data}")
                # ============================================
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


# 调用 Bing Web Search API 执行网络搜索。发送 GET 请求并设置超时时间（默认 20 秒）。返回 API 响应的 JSON 字典；若超时或请求出错，打印错误并返回空字典。
# 未指定 count 时，默认返回 10 个搜索结果。
# def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
#     """
#     Perform a search using the Bing Web Search API with a set timeout.

#     Args:
#         query (str): Search query.
#         subscription_key (str): Subscription key for the Bing Search API.
#         endpoint (str): Endpoint for the Bing Search API.
#         market (str): Market, e.g., "en-US" or "zh-CN".
#         language (str): Language of the results, e.g., "en".
#         timeout (int or float or tuple): Request timeout in seconds.
#                                          Can be a float representing the total timeout,
#                                          or a tuple (connect timeout, read timeout).

#     Returns:
#         dict: JSON response of the search results. Returns None or raises an exception if the request times out.
#     """
#     headers = {
#         "Ocp-Apim-Subscription-Key": subscription_key
#     }
#     params = {
#         "q": query,
#         "mkt": market,
#         "setLang": language,
#         "textDecorations": True,
#         "textFormat": "HTML"
#     }

#     try:
#         response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
#         response.raise_for_status()  # Raise exception if the request failed
#         search_results = response.json()
#         return search_results
#     except Timeout:
#         print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query}")
#         return {}  # Or you can choose to raise an exception
#     except requests.exceptions.RequestException as e:
#         print(f"Error occurred during Bing Web Search request: {e}")
#         return {}

def google_web_search(query, subscription_key, endpoint, market="en-US",  language="en", timeout=200,
    gl=None,                 # 显式覆盖国家
    hl=None,                # 显式覆盖语言
    num=None,                # 显式覆盖结果数量
    ):
    """
    Google Serper 版本的 bing_web_search，返回与原 Bing 版本相同结构的字典，
    方便 Search-o1 项目无缝切换。

    Args:
        query (str): 搜索关键词.
        subscription_key (str): Serper API Key（对应原 Bing 的 subscription_key）.
        endpoint (str): Serper 端点，例如 "https://google.serper.dev/search"  .
        gl (str): 显式指定 Serper 的国家码（优先于 market）.  国家
        hl (str): 显式指定 Serper 的语言码（优先于 language）. 语言
        timeout (int or float): 请求超时（秒）.
        num (int): 显式指定返回结果数量（优先于 topk 默认值）.

    Returns:
        dict: 结构与 Bing Web Search API 响应兼容，包含 "webPages" 等字段。
    """
    import requests
    from requests.exceptions import Timeout as RequestsTimeout

    # 1. 从 market / language 推断默认的 gl / hl
    #    market 形如 "en-US"，取 "-" 前部分作为国家，后部分作为语言
    if gl is None:
        gl = market.split("-")[-1].lower() if "-" in market else "us"
    if hl is None:
        hl = language.lower() if language else "en"

    # 2. 构造 Serper 请求参数（参考 google.py 里的 _call_serper_api）
    params = {
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": num or 10,  # 默认 10 条，可被外部 num 参数覆盖
        # 如需支持更多 Serper 参数，可在这里扩展
    }

    headers = {
        "X-API-KEY": subscription_key or "",
        "Content-Type": "application/json",
    }

    # 3. 发起请求
    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        serper_data = response.json()
        print(f"google_search.py里面的 google_web_search() 调用API结果 serper_data：{json.dumps(serper_data, indent=4, ensure_ascii=False)}")

    except RequestsTimeout:
        print(f"Google Serper request timed out ({timeout} seconds) for query: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Google Serper request: {e}")
        return {}

    # 4. 把 Serper 结果映射成 Bing 风格的响应结构
    #    主要是构造一个 "webPages" -> "value" 列表，每项包含：
    #    id, name(title), url, siteName, datePublished, snippet
    bing_style_response = {"webPages": {"value": []}}

    # 4.1 先处理 answerBox / knowledgeGraph，作为“高优结果”插入列表最前面
    #     参考 google.py 的 _parse_response，把 answerBox / knowledgeGraph 视为 position=0
    answer_box = serper_data.get("answerBox")
    knowledge_graph = serper_data.get("knowledgeGraph")

    if answer_box:
        # 优先使用 answer，其次 snippet / snippetHighlighted
        answer_text = (
            answer_box.get("answer")
            or answer_box.get("snippet")
            or answer_box.get("snippetHighlighted", "")
        )
        if answer_text:
            # 把答案框当作一条特殊网页结果，url 为空，siteName 标注来源
            bing_style_response["webPages"]["value"].append(
                {
                    "id": 1,  # 后面会重新编号
                    "name": "",  # Bing 里 name 对应标题，这里留空表示答案框
                    "url": answer_box.get("link", ""),
                    "siteName": "Answer Box",
                    "datePublished": answer_box.get("date", ""),
                    "snippet": str(answer_text).replace("\n", " "),
                }
            )

    if knowledge_graph:
        # 知识图谱：拼接 description + attributes 作为 snippet
        description = knowledge_graph.get("description", "")
        attributes = ". ".join(
            f"{k}: {v}" for k, v in knowledge_graph.get("attributes", {}).items()
        )
        snippet = f"{description}. {attributes}" if attributes else description

        # 标题使用 title + type
        title = f"{knowledge_graph.get('title', '')}: {knowledge_graph.get('type', '')}"

        bing_style_response["webPages"]["value"].append(
            {
                "id": 2,  # 后面会重新编号
                "name": title,
                "url": knowledge_graph.get("descriptionLink", ""),
                "siteName": "Knowledge Graph",
                "datePublished": knowledge_graph.get("date", ""),
                "snippet": snippet.replace("\n", " "),
            }
        )

    # 4.2 处理 organic 列表（普通网页结果）
    organic_results = serper_data.get("organic", [])
    for idx, result in enumerate(organic_results):
        # Serper organic 结构示例：
        # {
        #   "title": ...,
        #   "link": ...,
        #   "snippet": ...,
        #   "date": "Feb 1, 2026",   # 可选
        #   "position": 1,
        #   ...
        # }
        title = result.get("title", "")
        url = result.get("link", "")
        snippet = result.get("snippet", "")
        date_str = result.get("date", "")  # Serper 可能有 date 字段

        bing_style_response["webPages"]["value"].append(
            {
                "id": idx + 1,  # 后面会重新统一编号
                "name": title,
                "url": url,
                "siteName": "",  # Serper 不提供明确的 siteName，可按需从 URL 解析
                "datePublished": date_str,
                "snippet": snippet,
            }
        )

    # 4.3 重新统一编号（按原始顺序：answerBox / knowledgeGraph / organic）
    for new_id, item in enumerate(bing_style_response["webPages"]["value"], start=1):
        item["id"] = new_id

    # 5. 如果 Serper 返回了 peopleAlsoAsk / relatedSearches，也可以按需塞进响应
    #    这里先不处理，因为原 Bing 版本主要用 webPages，后续再扩展即可
    print(f"google_search.py里面的 最后的bing_style_response：{bing_style_response}")
    return bing_style_response


# 从指定 URL 下载 PDF 文件，使用 pdfplumber 提取所有页面的文本，将每页文本拼接后，用空格连接并限制最多 600 个词（split()[:600]）。返回清理后的文本字符串，若出错则返回错误信息。
def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=200, proxies=proxies)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


# 从 Bing 搜索返回的 JSON 结果中提取有用的信息。遍历 search_results['webPages']['value']，
# 提取每个结果的标题、URL、网站名、发布日期（仅日期部分）、摘要（snippet），并添加一个空的 context 字段供后续填充。
# 返回包含这些信息的列表。
def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    # print(f"google_search.py里面的 extract_relevant_info() 传进来的search_results：{json.dumps(search_results, indent=4, ensure_ascii=False)}")
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    print(f"google_search.py里面的 extract_relevant_info：{extract_relevant_info}")
    
    return useful_info


# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"
    
    # Subscription key and endpoint for Bing Search API
    GOOGLE_SUBSCRIPTION_KEY = "YOUR_GOOGLE_SUBSCRIPTION_KEY"
    if not GOOGLE_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
    google_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # Perform the search
    print("Performing Bing Web Search...")
    search_results = google_web_search(query, GOOGLE_SUBSCRIPTION_KEY, google_endpoint)
    
    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)  # 从 Bing 搜索返回的 JSON 结果中提取有用的信息。

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=True)  # Get full webpage text 从给定的 URL（网页或 PDF）中提取文本内容。
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])  # 从完整文本中定位最匹配给定摘要（snippet）的句子，并提取该句子前后各 context_chars 个字符作为上下文
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    # print("Your Search Query:", query)
    # print("Final extracted information with context:")
    # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
