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
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
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
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
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

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
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
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


# 调用 Bing Web Search API 执行网络搜索。发送 GET 请求并设置超时时间（默认 20 秒）。返回 API 响应的 JSON 字典；若超时或请求出错，打印错误并返回空字典。
def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns None or raises an exception if the request times out.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()  # Raise exception if the request failed
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query}")
        return {}  # Or you can choose to raise an exception
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Bing Web Search request: {e}")
        return {}


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
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
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
    
    return useful_info


# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"
    
    # Subscription key and endpoint for Bing Search API
    BING_SUBSCRIPTION_KEY = "YOUR_BING_SUBSCRIPTION_KEY"
    if not BING_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
    bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # Perform the search
    print("Performing Bing Web Search...")
    search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    
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
