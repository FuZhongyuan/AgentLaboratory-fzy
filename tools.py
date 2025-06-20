from utils import *

import os
import time
import arxiv
import io, sys
import traceback
import matplotlib
import numpy as np
import multiprocessing
# 设置多进程启动方法为'spawn'以解决CUDA初始化问题
if sys.platform != 'win32':  # Windows已默认使用'spawn'
    multiprocessing.set_start_method('spawn', force=True)
from pypdf import PdfReader
from datasets import load_dataset
from psutil._common import bytes2human
from datasets import load_dataset_builder
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import requests



class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr:
        :param dwn_thr:
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet the thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                # Check if the description is a non-empty string
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, weighted by likes and downloads.
        :param query: The search query string.
        :param N: The number of results to return.
        :param sim_w: Weight for cosine similarity.
        :param like_w: Weight for likes.
        :param dwn_w: Weight for downloads.
        :return: List of top N dataset items.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)
        # Compute final scores
        final_scores = (
                sim_w * cosine_similarities_norm +
                like_w * self.likes_norm +
                dwn_w * self.downloads_norm
        )
        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        # Convert indices to Python ints
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]
        # check if dataset has a test & train set
        has_test_set = list()
        has_train_set = list()
        ds_size_info = list()
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception as e:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue
            # Print number of examples for
            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)
            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples
            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))
        for _i in range(len(top_datasets)):
            top_datasets[_i]["has_test_set"] = has_test_set[_i]
            top_datasets[_i]["has_train_set"] = has_train_set[_i]
            top_datasets[_i]["test_download_size"] = ds_size_info[_i][0]
            top_datasets[_i]["test_element_size"] = ds_size_info[_i][1]
            top_datasets[_i]["train_download_size"] = ds_size_info[_i][2]
            top_datasets[_i]["train_element_size"] = ds_size_info[_i][3]
        return top_datasets

    def results_str(self, results):
        """
        Provide results as list of results in human-readable format.
        :param results: (list(dict)) list of results from search
        :return: (list(str)) list of results in human-readable format
        """
        result_strs = list()
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        paper_sums = list()
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f'Title: {results[_i].title}\n'
            paper_sum += f'Abstract: {results[_i].abstract}\n'
            paper_sum += f'Citations: {results[_i].citationCount}\n'
            paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
            paper_sum += f'Venue: {results[_i].venue}\n'
            paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        
    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH while preserving as much information as possible"""
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0
        
        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
            
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results` is a generator; you can iterate over its elements one by one...
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    #paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue
        return None

    def retrieve_full_paper_text(self, query, MAX_LEN=50000):
        pdf_text = str()
        max_retries = 5
        retry_count = 0
        backoff_factor = 2  # 指数退避因子
        
        while retry_count < max_retries:
            try:
                # 尝试获取论文信息
                paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
                
                # 尝试下载 PDF
                try:
                    paper.download_pdf(filename="downloaded-paper.pdf")
                except Exception as e:
                    # 如果 PDF 下载失败，增加重试次数并等待
                    retry_count += 1
                    wait_time = backoff_factor * retry_count
                    print(f"PDF 下载失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                # 尝试读取 PDF
                try:
                    reader = PdfReader('downloaded-paper.pdf')
                except Exception as e:
                    # 如果 PDF 读取失败，删除文件，增加重试次数并等待
                    if os.path.exists("downloaded-paper.pdf"):
                        os.remove("downloaded-paper.pdf")
                    retry_count += 1
                    wait_time = backoff_factor * retry_count
                    print(f"PDF 读取失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                # 遍历所有页面
                for page_number, page in enumerate(reader.pages, start=1):
                    # 提取页面文本
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        if os.path.exists("downloaded-paper.pdf"):
                            os.remove("downloaded-paper.pdf")
                        time.sleep(2.0)
                        return f"提取文本失败: {str(e)}"

                    # 处理文本
                    pdf_text += f"--- Page {page_number} ---"
                    pdf_text += text
                    pdf_text += "\n"
                
                # 成功处理，删除文件并返回结果
                if os.path.exists("downloaded-paper.pdf"):
                    os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return pdf_text[:MAX_LEN]
                
            except requests.exceptions.SSLError as e:
                # 特别处理 SSL 错误
                retry_count += 1
                wait_time = backoff_factor * retry_count
                print(f"SSL 错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                # 确保没有残留的 PDF 文件
                if os.path.exists("downloaded-paper.pdf"):
                    os.remove("downloaded-paper.pdf")
            
            except Exception as e:
                # 处理其他所有异常
                retry_count += 1
                wait_time = backoff_factor * retry_count
                print(f"发生错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                # 确保没有残留的 PDF 文件
                if os.path.exists("downloaded-paper.pdf"):
                    os.remove("downloaded-paper.pdf")
        
        # 所有重试都失败
        return f"无法获取论文内容: 已达到最大重试次数 ({max_retries})"


# Set the non-interactive backend early in the module
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def worker_run_code(code_str, output_queue, user_id=None, output_dir=None):
    """
    在隔离的进程中执行代码字符串
    @param code_str: 要执行的代码字符串
    @param output_queue: 用于返回输出的队列
    @param user_id: 用户ID，用于隔离文件生成
    @param output_dir: 输出目录路径，如果不指定则使用基于user_id的默认路径
    """
    output_capture = io.StringIO()
    sys.stdout = output_capture
    
    # 设置输出目录
    if user_id and not output_dir:
        output_dir = f"user_data/{user_id}"
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建一个globals字典，设置__name__为"__main__"
        globals_dict = {"__name__": "__main__"}
        
        # 如果存在输出目录，添加matplotlib配置来重定向图像保存位置
        if output_dir:
            # 在代码开头添加matplotlib配置，重定向savefig的默认路径
            matplotlib_config = f"""
import matplotlib.pyplot as plt
import os

# 设置matplotlib保存图像的默认路径
_original_savefig = plt.savefig

def _custom_savefig(fname, *args, **kwargs):
    # 如果是相对路径且没有指定目录，则保存到指定的用户目录
    if not os.path.isabs(fname) and '/' not in fname and '\\\\' not in fname:
        new_path = os.path.join("{output_dir}", fname)
        return _original_savefig(new_path, *args, **kwargs)
    return _original_savefig(fname, *args, **kwargs)

# 替换原始的savefig函数
plt.savefig = _custom_savefig
"""
            code_str = matplotlib_config + code_str
        
        # 检查代码中是否包含CUDA相关操作
        if "torch.cuda" in code_str or ".cuda()" in code_str or ".to('cuda')" in code_str or ".to(\"cuda\")" in code_str:
            # 添加CUDA设备检查和错误处理
            cuda_setup = """
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 设置默认CUDA设备
else:
    print("CUDA不可用，使用CPU进行计算")
"""
            code_str = cuda_setup + code_str
        
        # 如果有输出目录，则修改代码中的文件路径引用
        if output_dir:
            # 在代码中替换直接引用的图像文件名，防止直接使用根目录
            code_str = code_str.replace('plt.savefig("Figure_1.png")', f'plt.savefig("{output_dir}/Figure_1.png")')
            code_str = code_str.replace('plt.savefig("Figure_2.png")', f'plt.savefig("{output_dir}/Figure_2.png")')
            code_str = code_str.replace('print("\\nDone. Figures: Figure_1.png, Figure_2.png")', 
                                        f'print("\\nDone. Figures: {output_dir}/Figure_1.png, {output_dir}/Figure_2.png")')
        
        exec(code_str, globals_dict)
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
        traceback.print_exc(file=output_capture)
    finally:
        sys.stdout = sys.__stdout__
    
    # 返回输出和文件路径信息
    result = {
        "output": output_capture.getvalue(),
        "output_dir": output_dir if output_dir else ""
    }
    output_queue.put(result)

def execute_code(code_str, timeout=600, MAX_LEN=1000, user_id=None, lab_dir=None):
    """
    执行代码字符串并返回结果
    @param code_str: 要执行的代码字符串
    @param timeout: 执行超时时间（秒）
    @param MAX_LEN: 最大输出长度
    @param user_id: 用户ID，用于隔离文件生成
    @param lab_dir: 实验目录
    @return: 执行结果字符串
    """
    code_str = "from utils import *\n" + code_str
    
    # 检查不允许的操作
    if "load_dataset('pubmed" in code_str:
        return "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."
    
    # 设置输出目录，优先使用lab_dir，其次使用user_id
    output_dir = None
    if lab_dir:
        output_dir = lab_dir
    elif user_id:
        # 为每个用户创建独立的数据目录
        output_dir = f"user_data/{user_id}"
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否包含PyTorch多进程数据加载器
    if "DataLoader" in code_str and "num_workers" in code_str:
        # 添加多进程设置代码
        mp_setup = """
# 确保在创建DataLoader之前设置多进程启动方法
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
"""
        code_str = mp_setup + code_str
    
    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker_run_code, args=(code_str, output_queue, user_id, output_dir))
    proc.start()
    proc.join(timeout)
    
    if proc.is_alive():
        proc.terminate()  # 强制终止进程
        proc.join()
        return (f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. "
                "You must reduce the time complexity of your code.")
    else:
        if not output_queue.empty():
            result = output_queue.get()
            output = result["output"]
            # 记录输出目录，以便后续引用
            if result["output_dir"]:
                # 添加环境变量或全局状态来跟踪当前用户的输出目录
                os.environ["CURRENT_OUTPUT_DIR"] = result["output_dir"]
        else:
            output = ""
        return output
