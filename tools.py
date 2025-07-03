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
        original_query = query  # 保存原始查询ID
        
        # 尝试原始版本
        result = self._try_retrieve_paper(query, MAX_LEN, max_retries)
        if result != None:
            return result
            
        # 如果原始版本失败，尝试不同版本
        print(f"尝试查询论文 {query} 的其他版本...")
        
        # 检查是否包含版本号
        if 'v' in query:
            base_id = query.split('v')[0]  # 获取基础ID（不含版本号）
            current_version = int(query.split('v')[1]) if query.split('v')[1].isdigit() else 1
            
            # 尝试其他版本
            for version in range(1, 6):  # 尝试v1到v5
                if version == current_version:
                    continue  # 跳过已尝试的版本
                    
                alternate_query = f"{base_id}v{version}"
                print(f"尝试查询论文的替代版本: {alternate_query}")
                
                result = self._try_retrieve_paper(alternate_query, MAX_LEN, 2)  # 对替代版本使用较少的重试次数
                if result != None:
                    return f"原始版本 {original_query} 无法获取，已成功获取替代版本 {alternate_query}:\n\n{result}"
        else:
            # 如果没有版本号，尝试添加v1
            alternate_query = f"{query}v1"
            print(f"尝试查询论文的替代版本: {alternate_query}")
            
            result = self._try_retrieve_paper(alternate_query, MAX_LEN, 2)
            if result != None:
                return f"原始版本 {original_query} 无法获取，已成功获取替代版本 {alternate_query}:\n\n{result}"
        
        # 所有版本都失败
        return f"无法获取论文内容: 已尝试原始ID {original_query} 及其他版本，均未成功"
    
    def _try_retrieve_paper(self, query, MAX_LEN=50000, max_retries=5):
        """尝试获取指定ID的论文，如果成功则返回文本，否则返回None"""
        retry_count = 0
        backoff_factor = 2  # 指数退避因子
        
        while retry_count < max_retries:
            try:
                # 尝试获取论文信息
                paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
                
                # -----------------------------
                # 新增逻辑：将下载的 PDF 保存到用户专属 uploads 目录
                # -----------------------------
                # 获取用户工作目录（若在 execute_code 中会自动注入 CURRENT_OUTPUT_DIR）
                user_upload_dir = os.path.join(get_user_dir(), 'uploads')
                os.makedirs(user_upload_dir, exist_ok=True)

                # 生成目标 PDF 路径，使用查询 ID 作为文件名，避免重复
                pdf_path = os.path.join(user_upload_dir, f"{query}.pdf")

                # 若已存在同名文件，先删除，确保下载的是最新版本
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.remove(pdf_path)

                # 尝试下载 PDF 至指定路径
                try:
                    paper.download_pdf(filename=pdf_path)
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
                    reader = PdfReader(pdf_path)
                except Exception as e:
                    # 如果 PDF 读取失败，删除文件，增加重试次数并等待
                    if 'pdf_path' in locals() and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    retry_count += 1
                    wait_time = backoff_factor * retry_count
                    print(f"PDF 读取失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                pdf_text = str()
                # 遍历所有页面
                for page_number, page in enumerate(reader.pages, start=1):
                    # 提取页面文本
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        if 'pdf_path' in locals() and os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        time.sleep(2.0)
                        return f"提取文本失败: {str(e)}"

                    # 处理文本
                    pdf_text += f"--- Page {page_number} ---"
                    pdf_text += text
                    pdf_text += "\n"
                
                # 成功处理后，不再删除 pdf_path，保留供用户查看
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
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.remove(pdf_path)
            
            except Exception as e:
                # 处理其他所有异常
                retry_count += 1
                wait_time = backoff_factor * retry_count
                print(f"发生错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                # 确保没有残留的 PDF 文件
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.remove(pdf_path)
        
        # 所有重试都失败
        return None


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
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    # 设置输出目录
    if user_id and not output_dir:
        output_dir = f"user_data/{user_id}"
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件路径
    log_file_path = None
    log_file = None
    if output_dir:
        log_file_path = os.path.join(output_dir, "execution_log.txt")
        log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # 在关键点添加三重输出（控制台、StringIO捕获、日志文件）
    def dual_print(message):
        output_capture.write(message + "\n")
        original_stdout.write(message + "\n")
        original_stdout.flush()  # 确保立即显示
        # 如果日志文件存在，也写入日志
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()  # 确保立即写入文件
    
    # 记录原始工作目录
    original_cwd = os.getcwd()
    
    try:
        # 如果指定了输出目录，切换当前工作目录
        if output_dir:
            os.chdir(output_dir)
            dual_print(f"已切换工作目录到: {os.path.abspath(output_dir)}")
            
            # 创建.cache目录用于存储数据集和模型
            os.makedirs('.cache', exist_ok=True)
            
            # 设置常见机器学习库的缓存目录
            # 1. 设置scikit-learn的缓存目录
            os.environ['SKLEARN_HOME'] = os.path.join(os.getcwd(), '.cache/scikit_learn')
            # 2. 设置TensorFlow的缓存目录
            os.environ['KERAS_HOME'] = os.path.join(os.getcwd(), '.cache/keras')
            # 3. 设置NLTK的数据目录
            os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), '.cache/nltk_data')
            # 4. 重定向gensim缓存
            os.environ['GENSIM_DATA_DIR'] = os.path.join(os.getcwd(), '.cache/gensim')
            # 5. 重定向SpaCy缓存
            os.environ['SPACY_DATA_PATH'] = os.path.join(os.getcwd(), '.cache/spacy')
            
            dual_print(f"已设置机器学习库缓存目录在: {os.path.join(os.getcwd(), '.cache')}")
        
        # 创建一个globals字典，设置__name__为"__main__"
        globals_dict = {"__name__": "__main__", "OUTPUT_DIR": output_dir}
        
        # 如果存在输出目录，添加matplotlib配置来重定向图像保存位置
        if output_dir:
            # 在代码开头添加matplotlib配置，重定向savefig的默认路径
            matplotlib_config = f"""
import matplotlib.pyplot as plt
import os

# 设置当前工作目录的环境变量
os.environ['CURRENT_OUTPUT_DIR'] = "{output_dir}"

# 设置matplotlib保存图像的默认路径
_original_savefig = plt.savefig

def _custom_savefig(fname, *args, **kwargs):
    # 如果是相对路径且没有指定目录，则保存到指定的用户目录
    if not os.path.isabs(fname) and '/' not in fname and '\\\\' not in fname:
        return _original_savefig(fname, *args, **kwargs)
    return _original_savefig(fname, *args, **kwargs)

# 替换原始的savefig函数
plt.savefig = _custom_savefig

# 重定向其他常见的文件操作
_original_open = open

def _custom_open(file, *args, **kwargs):
    # 如果是相对路径且不是以./或../开头，则直接使用
    return _original_open(file, *args, **kwargs)

# 替换内置的open函数
open = _custom_open

# 重定向数据集下载目录
import tempfile
import shutil
from pathlib import Path

# 保存原始的数据集缓存目录
os.environ['ORIGINAL_HF_HOME'] = os.environ.get('HF_HOME', '')
os.environ['ORIGINAL_TORCH_HOME'] = os.environ.get('TORCH_HOME', '')
os.environ['ORIGINAL_XDG_CACHE_HOME'] = os.environ.get('XDG_CACHE_HOME', '')

# 设置本地数据缓存目录
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache/huggingface')
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(os.getcwd(), '.cache')

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
        
        # 检查代码中是否包含BatchNorm相关操作，并添加批处理大小检查
        if "BatchNorm" in code_str or "batch_norm" in code_str:
            batch_norm_fix = """
# 添加批处理大小检查，解决BatchNorm在批处理大小为1时的问题
def ensure_batchnorm_works(model, batch_size=None):
    '''检查批处理大小，对于小批量在推理时调整BatchNorm层'''
    if batch_size is None:
        # 尝试从代码中判断批处理大小
        import re
        try:
            batch_size_pattern = re.compile(r'batch_size\\s*=\\s*(\\d+)')
            matches = batch_size_pattern.findall(open('generated_code.py').read())
            batch_size = int(matches[0]) if matches else None
        except:
            batch_size = None
    
    if batch_size == 1:
        print("警告: 检测到批处理大小为1，自动将所有BatchNorm层设置为eval模式")
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
    return model
"""
            code_str = batch_norm_fix + code_str
            
            # 在模型定义后添加批处理大小检查
            if "model = " in code_str:
                code_str = code_str.replace("model = ", "model = ensure_batchnorm_works(", 1)
                # 确保在model定义后有一个闭合的括号
                next_line_index = code_str.find("\n", code_str.find("model = ensure_batchnorm_works("))
                if next_line_index > 0:
                    code_str = code_str[:next_line_index] + ")" + code_str[next_line_index:]
        
        # 如果有输出目录，则修改代码中的文件路径引用
        if output_dir:
            # 在代码中添加一个辅助函数，用于处理文件路径
            path_helper = """
# 辅助函数：确保文件路径在用户目录下
def ensure_user_path(path):
    # 已经在用户目录下，直接返回路径
    return path
"""
            code_str = path_helper + code_str
            
            # 替换常见的文件操作模式
            code_str = code_str.replace('plt.savefig("Figure_1.png")', 'plt.savefig("Figure_1.png")')
            code_str = code_str.replace('plt.savefig("Figure_2.png")', 'plt.savefig("Figure_2.png")')
            code_str = code_str.replace('print("\\nDone. Figures: Figure_1.png, Figure_2.png")', 
                                      'print("\\nDone. Figures: Figure_1.png, Figure_2.png")')
            
            # 替换常见的文件写入操作 - 修复括号闭合问题
            # 使用更安全的方式替换文件操作，避免破坏原有的括号结构
            # 对于 with open 操作
            code_str = code_str.replace('with open("', 'with open("')
            
            # 对于 to_csv 操作，使用正则表达式进行更精确的替换
            import re
            # 匹配 .to_csv('filename' 或 .to_csv("filename" 模式
            code_str = re.sub(r'\.to_csv\((["\'])', r'.to_csv(\1', code_str)
            
            # 对于其他常见的文件操作也采用类似的方式
            code_str = re.sub(r'\.to_excel\((["\'])', r'.to_excel(\1', code_str)
            code_str = re.sub(r'\.to_json\((["\'])', r'.to_json(\1', code_str)
            code_str = re.sub(r'\.to_pickle\((["\'])', r'.to_pickle(\1', code_str)
        
        # 将代码保存到文件中，便于用户查看和调试
        code_filename = "generated_code.py"
        # 确保使用当前工作目录中的路径
        code_filepath = os.path.join(os.getcwd(), code_filename)
        
        with open(code_filepath, 'w', encoding='utf-8') as f:
            f.write(code_str)
        dual_print(f"生成的代码已保存到: {code_filepath}")
        
        # 重定向标准输出到同时写入StringIO和日志文件
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()  # 立即刷新缓冲区
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # 如果有日志文件，设置多重输出
        if log_file:
            sys.stdout = TeeOutput(output_capture, log_file)
        
        # 执行代码 - 改为执行文件而不是字符串
        dual_print(f"正在执行文件: {code_filepath}")
        exec(f"exec(open('{code_filepath}').read())", globals_dict)
        
    except Exception as e:
        import urllib.error
        # 基于异常类型的通用网络下载错误检测，避免硬编码数据集名
        network_error_types = (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            TimeoutError,
            OSError,  # 某些网络超时也表现为 OSError: [Errno 101] Network is unreachable
        )

        err_prefix = "[CODE EXECUTION ERROR]: "
        # 如果是下载/连接相关错误，则额外附加 DATASET_DOWNLOAD_FAILED 标志
        if isinstance(e, network_error_types):
            err_prefix = "[DATASET_DOWNLOAD_FAILED] " + err_prefix
        dual_print(f"{err_prefix}{str(e)}")
    finally:
        # 恢复原始工作目录
        if output_dir:
            os.chdir(original_cwd)
            print(f"已恢复原始工作目录: {original_cwd}")
            
        # 恢复原始环境变量
        for env_var, original_value in [
            ('HF_HOME', os.environ.get('ORIGINAL_HF_HOME', '')),
            ('TORCH_HOME', os.environ.get('ORIGINAL_TORCH_HOME', '')),
            ('XDG_CACHE_HOME', os.environ.get('ORIGINAL_XDG_CACHE_HOME', '')),
            ('SKLEARN_HOME', ''),
            ('KERAS_HOME', ''),
            ('NLTK_DATA', ''),
            ('GENSIM_DATA_DIR', ''),
            ('SPACY_DATA_PATH', '')
        ]:
            if original_value:
                os.environ[env_var] = original_value
            elif env_var in os.environ:
                del os.environ[env_var]
                
        print("已恢复原始环境变量设置")
        sys.stdout = sys.__stdout__
        
        # 关闭日志文件
        if log_file:
            log_file.close()
    
    # 返回输出和文件路径信息
    result = {
        "output": output_capture.getvalue(),
        "output_dir": output_dir if output_dir else "",
        "code_file": code_filepath if 'code_filepath' in locals() else None,
        "log_file": log_file_path if log_file_path else None
    }
    output_queue.put(result)

def execute_code(code_str, timeout=2400, MAX_LEN=1000, user_id=None, lab_dir=None):
    """
    执行代码字符串并返回结果
    @param code_str: 要执行的代码字符串
    @param timeout: 执行超时时间（秒）
    @param MAX_LEN: 最大输出长度
    @param user_id: 用户ID，用于隔离文件生成
    @param lab_dir: 实验目录
    @return: 执行结果字典，包含output和code_file
    """
    # code_str = "from utils import *\n" + code_str
    
    # 检查不允许的操作
    if "load_dataset('pubmed" in code_str:
        return {
            "output": "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated",
            "code_file": None
        }
    if "exit(" in code_str:
        return {
            "output": "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this.",
            "code_file": None
        }
    
    # 设置输出目录，优先使用lab_dir，其次使用user_id
    output_dir = None
    if lab_dir:
        output_dir = lab_dir
    elif user_id:
        # 为每个用户创建独立的数据目录
        output_dir = f"user_data/{user_id}"
        os.makedirs(output_dir, exist_ok=True)
    
    # 打印明确的工作目录信息，便于排查问题
    print(f"执行代码的目标输出目录: {output_dir}")
    if output_dir:
        print(f"确保输出目录存在: {os.path.abspath(output_dir)}")
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
        return {
            "output": f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. You must reduce the time complexity of your code.",
            "code_file": None
        }
    else:
        if not output_queue.empty():
            result = output_queue.get()
            output = result["output"]
            code_file = result.get("code_file", None)
            log_file = result.get("log_file", None)
            # 记录输出目录，以便后续引用
            if result["output_dir"]:
                # 添加环境变量或全局状态来跟踪当前用户的输出目录
                os.environ["CURRENT_OUTPUT_DIR"] = result["output_dir"]
            
            # 返回输出和代码文件路径（确保使用完整路径）
            if code_file:
                if not os.path.isabs(code_file):
                    if output_dir:
                        code_file = os.path.join(os.path.abspath(output_dir), os.path.basename(code_file))
                    else:
                        code_file = os.path.abspath(code_file)
                print(f"返回代码文件路径: {code_file}")
            
            if log_file:
                if not os.path.isabs(log_file):
                    if output_dir:
                        log_file = os.path.join(os.path.abspath(output_dir), os.path.basename(log_file))
                    else:
                        log_file = os.path.abspath(log_file)
                print(f"返回日志文件路径: {log_file}")
                
            return {
                "output": output,
                "code_file": code_file,
                "log_file": log_file
            }
        else:
            return {
                "output": "",
                "code_file": None,
                "log_file": None
            }
