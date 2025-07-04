import os, re
import shutil
import time
import tiktoken, openai
import subprocess, string
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient


def query_deepseekv3(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False, temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to query deepseekv3 failed: {e}"
        return query_deepseekv3(prompt, system, attempt+1)


def query_qwen(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = InferenceClient(api_key=api_key)
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]

        completion = client.chat.completions.create(
            model="Qwen/QwQ-32B",
            messages=messages,
            max_tokens=500,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_qwen(prompt, system, attempt+1)


def query_gpt4omini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query 4o-mini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4omini(prompt, system, attempt+1)



def query_gpt4o(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "user", "content":system + prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query gpr-4o error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4o(prompt, system, attempt+1)



def query_gemini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini(prompt, system, attempt+1)



def query_gemini2p0(prompt, system, api_key, attempt=0, temperature=0.0,):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini2p0(prompt, system, attempt+1)


def compile_latex(latex_code, output_path, compile=True, timeout=30):
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")
    #print(latex_code)
    dir_path = f"{output_path}/tex"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w") as f:
        f.write(latex_code)

    if not compile:
        return f"Compilation successful"

    # Compiling the LaTeX code using pdflatex with non-interactive mode and timeout
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,                   # Raises a CalledProcessError on non-zero exit codes
            stdout=subprocess.PIPE,        # Capture standard output
            stderr=subprocess.PIPE,        # Capture standard error
            timeout=timeout,               # Timeout for the process
            cwd=dir_path
        )

        # If compilation is successful, return the success message
        return f"Compilation successful: {result.stdout.decode('utf-8')}"

    except subprocess.TimeoutExpired:
        # If the compilation takes too long, return a timeout message
        return "[CODE EXECUTION ERROR]: Compilation timed out after {} seconds".format(timeout)
    except subprocess.CalledProcessError as e:
        # If there is an error during LaTeX compilation, return the error message
        return f"[CODE EXECUTION ERROR]: Compilation failed. There was an error in your latex."


def compile_latex_chinese(latex_code, output_path, compile: bool = True, timeout: int = 60):
    """
    专为中文（ctexart 文档类）报告设计的 LaTeX 编译辅助函数。

    参考 compile_latex，但使用 xelatex/latexmk -xelatex 以获得对 UTF-8 中文字体的良好支持。
    参数说明
    ----------
    latex_code : str
        完整的 LaTeX 源码字符串（一般来自 report_中文.txt）。
    output_path : str
        用于保存 tex 及 pdf 的目录，函数会在其中创建 tex/ 子目录。
    compile : bool, default True
        是否实际调用 LaTeX 引擎进行编译。若仅想测试生成 tex 文件，可设置为 False。
    timeout : int, default 60
        编译超时时间（秒）。
    返回
    ------
    str
        代表编译结果的字符串信息。
    """
    # 1. 强制使用 ctexart 文档类，并追加常用宏包（与英文版本保持一致，额外保证中文兼容）。
    if r"\documentclass[UTF8]{ctexart}" in latex_code:
        latex_code = latex_code.replace(
            r"\documentclass[UTF8]{ctexart}",
            "\\documentclass[UTF8]{ctexart}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")
    else:
        # 假如用户错误地使用了 article，我们自动替换为支持中文的 ctexart
        latex_code = latex_code.replace(
            r"\documentclass{article}",
            "\\documentclass[UTF8]{ctexart}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")

    # 2. 创建 tex 目录并写入 temp.tex
    dir_path = os.path.join(output_path, "tex")
    os.makedirs(dir_path, exist_ok=True)
    tex_file_path = os.path.join(dir_path, "temp-zh.tex")
    with open(tex_file_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    if not compile:
        return "TeX 文件已生成，跳过编译（compile=False）"

    # 3. 优先尝试 latexmk -xelatex（效果更好），若不可用则回退至 xelatex。
    compile_cmds = [
        ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "temp-zh.tex"],
        ["xelatex", "-interaction=nonstopmode", "temp-zh.tex"]
    ]

    last_error = None
    for cmd in compile_cmds:
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                cwd=dir_path
            )
            return f"Compilation successful (command: {' '.join(cmd)}): {result.stdout.decode('utf-8')}"
        except FileNotFoundError as e:
            # 对于 latexmk/xelatex 未安装的情况，继续尝试下一条指令
            last_error = e
            continue
        except subprocess.TimeoutExpired:
            return f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds"
        except subprocess.CalledProcessError as e:
            return f"[CODE EXECUTION ERROR]: Compilation failed with exit code {e.returncode}. Stderr: {e.stderr.decode('utf-8', errors='ignore')}"

    return f"[CODE EXECUTION ERROR]: 编译器 (latexmk/xelatex) 未找到或执行失败: {last_error}"


def count_tokens(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(enc.encode(message["content"])) for message in messages])
    return num_tokens

def remove_figures(user_dir=None):
    """
    删除图像文件
    @param user_dir: 用户目录，如果提供则只删除该目录下的图像
    """
    if user_dir:
        # 只删除指定用户目录下的图像
        for root, dirs, files in os.walk(user_dir):
            for _file in files:
                if "Figure_" in _file and ".png" in _file:
                    os.remove(os.path.join(root, _file))
    else:
        # 兼容旧代码，删除当前目录下的图像
        for _file in os.listdir("."):
            if "Figure_" in _file and ".png" in _file:
                os.remove(_file)

def get_user_dir(user_id=None, lab_dir=None):
    """
    获取用户工作目录
    @param user_id: 用户ID
    @param lab_dir: 实验目录
    @return: 工作目录路径
    """
    if lab_dir:
        return lab_dir
        
    if user_id:
        user_dir = f"user_data/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
        
    if "CURRENT_OUTPUT_DIR" in os.environ:
        return os.environ["CURRENT_OUTPUT_DIR"]
        
    return "."

def save_to_file(location, filename, data):
    """
    保存数据到文件
    @param location: 目录路径
    @param filename: 文件名
    @param data: 要保存的数据
    """
    # 处理路径中的相对路径前缀（如 ./）
    if location.startswith('./'):
        location = location[2:]
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 判断location是否是相对于当前工作目录的绝对路径
    # 如果location已经是当前工作目录的一部分，则不需要再拼接
    if os.path.isabs(location) or os.path.exists(location):
        # 目录已存在或是绝对路径，直接使用
        target_dir = location
    else:
        # 构建相对于当前工作目录的路径
        target_dir = location
    
    # 确保目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 构建完整的文件路径
    filepath = os.path.join(target_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存文件 {filename} 时出错: {e}")

def cleanup_user_data(user_id, older_than_days=30):
    """
    清理用户旧数据
    @param user_id: 用户ID
    @param older_than_days: 超过多少天的数据被认为是旧数据
    @return: 清理的文件数量
    """
    user_dir = f"user_data/{user_id}"
    if not os.path.exists(user_dir):
        return 0
        
    now = time.time()
    cutoff = now - (older_than_days * 86400)
    
    count = 0
    for root, dirs, files in os.walk(user_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = os.path.getmtime(file_path)
            if file_time < cutoff:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
                    
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    count += 1
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")
    
    return count

def remove_directory(dir_path):
    """Remove a directory if it exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")


def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum([len(enc.encode(message["content"])) for message in messages])

    if total_tokens <= max_tokens:
        return messages  # No need to clip if under the limit

    # Start removing tokens from the beginning
    tokenized_messages = []
    for message in messages:
        tokenized_content = enc.encode(message["content"])
        tokenized_messages.append({"role": message["role"], "content": tokenized_content})

    # Flatten all tokens
    all_tokens = [token for message in tokenized_messages for token in message["content"]]

    # Remove tokens from the beginning
    clipped_tokens = all_tokens[total_tokens - max_tokens:]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for message in tokenized_messages:
        message_token_count = len(message["content"])
        if current_idx + message_token_count > len(clipped_tokens):
            clipped_message_content = clipped_tokens[current_idx:]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            break
        else:
            clipped_message_content = clipped_tokens[current_idx:current_idx + message_token_count]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            current_idx += message_token_count
    return clipped_messages



def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code

from typing import Dict, List

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def clean_answer(s):
    s = s.replace("\\dfrac", "\\frac") # makes no difference but can lead to errors
    s = s.replace("x \\in", "")
    return s

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return clean_answer(s[len(left) : -1])


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    if "(x - 3)(x + 3)" in string:
        string = string.replace("(x - 3)(x + 3)", "(x+3)(x-3)")

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
