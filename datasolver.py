import random
from copy import copy
from copy import deepcopy
from common_imports import *
from abc import abstractmethod

from tools import *
from inference import *
from pathlib import Path

from contextlib import contextmanager
import sys, os
import logging
import warnings
import pandas as pd
import numpy as np
import re

# Configure logging and warnings
os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)

# Constants
GLOBAL_REPAIR_ATTEMPTS = 2

class Command:
    """
    Abstract base class for command pattern implementation
    """
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        """Return documentation string for the command"""
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        """Execute the command with given arguments"""
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        """Check if the command string matches this command"""
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        """Parse command string into arguments"""
        pass


"""
@@@@@@@@@@@@@@@@@@@@
@@ DATA PREP TOOLS @@
@@@@@@@@@@@@@@@@@@@@
"""

class Replace(Command):
    """
    Replace command for data preparation code
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "DATA-replace"

    def docstring(self) -> str:
        return (
            "============= DATA PREPARATION CODE REPLACEMENT TOOL =============\n"
            "You have access to a data preparation code replacing tool.\n"
            "This tool allows you to entirely re-write/replace all of the current data preparation code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes to the data preparation process, such as entirely changing the dataset loading method or preprocessing steps. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new code
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        # Get the current working directory as lab_dir
        current_dir = os.getcwd()
        # Check if running in user directory
        if "user_data" in current_dir:
            lab_dir = current_dir
        else:
            # Try to get output directory from environment variable
            lab_dir = os.environ.get("CURRENT_OUTPUT_DIR", current_dir)
            
        # 如果是在DataSolver类中调用，尝试获取lab_dir参数
        if len(args) > 2 and args[2]:
            lab_dir = args[2]
            
        # Pass lab_dir parameter to execute_code
        code_ret = execute_code(code_exec, lab_dir=lab_dir)
        if "[CODE EXECUTION ERROR]" in code_ret['output']: 
            return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)


class Edit(Command):
    """
    Edit command for data preparation code
    """
    def __init__(self):
        super().__init__()
        self.cmd_type = "DATA-edit"

    def docstring(self) -> str:
        return (
            "============= DATA PREPARATION CODE EDITING TOOL =============\n"
            "You have access to a data preparation code editing tool.\n"
            "This tool allows you to edit specific lines in the current data preparation code.\n"
            "You can use this tool via the following command: ```EDIT\nN\nM\n<code here>\n```, where N is the starting line number (0-indexed), M is the ending line number (0-indexed), and <code here> is the new code that will replace lines N through M. This tool is useful for making targeted changes to specific parts of the data preparation process without rewriting the entire code. Before applying your edits, the modified code will be tested and if it returns an error, the edits will not be applied."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> N (int)
        # args[1] -> M (int)
        # args[2] -> old code
        # args[3] -> new lines to replace
        # args[4] -> dataset code
        # args[5] -> lab_dir (optional)
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))
            for _ln in lines_to_replace:
                current_code.pop(_ln)
            for _line in lines_to_add:
                current_code.insert(args[0], _line)
            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            
            # Get current working directory as lab_dir
            current_dir = os.getcwd()
            # Check if running in user directory
            if "user_data" in current_dir:
                lab_dir = current_dir
            else:
                # Try to get output directory from environment variable
                lab_dir = os.environ.get("CURRENT_OUTPUT_DIR", current_dir)
                
            # 如果提供了lab_dir参数，使用它
            if len(args) > 5 and args[5]:
                lab_dir = args[5]
                
            # Pass lab_dir parameter to execute_code
            code_ret = execute_code(code_exec, lab_dir=lab_dir)
            
            if "[CODE EXECUTION ERROR]" in code_ret['output']: 
                return (False, None, code_ret['output'])
            return (True, current_code, code_ret['output'])
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str: 
            return True
        return False

    def parse_command(self, *args) -> tuple:
        # args[0] -> command string
        # args[1] -> old code
        # args[2] -> dataset code
        # args[3] -> lab_dir (optional)
        try:
            # Extract edit parameters
            edit_params = extract_prompt(args[0], "EDIT")
            edit_params = edit_params.strip().split('\n')
            
            # Parse line numbers
            n = int(edit_params[0])
            m = int(edit_params[1])
            
            # Get new code lines
            new_lines = edit_params[2:]
            
            # Get lab_dir if provided
            lab_dir = None
            if len(args) > 3:
                lab_dir = args[3]
                
            return True, (n, m, args[1], new_lines, args[2], lab_dir)
        except Exception as e:
            return False, str(e)


def get_data_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    """
    Evaluate the data preparation code quality and effectiveness

    Args:
        outlined_plan (str): Research plan description
        code (str): The data preparation code to evaluate
        code_return (str): Output from executing the code
        REWARD_MODEL_LLM (str): LLM model identifier for evaluation
        attempts (int): Number of evaluation attempts
        openai_api_key (str, optional): OpenAI API key

    Returns:
        tuple: (score, message, success_flag)
    """
    error_msg = str()
    
    for _attempt in range(attempts):
        try:
            # Data preparation specific reward model system prompt
            sys_prompt = (
                "You are a professor specializing in data science who evaluates data preparation code quality. "
                "You examine research plans, data preparation code, and execution output to determine how well "
                "the code follows best practices for data cleaning, preprocessing, and preparation. "
                "Score from 0 to 1 as a float based on:\n"
                "- Code cleanliness and readability\n"
                "- Proper handling of missing values\n"
                "- Effective feature engineering\n"
                "- Appropriate data validation steps\n"
                "- Adherence to the research plan requirements\n"
                "- Proper train/test splitting when needed\n"
                "- Data quality checks and statistics reporting\n\n"
                "Format your score exactly as: ```SCORE\n<score here>\n``` where <score here> is a float between 0 and 1."
            )
            
            # Get evaluation from the model
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys_prompt,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Research plan: {outlined_plan}\n\n"
                    f"Data preparation code:\n{code}\n\n"
                    f"Code execution output:\n{code_return}\n\n"
                    f"Please evaluate and score the data preparation code quality:"
                ), 
                temp=0.6
            )
            
            # Extract and return the score
            performance = extract_prompt(text=scoring, word="SCORE")
            performance = float(performance)
            return performance, f"Data preparation code score: {performance}", True
            
        except Exception as e:
            error_msg = str(e)
            continue
            
    return 0, error_msg, False


def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    """
    Repair code based on execution errors
    
    Args:
        code (str): Code to repair
        error (str): Error message
        ctype (str): Command type (replace or edit)
        REPAIR_LLM (str): LLM model identifier for repair
        openai_api_key (str, optional): OpenAI API key
        
    Returns:
        str: Repaired code
    """
    if ctype == "replace":
        repair_sys = (
            "You are a data preparation code repair tool.\n"
            "Your goal is to fix errors in data preparation code without changing the intended functionality.\n"
            "Focus on addressing the specific error and improving data quality checks.\n"
            "Your output should match the original code structure as closely as possible while fixing the issue.\n"
            "Format your response with the fixed code wrapped in ```python\n<code here>\n```\n"
        )
        
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Error message:\n{error}\n\nCode with error:\n\n{code}",
            temp=0.8
        )
        
        return extract_prompt(model_resp, "python")
        
    elif ctype == "edit":
        repair_sys = (
            "You are a data preparation code repair tool specializing in targeted fixes.\n"
            "Your goal is to identify and fix the specific error in the data preprocessing code.\n"
            "Use the code editing tool to make minimal, focused changes that resolve the issue.\n"
            "Format your response with the edit command: ```EDIT N M\n<new lines to replace old lines>\n```\n"
            "Where N is the first line to replace, M is the last line to replace, and the lines between are your fixes.\n"
            "Choose the smallest range of lines needed to fix the problem."
        )
        
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Error message:\n{error}\n\nCode with error:\n\n{code}",
            temp=0.2
        )
        
        return model_resp


class DataSolver:
    """
    Main class for data preparation workflow
    """
    def __init__(self, dataset_code, openai_api_key=None, notes=None, max_steps=10, insights=None, plan=None, llm_str=None, lab_dir=None):
        self.suppress_print = False
        self.notes = [] if notes is None else notes
        self.dataset_code = dataset_code
        # 标记数据集下载是否失败
        self.dataset_failed = False
        self.plan = "" if plan is None else plan
        self.llm_str = llm_str
        self.verbose = False
        self.max_codes = 2  # Keep more variations for data processing
        self.st_hist_len = 3  # Longer history for data preparation context
        self.min_gen_trials = 2  # More trials to find optimal data prep
        self.code_lines = str()
        self.st_history = list()
        self.insights = insights
        self.code_reflect = str()
        self.max_steps = max_steps
        self.prev_code_ret = str()
        self.should_execute_code = True
        self.openai_api_key = openai_api_key
        # Gather runtime environment information for prompt context
        self.env_info = self._collect_env_info()
        self.lab_dir = lab_dir
        # Data preparation specific fields
        self.data_stats = {}  # Store dataset statistics
        self.validation_metrics = {}  # Data validation metrics
        # Store latest feedback generated during command processing
        self.latest_feedback = None

    def initial_solve(self):
        """
        Initialize the solver and get an initial set of data preparation code
        """
        self.best_score = None
        self.commands = [Replace()]  # Start with only Replace to generate initial code
        self.model = f"{self.llm_str}"
        
        # Generate initial data preparation code
        init_code, init_return, self.best_score = self.gen_initial_code()
        
        # Store best codes for later potential reuse/combination
        self.best_codes = [(copy(init_code), self.best_score, init_return) for _ in range(1)]
        
        # Set current code state
        self.code_lines = init_code
        
        # Update available commands after initial code generation
        self.model = f"{self.llm_str}"
        self.commands = [Edit(), Replace()]
        self.prev_working_code = copy(self.code_lines)
        
        # Extract and store initial data statistics if available
        self._extract_data_stats(init_return)

        # ---------------- 添加初始反馈并写入历史 ----------------
        try:
            feedback_msg = self.feedback(init_return['output'] if isinstance(init_return, dict) else init_return)
        except Exception as _fb_init_e:
            feedback_msg = f"[Feedback generation failed at initialization]: {_fb_init_e}"
            print(f"@@@ INIT DATA PREP FEEDBACK GENERATION FAILED: {feedback_msg}")

        # 记录首轮历史，使用占位 cmd 名称 INITIAL_REPLACE 方便追踪
        self.st_history.append(["INITIAL_REPLACE", feedback_msg, copy(self.code_lines), "INITIAL_REPLACE"])
        # 控制历史长度
        if len(self.st_history) > self.st_hist_len:
            self.st_history.pop(0)

    @staticmethod
    def clean_text(text):
        """Clean command text for processing"""
        text = text.replace("```\n", "```")
        text = text.replace("```python\n", "```REPLACE\n")
        return text

    def gen_initial_code(self):
        """
        Generate initial data preparation code
        
        Returns:
            tuple: (code_lines, code_output, score)
        """
        num_attempts = 0
        error_hist = list()
        
        while True:
            # Error history management
            if num_attempts == 0:
                err = str()
                err_hist = str()
            else:
                err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the data preparation challenge."
                error_hist.append(err)
                if len(error_hist) > 5:  # Keep last 5 errors
                    error_hist.pop(0)
                err = "\n".join(error_hist)
                err_hist = "The following is a history of your previous errors\n" + err + "\nDO NOT REPEAT THESE."
            
            # Generate initial data preparation code
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\nYou should now use ```REPLACE to create initial data preparation code. Focus on data loading, cleaning, preprocessing, and validation. Now please enter the ```REPLACE command below:\n",
                temp=0.8  # Slightly lower temperature for more predictable data prep code
            )
            
            # Process and execute the command
            model_resp = self.clean_text(model_resp)
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)
            
            # Logging
            if not self.suppress_print:
                print(f"@@@ INIT DATA PREP ATTEMPT {num_attempts}: ", str(cmd_str).replace("\n", " | "))
                print(f"$$$ Data Prep Score: {score}")
            
            # Break if successful code generated
            if score is not None:
                break
                
            num_attempts += 1
            
            # Fail-safe to avoid infinite loops
            if num_attempts > 10:
                # Create minimal valid code if all attempts fail
                code_lines = [
                    "import pandas as pd",
                    "import numpy as np",
                    "from sklearn.model_selection import train_test_split",
                    "",
                    "# Basic data loading and preparation",
                    "def prepare_data(data_path):",
                    "    # Load the data",
                    "    df = pd.read_csv(data_path)",
                    "    print(f'Loaded data with shape: {df.shape}')",
                    "    ",
                    "    # Basic preprocessing",
                    "    # Handle missing values",
                    "    df = df.fillna(df.mean())",
                    "    ",
                    "    # Split data",
                    "    X = df.drop('target', axis=1, errors='ignore')",
                    "    y = df['target'] if 'target' in df.columns else None",
                    "    ",
                    "    # Create train/test split if target exists",
                    "    if y is not None:",
                    "        X_train, X_test, y_train, y_test = train_test_split(",
                    "            X, y, test_size=0.2, random_state=42)",
                    "        return X_train, X_test, y_train, y_test, df",
                    "    ",
                    "    return X, df",
                    "",
                    "# Run data preparation",
                    "print('Starting data preparation...')",
                    "try:",
                    "    data_prepared = prepare_data('data.csv')",
                    "    print('Data preparation complete!')",
                    "except Exception as e:",
                    "    print(f'Error in data preparation: {e}')"
                ]
                prev_code_ret = "Emergency fallback code created."
                score = 0.1  # Minimal passing score
                break
                
        return code_lines, prev_code_ret, score

    def solve(self):
        """
        Solve data preparation tasks iteratively
        
        Returns:
            tuple: (model_response, command_string)
        """
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_code_ret = None
        self.should_execute_code = False
        
        while True:
            # Different prompt based on available commands
            if len(self.commands) == 2:
                cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. "
            else:
                cmd_app_str = ""
                
            # 将上一轮即时反馈嵌入提示，帮助模型聚焦最新问题
            latest_fb_str = f"\nLATEST_FEEDBACK:\n{self.latest_feedback}\n" if self.latest_feedback else ""

            # Generate next data preparation improvement
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=(
                    f"The following is your history:{self.history_str()}\n\n"
                    f"{latest_fb_str}"
                    f"{cmd_app_str}Now please analyze the current data preparation code and results, "
                    f"identify possible improvements, and enter a command to improve the data preparation: "
                ),
                temp=0.7
            )
            
            model_resp = self.clean_text(model_resp)
            
            # Use one of the best previous solutions as starting point
            self.code_lines = copy(random.choice(self.best_codes)[0])
            
            # Process the command
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)
            
            # 直接使用在 process_command 内部即时生成的反馈
            feedback_msg = getattr(self, "latest_feedback", None)
            
            self.st_history.append([model_resp, feedback_msg, code_lines, cmd_str])
            if len(self.st_history) > self.st_hist_len:
                self.st_history.pop(0)
                
            # Check and store best solution
            if score is not None:
                if top_score is None:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(cmd_str)
                    top_score = score
                    
            # Logging
            if not self.suppress_print:
                print(f"@@@ DATA PREP COMMAND // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
                print(f"$$$ Data Prep Score: {score}")
                
            # Stop when we've done enough trials and have a good solution
            if num_attempts >= self.min_gen_trials and top_score is not None:
                break
                
            num_attempts += 1
            
            # Fail-safe to prevent infinite loops
            if num_attempts > self.max_steps * 2:
                break
                
        # 如果没有有效 best_pkg（极端情况下 score 始终为 None），使用当前 best_codes[0] 作为回退
        if best_pkg is None:
            fallback_code, fallback_score, fallback_ret = self.best_codes[0]
            best_pkg = (copy(fallback_code), copy(fallback_ret), False, "NO_VALID_CMD", "NO_VALID_CMD")

        # Set the best state as the current state
        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg
        if not self.suppress_print:
            print(self.prev_code_ret)
            
        # Update best codes collection with top scorer if it's better
        if top_score > self.best_codes[-1][1]:
            # Replace the lowest scoring one
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                # Generate reflection on best solutions
                self.code_reflect = self.reflect_code()
                
            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            # Sort by score
            self.best_codes.sort(key=lambda x: x[1], reverse=True)
            
        return model_resp, cmd_str
        
    def reflect_code(self):
        """
        Provide a reflection on produced data preparation code for next execution
        
        Returns:
            str: Language model-produced reflection on data preparation
        """
        # Combine all best code examples with their scores
        code_strs = ("$"*40 + "\n\n").join([
            self.generate_code_lines(_code[0]) + 
            f"\nData Preparation Results: {_code[2]}\nScore: {_code[1]}" 
            for _code in self.best_codes
        ])
        
        # Create reflection prompt focused on data preparation
        reflection_prompt = (
            f"Please reflect on the following sets of data preparation code: {code_strs} "
            f"and come up with generalizable insights that will help improve data preparation. "
            f"Focus on data cleaning techniques, feature engineering approaches, and validation methods "
            f"that could further enhance the data quality for the research task."
        )
        
        # Get system prompt without commands
        syst = self.system_prompt(commands=False) + reflection_prompt
        
        # Query the model for reflection
        return query_model(
            prompt=(
                "Please reflect on ways to improve your current data preparation code. "
                "Examine the provided code and suggest specific improvements for:\n"
                "1. Data cleaning and handling missing values\n"
                "2. Feature engineering and selection\n"
                "3. Data validation and quality checks\n"
                "4. Efficient data processing pipeline design\n"
                "Provide line-by-line examples where appropriate:"
            ), 
            system_prompt=syst, 
            model_str=f"{self.llm_str}", 
            openai_api_key=self.openai_api_key
        )
    
    def process_command(self, model_resp):
        """
        Take command from language model and execute if valid
        
        Args:
            model_resp (str): Language model output
            
        Returns:
            tuple: Contains the following items
                - cmd_str: Command execution result and success flag
                - code_lines: List of code lines as strings
                - prev_code_ret: Output from running code
                - should_execute_code: Flag indicating if code changed and needs re-execution
                - score: Score of model output
        """
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)
        
        # Clear any previous figures
        if 'remove_figures' in globals():
            remove_figures()
        
        # Process commands
        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                # Process DATA-edit command
                if cmd.cmd_type == "DATA-edit":
                    score = None
                    failed = True
                    code_err = str()
                    
                    # Try multiple repair attempts if needed
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        success, args = cmd.parse_command(model_resp, copy(self.code_lines), self.dataset_code, self.lab_dir)
                        if success:
                            cmd_return = cmd.execute_command(args)
                            code_err = f"Return from executing code: {cmd_return[2]}"
                            
                            if cmd_return[0]:  # If success
                                code_lines = copy(cmd_return[1])
                                # Use data-specific scoring function
                                score, cmd_str, is_valid = get_data_score(
                                    self.plan, 
                                    "\n".join(code_lines), 
                                    cmd_return[2], 
                                    openai_api_key=self.openai_api_key, 
                                    REWARD_MODEL_LLM=self.llm_str
                                )
                                
                                if is_valid:
                                    failed = False
                                    break
                                    
                                code_err += f"\nReturn from executing code on real data: {cmd_str}"
                                
                        # Try to repair code if failed —— 附加上一轮反馈信息
                        code_err_with_fb = (
                            code_err + f"\n\nLATEST_FEEDBACK:\n{self.latest_feedback}\n" if self.latest_feedback else code_err
                        )
                        repaired_code = code_repair(
                            model_resp, 
                            code_err_with_fb, 
                            REPAIR_LLM=self.llm_str, 
                            ctype="edit", 
                            openai_api_key=self.openai_api_key
                        )
                        model_resp = repaired_code
                        
                        if not self.suppress_print: 
                            print(f"     * Attempting data code repair // try {_tries+1}*")
                    
                    # Handle final result
                    if failed:
                        cmd_str = f"Data code editing FAILED due to the following error: {code_err}. Code was reverted back to original state."
                        if not self.suppress_print: 
                            print("$$$$ DATA CODE EDIT (failed)")
                    else:
                        cmd_str = "Data preparation code was successfully edited."
                        prev_code_ret = copy(cmd_return[2])
                        if not self.suppress_print: 
                            print("$$$$ DATA CODE EDIT (success)")
                        should_execute_code = True
                        
                        # Extract data statistics from successful execution
                        self._extract_data_stats(prev_code_ret)
                        
                    # ---- 生成反馈：区分成功 / 失败，保证反馈与保存的代码状态一致 ----
                    try:
                        # 选取用于反馈的代码行与运行输出
                        if failed:
                            feedback_output = code_err  # 失败时使用错误信息
                            feedback_code = self.code_lines  # 未保存，仍保持原代码
                        else:
                            feedback_output = prev_code_ret['output'] if isinstance(prev_code_ret, dict) else prev_code_ret
                            feedback_code = code_lines  # 成功时使用更新后的代码

                        _old_code_lines = copy(self.code_lines)
                        self.code_lines = copy(feedback_code)
                        self.latest_feedback = self.feedback(feedback_output)
                        self.code_lines = _old_code_lines
                    except Exception as _fb_e:
                        self.latest_feedback = f"[Feedback generation failed]: {_fb_e}"

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score
                
                # Process DATA-replace command
                elif cmd.cmd_type == "DATA-replace":
                    score = None
                    failed = True
                    code_err = str()
                    
                    # Try multiple repair attempts if needed
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        success, args = cmd.parse_command(model_resp, self.dataset_code, self.lab_dir)
                        code_err = f"Return from executing code: {args[1]}"
                        
                        if success:
                            code_lines = copy(args[0])
                            # Use data-specific scoring function
                            score, cmd_str, is_valid = get_data_score(
                                self.plan, 
                                "\n".join(code_lines), 
                                args[1], 
                                openai_api_key=self.openai_api_key, 
                                REWARD_MODEL_LLM=self.llm_str
                            )
                            
                            if is_valid:
                                failed = False
                                break
                                
                            code_err += f"\nReturn from executing code on real data: {cmd_str}"
                            
                        # Try to repair code if failed —— 附加上一轮反馈信息
                        code_err_with_fb = (
                            code_err + f"\n\nLATEST_FEEDBACK:\n{self.latest_feedback}\n" if self.latest_feedback else code_err
                        )
                        repaired_code = code_repair(
                            extract_prompt(model_resp, "REPLACE"), 
                            code_err_with_fb, 
                            ctype="replace", 
                            openai_api_key=self.openai_api_key, 
                            REPAIR_LLM=self.llm_str
                        )
                        repaired_code = f"```REPLACE\n{repaired_code}\n```"
                        model_resp = repaired_code
                        
                        if not self.suppress_print: 
                            print(f"     * Attempting data code repair // try {_tries+1}*")
                    
                    # Handle final result
                    if failed:
                        cmd_str = f"Data code replacement FAILED due to the following error: {code_err}. Code was reverted back to original state."
                        if not self.suppress_print: 
                            print("$$$$ DATA CODE REPLACE (failed)")
                    else:
                        cmd_str = "Data preparation code was successfully replaced."
                        code_lines = copy(args[0])
                        prev_code_ret = copy(args[1])
                        if not self.suppress_print: 
                            print("$$$$ DATA CODE REPLACE (success)")
                        should_execute_code = True
                        
                        # Extract data statistics from successful execution
                        self._extract_data_stats(prev_code_ret)
                        
                    # ---- 生成反馈：区分成功 / 失败，保证反馈与保存的代码状态一致 ----
                    try:
                        if failed:
                            feedback_output = code_err
                            feedback_code = self.code_lines
                        else:
                            feedback_output = prev_code_ret['output'] if isinstance(prev_code_ret, dict) else prev_code_ret
                            feedback_code = code_lines

                        _old_code_lines = copy(self.code_lines)
                        self.code_lines = copy(feedback_code)
                        self.latest_feedback = self.feedback(feedback_output)
                        self.code_lines = _old_code_lines
                    except Exception as _fb_e:
                        self.latest_feedback = f"[Feedback generation failed]: {_fb_e}"

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score
        
        # No valid command found
        if not self.suppress_print: 
            print("$$$$ INVALID DATA COMMAND (failed)")
        self.latest_feedback = "No valid data command executed in this turn."
        return "Command not supported for data preparation, choose from existing commands", None, None, None, None
    
    def _extract_data_stats(self, code_output):
        """
        Extract and store data statistics from code execution output
        
        Args:
            code_output (str): Output from code execution
        """
        try:
            # Look for data shape information
            shape_matches = re.findall(r'shape:\s*\((\d+),\s*(\d+)\)', code_output)
            if shape_matches:
                rows, cols = shape_matches[0]
                self.data_stats['rows'] = int(rows)
                self.data_stats['columns'] = int(cols)
            
            # Look for missing value information
            missing_matches = re.findall(r'Missing values:\s*(\d+)', code_output)
            if missing_matches:
                self.data_stats['missing_values'] = int(missing_matches[0])
            
            # Look for data types information
            if 'dtypes' in code_output:
                dtype_section = code_output.split('dtypes')[1].split('\n\n')[0]
                self.data_stats['dtypes_summary'] = dtype_section.strip()
            
            # Look for correlation information
            if 'correlation' in code_output.lower():
                self.data_stats['has_correlation_info'] = True
            
            # Store validation metrics if available
            if 'validation score:' in code_output.lower():
                score_matches = re.findall(r'validation score:\s*([\d\.]+)', code_output.lower())
                if score_matches:
                    self.validation_metrics['score'] = float(score_matches[0])
        except Exception as e:
            # Silently handle extraction errors
            if self.verbose:
                print(f"Error extracting data stats: {e}")

    def history_str(self):
        """
        Generate well-formatted history string for context
        
        Returns:
            str: History string with data preparation context
        """
        hist_str = ""
        for _hist in range(len(self.st_history)):
            hist_str += f"-------- Data Preparation History ({len(self.st_history)-_hist} steps ago) -----\n"
            hist_str += f"Because of the following response: {self.st_history[_hist][0]}\n" if len(self.st_history[_hist][0]) > 0 else ""
            hist_str += f"and the following COMMAND response output: {self.st_history[_hist][3]}\n"
            hist_str += f"With the following data preparation code used: {'#'*20}\n{self.st_history[_hist][2]}\n{'#'*20}\n\n"
            hist_str += f"The environment feedback and reflection was as follows: {self.st_history[_hist][1]}\n"
            
            # Add data statistics if available
            if self.data_stats:
                hist_str += f"Data statistics: {self.data_stats}\n"
                
            hist_str += f"-------- End of data preparation history ({len(self.st_history)-_hist} steps ago) -------\n"
            
        return hist_str

    def system_prompt(self, commands=True):
        """
        Produce a system prompt for the data-solver focused on data preparation
        
        Args:
            commands (bool): Whether to include command instructions
            
        Returns:
            str: System prompt for data preparation
        """
        # 如果之前的数据集下载失败，强制提示更换数据集
        dataset_failed_note = ""  # 默认无额外提示
        if hasattr(self, 'dataset_failed') and self.dataset_failed:
            dataset_failed_note = (
                "\n\nIMPORTANT: The previously selected dataset failed to download or was unreachable. "
                "In your VERY NEXT COMMAND, you MUST switch to a different lightweight public dataset (≤500MB) "
                "available from sources like Hugging Face Datasets, Kaggle, or the UCI repository. "
                "Do NOT attempt to download the same dataset again."
            )

        return (
            # ROLE DESCRIPTION
            f"{self.role_description()}.\n"
            # ENV INFO
            f"Runtime environment information (Python/GPU/Packages):\n{self.env_info}\n"
            # TASK INSTRUCTIONS
            f"The following are your task instructions: {self.phase_prompt()}\n"
            # LIT REVIEW INSIGHTS
            f"Provided below are some insights from a literature review summary:\n{self.insights}\n"
            # CODE INSIGHTS
            f"{self.code_reflect}"
            # NOTES
            f"The following are notes, instructions, and general tips for you: {self.notes}"
            # PLAN DESCRIPTION
            f"You are given a data preparation task described, where the plan is described as follows: {self.plan}\n"
            # DATASET DESCRIPTION            
            f"{self.generate_dataset_descr_prompt()}"
            # Data statistics if available
            f"{self._format_data_stats()}\n"
            # Transition
            f"Your goal is to prepare high-quality data for the research plan. You will receive a score after you write the code and should aim to maximize the score by following best practices for data preparation.\n"
            f"Before each data processing step, please include a print statement explaining what the step is doing and what it will produce.\n"
            # COMMAND SET
            f"The following are commands you have access to: {self.command_descriptions()}\n. You should try to have a diversity of command responses if appropriate. Do not repeat the same command too many times." if commands else ""
        )
    
    def _format_data_stats(self):
        """Format data statistics for inclusion in prompts"""
        if not self.data_stats:
            return ""
            
        stats_str = "Current data statistics:\n"
        for key, value in self.data_stats.items():
            stats_str += f"- {key}: {value}\n"
            
        if self.validation_metrics:
            stats_str += "\nValidation metrics:\n"
            for key, value in self.validation_metrics.items():
                stats_str += f"- {key}: {value}\n"
                
        return stats_str

    def generate_code_lines(self, code):
        """
        Generate well-formatted code lines with line numbers
        
        Args:
            code (list): List of code line strings
            
        Returns:
            str: Code lines formatted with line numbers
        """
        codestr = str()
        for _index in range(len(code)):
            codestr += f"{_index} |{code[_index]}\n"
        return codestr

    def feedback(self, code_return):
        """
        Provide execution feedback after data preparation command is run
        
        Args:
            code_return (str): Return from code execution
            
        Returns:
            str: Feedback string with data preparation context
        """
        if code_return is not None:
            code_str = self.generate_code_lines(self.code_lines)
            
            if "[CODE EXECUTION ERROR]" in code_return:
                if not self.suppress_print: 
                    print(f"@@@@ DATA PREP ERROR")
                reflect_prompt = (
                    f"This is your data preparation code: {code_str}\n\n"
                    f"Your code returned the following error {code_return}. "
                    f"Please provide a detailed reflection on why this error occurred in the data preparation process, "
                    f"which lines in the code caused this error, and exactly (line by line) how you hope to fix this "
                    f"in the next update. Focus on data-specific issues like missing values, incorrect data types, "
                    f"or data validation problems. This reflection will help your future self fix the error better.\n"
                    f"If the error suggests that the selected dataset cannot be accessed or downloaded — for instance the dataset URL is unreachable, the repository does not exist, or network requests repeatedly timeout — then in your NEXT COMMAND you MUST switch to a different lightweight public dataset rather than retrying the same one. Do NOT rely on exact keyword matching; make this judgment based on the semantic meaning of the error message itself."
                )
            elif "data preparation complete" in code_return.lower() or "preprocessing complete" in code_return.lower():
                self.prev_working_code = copy(self.code_lines)
                grade_return = get_data_score(
                    self.plan, 
                    "\n".join(self.prev_working_code), 
                    code_return, 
                    openai_api_key=self.openai_api_key, 
                    REWARD_MODEL_LLM=self.llm_str
                )[0]
                
                if not self.suppress_print: 
                    print(f"@@@@ DATA PREPARATION COMPLETE: quality score {grade_return}")
                    
                reflect_prompt = (
                    f"This is your data preparation code: {code_str}\n\n"
                    f"Your code successfully completed the data preparation process. "
                    f"Your data quality score was {grade_return}.\n\n"
                    f"Consider further improving your data preparation through advanced techniques like:"
                    f"1. More sophisticated handling of outliers and missing values\n"
                    f"2. Better feature engineering and selection\n"
                    f"3. Enhanced data validation and quality checks\n"
                    f"4. More efficient data processing pipeline\n"
                    f"Please provide a detailed reflection on how to improve your data preparation, "
                    f"which lines in the code could be enhanced, and exactly (line by line) how you hope "
                    f"to improve this in the next update."
                )
            else:
                if not self.suppress_print: 
                    print("@@@@ Incomplete data preparation")
                reflect_prompt = (
                    f"This is your data preparation code: {code_str}\n\n"
                    f"Your code did not return an error, but also did not clearly indicate successful data preparation. "
                    f"Please reflect on how you can improve your data preparation process for the next cycle to ensure "
                    f"it properly loads, cleans, validates, and prepares the data for the research task."
                )
        elif not self.should_execute_code:
            code_return = "No changes were made to the data preparation code."
            reflect_prompt = "Reflect on your future plans and next steps to improve the data preparation process."
            
        reflection = self.reflection(reflect_prompt, code_str, code_return)
        return f"Data preparation code return: {code_return}\n\nReflection: {reflection}"

    def reflection(self, reflect_prompt, code_str, code_return):
        """
        Generate reflection on data preparation code
        
        Args:
            reflect_prompt (str): Reflection prompt
            code_str (str): Code string
            code_return (str): Code execution return
            
        Returns:
            str: Reflection string
        """
        # Create a data preparation focused system prompt
        data_sys_prompt = self.system_prompt(commands=False) + (
            "You are a data preparation expert. Your task is to analyze the provided code and execution results, "
            "then provide specific, actionable feedback to improve the data preparation process."
        )
        
        # Get reflection from the model
        refl = query_model(
            prompt=reflect_prompt, 
            system_prompt=data_sys_prompt, 
            model_str=f"{self.llm_str}", 
            openai_api_key=self.openai_api_key
        )
        
        return (
            f"During the previous execution, the following data preparation code was run: \n\n{code_str}\n\n"
            f"This code returned the following: \n{code_return}\n"
            f"The following is your reflection from this feedback {refl}\n"
        )

    def generate_dataset_descr_prompt(self):
        """
        Generate description prompt for dataset
        
        Returns:
            str: Dataset description prompt
        """
        return f"\n- The following dataset code will be added to the beginning of your data preparation code always, so this does not need to be rewritten: {self.dataset_code}"

    def phase_prompt(self):
        """
        Describe system role and general tips for data preparation
        
        Returns:
            str: Phase prompt for data preparation
        """
        # 如果之前的数据集下载失败，强制提示更换数据集
        dataset_failed_note = ""  # 默认无额外提示
        if hasattr(self, 'dataset_failed') and self.dataset_failed:
            dataset_failed_note = (
                "\n\nIMPORTANT: The previously selected dataset failed to download or was unreachable. "
                "In your VERY NEXT COMMAND, you MUST switch to a different lightweight public dataset (≤500MB) "
                "available from sources like Hugging Face Datasets, Kaggle, or the UCI repository. "
                "Do NOT attempt to download the same dataset again."
            )

        phase_str = (
            "You are a data scientist specializing in data preparation and preprocessing.\n"
            "Your goal is to produce high-quality code that prepares data for machine learning research. "
            "You should focus on data cleaning, feature engineering, and validation to ensure the data is "
            "ready for modeling. The dataset code will be added to the beginning of your code always, so "
            "this does not need to be rewritten.\n\n"
            "Your data preparation should include:\n"
            "1. Loading and initial exploration of the data\n"
            "2. Handling missing values appropriately\n"
            "3. Detecting and addressing outliers\n"
            "4. Feature engineering and selection\n"
            "5. Data validation and quality checks\n"
            "6. Preparing train/test splits when appropriate\n"
            "7. Scaling and normalization when needed\n\n"
            "IMPORTANT: Always download lightweight datasets from online sources rather than using local datasets. "
            "Prefer datasets from Hugging Face, Kaggle, or UCI ML Repository that are small in size (preferably under 500MB). "
            "This ensures better reproducibility and avoids local file dependency issues. If using PyTorch or TensorFlow "
            "built-in datasets, choose the smallest appropriate version for the task.\n\n"
            "You cannot pip install new libraries, but many data science libraries already work including "
            "pandas, numpy, scikit-learn, and matplotlib. Use print statements to show important data statistics "
            "and validation results." + dataset_failed_note
        )
        return phase_str

    def role_description(self):
        """
        Provide role description for data preparation expert
        
        Returns:
            str: Role description
        """
        return "You are an expert data scientist specializing in data preparation, cleaning, and preprocessing for machine learning research."

    @staticmethod
    def _common_code_errors():
        """
        Some general tips to avoid common data preparation code errors
        
        Returns:
            str: Common code errors
        """
        return (
            "Make sure to import everything that you are using.\n"
            "Handle missing values appropriately before applying operations that don't support them.\n"
            "Check data types before performing operations that require specific types.\n"
            "Validate your data transformations with print statements showing before/after statistics.\n"
            "YOU MUST USE COMMANDS PROPERLY. Do not use the word COMMAND for the command that is incorrect. "
            "You must use an actual command (e.g. EDIT, REPLACE...) NOT THE WORD COMMAND.\n"
            "Ensure your data preparation code is robust to different data distributions and edge cases.\n"
        )

    def command_descriptions(self):
        """
        Provide command descriptions for data preparation
        
        Returns:
            str: Command descriptions
        """
        cmd_strings = "\n".join([_cmd.docstring() for _cmd in self.commands])
        return (
            f"\nYou have access to tools which can be interacted with using the following structure: "
            f"```COMMAND\n<command information here>\n```, where COMMAND is whichever command you want to run "
            f"(e.g. EDIT, REPLACE...), <command information here> is information used for the command, such as "
            f"code to run, and ``` are meant to encapsulate the command. ``` must be included as part of the "
            f"command both at the beginning and at the end of the code. DO NOT FORGOT TO HAVE ``` AT THE TOP AND "
            f"BOTTOM OF CODE. This structure must be followed to execute a command correctly. "
            f"YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME! Do not try to perform multiple commands EVER only one. "
            f"{self._common_code_errors()}"
        ) + cmd_strings

    def run_code(self):
        """
        Execute the data preparation code that was generated
        
        Returns:
            str: Code execution return
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            # Determine appropriate directory for execution
            lab_dir = self.lab_dir
            if lab_dir is None:
                # Get current working directory as lab_dir
                current_dir = os.getcwd()
                # Check if running in user directory
                if "user_data" in current_dir:
                    lab_dir = current_dir
                else:
                    # Try to get output directory from environment variable
                    lab_dir = os.environ.get("CURRENT_OUTPUT_DIR", current_dir)
            else:
                # 确保使用绝对路径
                lab_dir = os.path.abspath(lab_dir)
                
            # 打印明确的目录信息，便于排查问题
            if not self.suppress_print:
                print(f"数据准备代码将在目录 {lab_dir} 中执行")
                    
            # Execute the code
            code_result = execute_code("\n".join(self.code_lines), lab_dir=lab_dir)
            
            # Check for log file and read contents if available
            log_file = code_result.get("log_file") if isinstance(code_result, dict) else None
            log_content = ""
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    if not self.suppress_print:
                        print(f"Execution log saved to: {log_file}")
                except Exception as e:
                    if not self.suppress_print:
                        print(f"Error reading execution log: {e}")
            
            # Ensure string output is returned
            if isinstance(code_result, dict):
                output = code_result["output"]
                # Add log content if available and different from output
                if log_content and log_content.strip() != output.strip():
                    output += f"\n\n[Program Execution Log]\n{log_content}"
                # 如果检测到数据集下载失败标记，则设置标志位，供下次提示使用
                if "[DATASET_DOWNLOAD_FAILED]" in output:
                    self.dataset_failed = True
                return output
            elif isinstance(code_result, str):
                # Handle direct string return case
                return code_result
            else:
                # Handle other return types
                return str(code_result)
                
        return "Changes have not yet been made to the data preparation code."

    @staticmethod
    def _collect_env_info():
        """
        Collect brief information about the current Python runtime, GPU availability and a subset of installed packages.
        Returns:
            str: JSON-like string summarizing environment info (truncated if overly long)
        """
        import sys, json
        info = {
            "python_version": sys.version.split(" ")[0]
        }
        # GPU info via torch if available
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except ImportError:
            info["cuda_available"] = False
        # Include subset of installed packages for brevity
        try:
            import pkg_resources
            pkgs = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
            info["top_packages"] = pkgs[:80]
        except Exception:
            pass
        env_json = json.dumps(info, ensure_ascii=False)
        if len(env_json) > 4000:
            env_json = env_json[:4000] + "...(truncated)"
        return env_json
