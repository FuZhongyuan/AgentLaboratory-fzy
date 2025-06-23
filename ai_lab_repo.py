import PyPDF2
import threading
from agents import *
from copy import copy
from pathlib import Path
from datetime import date
from common_imports import *
from mlesolver import MLESolver
from datasolver import DataSolver
import argparse, pickle, yaml
import json
import time
import multiprocessing
from tools import *

GLOBAL_AGENTRXIV = None
DEFAULT_LLM_BACKBONE = "o4-mini-yunwu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaboratoryWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None, compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=5, datasolver_max_steps=3, paper_index=0, except_if_fail=False, parallelized=False, lab_dir=None, lab_index=0, agentRxiv=False, agentrxiv_papers=5, state_callback=None):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        @param state_callback: (function) callback function to notify when state is saved
        """
        self.agentRxiv = agentRxiv
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        self.lab_index = lab_index
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key
        self.except_if_fail = except_if_fail
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review
        self.state_callback = state_callback

        self.print_cost = True
        self.review_override = True # should review be overridden?
        self.review_ovrd_steps = 0 # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0 # num steps to take if overridden
        self.arxiv_num_summaries = 5
        self.num_agentrxiv_papers = agentrxiv_papers
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps
        self.datasolver_max_steps = datasolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            # todo: check if valid
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)


    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        state_file = None
        
        # 使用用户特定的目录（lab_dir）来保存状态
        if self.lab_dir:
            # 在用户研究目录中创建state_saves子目录
            state_dir = os.path.join(self.lab_dir, "state_saves")
            os.makedirs(state_dir, exist_ok=True)
            
            # 保存到用户特定的目录
            state_file = os.path.join(state_dir, f"Paper{self.paper_index}_{phase}.pkl")
            with open(state_file, "wb") as f:
                pickle.dump(self, f)
            if self.verbose:
                print(f"状态已保存到 {state_file}")
        else:
            # 如果没有指定lab_dir，则使用默认目录
            try:
                os.makedirs("state_saves", exist_ok=True)
                state_file = f"state_saves/Paper{self.paper_index}_{phase}.pkl"
                with open(state_file, "wb") as f:
                    pickle.dump(self, f)
                if self.verbose:
                    print(f"状态已保存到 {state_file}")
            except Exception as e:
                print(f"保存状态时出错: {e}")
                # 状态保存失败不应该影响整个研究流程
                pass
        
        # 如果有回调函数，调用它并传递当前阶段和状态文件路径
        if self.state_callback and state_file:
            self.state_callback(phase, state_file)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.agentRxiv:
                    if self.verbose: print(f"{'&' * 30}\n[Lab #{self.lab_index} Paper #{self.paper_index}] Beginning subtask: {subtask}\n{'&' * 30}")
                else:
                    if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else: self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: 
                            try:
                                self.save_state(subtask)
                            except Exception as e:
                                print(f"保存状态时出错: {e}")
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: 
                    try:
                        self.save_state(subtask)
                    except Exception as e:
                        print(f"保存状态时出错: {e}")
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose: print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate mle-solver
        from papersolver import PaperSolver
        self.reference_papers = []
        solver = PaperSolver(notes=report_notes, max_steps=self.papersolver_max_steps, plan=self.phd.plan, exp_code=self.phd.results_code, exp_results=self.phd.exp_results, insights=self.phd.interpretation, lit_review=self.phd.lit_review, ref_papers=self.reference_papers, topic=self.research_topic, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["report writing"], compile_pdf=self.compile_pdf, save_loc=self.lab_dir)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps): solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        match = re.search(r'\\title\{([^}]*)\}', report)
        if match: report_title = match.group(1).replace(" ", "_")
        else: report_title = "\n".join([str(random.randint(0, 10)) for _ in range(10)])
        if self.agentRxiv: shutil.copyfile(self.lab_dir + "/tex/temp.pdf", f"uploads/{report_title}.pdf")
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file(self.lab_dir, "readme.md", readme)
        save_to_file(self.lab_dir, "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, notes=experiment_notes, insights=self.ml_engineer.lit_review_sum, max_steps=self.mlesolver_max_steps, plan=self.ml_engineer.plan, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["running experiments"], lab_dir=self.lab_dir)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # 执行代码并传递lab_dir，以便使用正确的用户目录保存文件
        code_result = execute_code(code, lab_dir=self.lab_dir)
        code_output = code_result["output"]
        code_file = code_result.get("code_file", None)
        log_file = code_result.get("log_file", None)
        
        if self.verbose:
            print(f"CODE OUTPUT: {code_output}")
            if code_file:
                print(f"代码已保存到: {code_file}")
            if log_file:
                print(f"执行日志已保存到: {log_file}")
                
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry: return retry
        save_to_file(os.path.join(self.lab_dir, "src"), "run_experiments.py", code)
        # 确保 exp_results 是字符串类型
        if isinstance(exp_results, dict):
            exp_results_str = json.dumps(exp_results, indent=2)
        else:
            exp_results_str = str(exp_results)
        save_to_file(os.path.join(self.lab_dir, "src"), "experiment_output.log", exp_results_str)
        # 如果有执行日志，将其作为附加信息保存
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                save_to_file(os.path.join(self.lab_dir, "src"), "execution_log.txt", log_content)
            except Exception as e:
                print(f"读取或保存执行日志时出错: {e}")
        # set results, reset agent state
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        self.reset_agents()
        self.statistics_per_phase["running experiments"]["steps"] = self.mlesolver_max_steps
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        # 数据准备笔记
        data_prep_notes = [_note["note"] for _note in self.ml_engineer.notes if "data preparation" in _note["phases"]]
        data_prep_notes = f"Notes for the data preparation task: {data_prep_notes}\n" if len(data_prep_notes) > 0 else ""
        
        # 确保lab_dir是绝对路径
        user_lab_dir = self.lab_dir
        if user_lab_dir:
            user_lab_dir = os.path.abspath(user_lab_dir)
            if self.verbose:
                print(f"数据准备将使用用户特定目录: {user_lab_dir}")
        
        # 实例化 DataSolver
        solver = DataSolver(dataset_code="", notes=data_prep_notes, insights=self.ml_engineer.lit_review_sum, 
                           max_steps=self.datasolver_max_steps, plan=self.ml_engineer.plan, 
                           openai_api_key=self.openai_api_key, 
                           llm_str=self.model_backbone["data preparation"], 
                           lab_dir=user_lab_dir)
        
        # 运行初始化
        solver.initial_solve()
        
        # 运行求解器进行数据准备优化
        for _ in range(self.datasolver_max_steps-1):
            solver.solve()
            
        # 获取最佳代码结果
        code = "\n".join(solver.best_codes[0][0])
        code_output = solver.run_code()
        
        if self.verbose:
            print(f"DATA PREPARATION OUTPUT: {code_output}")
            
        # 评分
        score = solver.best_codes[0][1]
        if self.verbose: 
            print(f"Data preparation completed, quality score: {score}")
            
        # 人工干预检查
        if self.human_in_loop_flag["data preparation"]:
            retry = self.human_in_loop("data preparation", code)
            if retry: return retry
            
        # 保存代码
        save_to_file(os.path.join(user_lab_dir, "src"), "load_data.py", code)
        self.set_agent_attr("dataset_code", code)
        
        # 重置代理状态
        self.reset_agents()
        self.statistics_per_phase["data preparation"]["steps"] = self.datasolver_max_steps
        return False

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            # inference postdoc to
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Plan Formulation")
        else:
            plan = "No plan specified."
            if self.human_in_loop_flag["plan formulation"]:
                retry = self.human_in_loop("plan formulation", plan)
                if retry: return retry
            self.set_agent_attr("plan", plan)
            # reset agent state
            self.reset_agents()
            return False

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps # lit review often requires extra steps
        # get initial response from PhD agent
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)
        if self.verbose: print(resp, "\n~~~~~~~~~~~")
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            feedback = str()
            
            # 在每次迭代开始时，提供已添加论文的列表信息
            if len(self.phd.lit_review) > 0:
                added_papers = "已添加的论文列表:\n" + "\n".join([f"- {paper['arxiv_id']}: {paper['summary'][:100]}..." for paper in self.phd.lit_review])
                feedback = added_papers + "\n\n"
            
            # grab summary of papers from arxiv
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                if self.agentRxiv:
                    if GLOBAL_AGENTRXIV.num_papers() > 0:
                        papers += GLOBAL_AGENTRXIV.search_agentrxiv(query, self.num_agentrxiv_papers,)
                feedback += f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            # grab full text from arxiv ID
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                
                # 检查是否已经下载过这篇论文
                existing_paper = next((paper for paper in self.phd.lit_review if paper["arxiv_id"] == query), None)
                if existing_paper:
                    feedback = f"论文 {query} 已经下载过，无需重复下载。以下是论文内容：\n{existing_paper['full_text']}"
                else:
                    if self.agentRxiv and "AgentRxiv" in query: full_text = GLOBAL_AGENTRXIV.retrieve_full_text(query,)
                    else: full_text = arx_eng.retrieve_full_paper_text(query)
                    # expiration timer so that paper does not remain in context too long
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + full_text + "```"
                    feedback = arxiv_paper

            # if add paper, extract and add to lit review, provide feedback
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                if self.agentRxiv and "AgentRxiv" in query: feedback, text = self.phd.add_review(query, arx_eng, agentrxiv=True, GLOBAL_AGENTRXIV=GLOBAL_AGENTRXIV)
                else: feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            # completion condition
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "literature review", feedback=feedback, step=_i + 1, temp=0.4)
            if self.verbose: print(resp, "\n~~~~~~~~~~~")
        if self.except_if_fail: raise Exception("Max tries during phase: Literature Review")
        else:
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False

class AgentRxiv:
    def __init__(self, lab_index=0, port=None):
        self.lab_index = lab_index
        self.server_thread = None
        # 不再初始化服务器，因为应该只通过app.py启动
        # self.initialize_server()
        self.pdf_text = dict()
        self.summaries = dict()
        self.port = port if port is not None else 5000 + self.lab_index
        
        # 检查服务器是否已经运行
        if not self.check_server_running():
            print(f"[警告] AgentRxiv 服务器未在端口 {self.port} 运行")
            print(f"请先运行 'python app.py --port {self.port}' 以启动服务器")
            print("继续执行，但在需要访问AgentRxiv时可能会失败")
    
    def check_server_running(self):
        """检查服务器是否已经运行"""
        try:
            # 尝试连接到服务器
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', self.port))
            s.close()
            return result == 0
        except:
            return False

    def initialize_server(self):
        # 这个方法保留但不再启动服务器
        # 端口号已经在初始化方法中设置
        print(f"AgentRxiv使用端口 {self.port}，请确保app.py已经在该端口上运行")

    @staticmethod
    def num_papers():
        return len(os.listdir("uploads"))

    def retrieve_full_text(self, arxiv_id):
        try:
            return self.pdf_text[arxiv_id]
        except Exception:
            return "Paper ID not found?"

    @staticmethod
    def read_pdf_pypdf2(pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def search_agentrxiv(self, search_query, num_papers):
        # 使用动态端口号
        port = 5000 + self.lab_index
        url = f'http://127.0.0.1:{port}/api/search?q={search_query}'
        return_str = str()
        try:
            # 不再导入app模块和使用应用上下文
            # 直接通过API请求获取数据
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return_str += "Search Query:" + data['query']
            return_str += "Results:"
            for result in data['results'][:num_papers]:
                arxiv_id = f"AgentRxiv:ID_{result['id']}"
                if arxiv_id not in self.summaries:
                    filename = Path(f'_tmp_{self.lab_index}.pdf')
                    response = requests.get(result['pdf_url'])
                    filename.write_bytes(response.content)
                    self.pdf_text[arxiv_id] = self.read_pdf_pypdf2(f'_tmp_{self.lab_index}.pdf')
                    self.summaries[arxiv_id] = query_model(
                        prompt=self.pdf_text[arxiv_id],
                        system_prompt="Please provide a 5 sentence summary of this paper.",
                        openai_api_key=os.getenv('OPENAI_API_KEY'),
                        model_str="gpt-4o-mini"
                    )
                return_str += f"Title: {result['filename']}"
                return_str += f"Summary: {self.summaries[arxiv_id]}\n"
                formatted_date = date.today().strftime("%d/%m/%Y")
                return_str += f"Publication Date: {formatted_date}\n"
                return_str += f"arXiv paper ID: AgentRxiv:ID_{result['id']}"
                return_str += "-" * 40
        except Exception as e:
            print(f"AgentRxiv Error: {e}")
            return_str += f"Error: {e}"
        return return_str

    def run_server(self, port):
        # 不再从app中导入run_app并执行
        # 此方法现在只是一个占位符，因为服务器应该只通过app.py启动
        print(f"[INFO] AgentRxiv server should be started separately via 'python app.py' with port {port}")
        # 可以在这里添加一些监控代码，但不应该启动服务器


def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        '--yaml-location',
        type=str,
        default="experiment_configs/MATH_agentlab.yaml",
        help='Location of YAML to load config data.'
    )

    return parser.parse_args()


def parse_yaml(yaml_file_loc):
    with open(yaml_file_loc, 'r') as file: agentlab_data = yaml.safe_load(file)
    class YamlDataHolder:
        def __init__(self): pass
    parser = YamlDataHolder()
    if "copilot_mode" in agentlab_data: parser.copilot_mode = agentlab_data["copilot_mode"]
    else: parser.copilot_mode = False
    if 'load-previous' in agentlab_data: parser.load_previous = agentlab_data["load-previous"]
    else: parser.load_previous = False
    if 'research-topic' in agentlab_data: parser.research_topic = agentlab_data["research-topic"]
    if 'api-key' in agentlab_data: parser.api_key = agentlab_data["api-key"]
    if 'deepseek-api-key' in agentlab_data: parser.deepseek_api_key = agentlab_data["deepseek-api-key"]
    if 'compile-latex' in agentlab_data: parser.compile_latex = agentlab_data["compile-latex"]
    else: parser.compile_latex = True
    if 'research-dir-path' in agentlab_data: parser.research_dir_path = agentlab_data["research-dir-path"]
    else: parser.research_dir_path = "MATH_research_dir"
    if 'llm-backend' in agentlab_data: parser.llm_backend = agentlab_data["llm-backend"]
    else: parser.llm_backend = "o3-mini"
    if 'lit-review-backend' in agentlab_data: parser.lit_review_backend = agentlab_data["lit-review-backend"]
    else: parser.lit_review_backend = "gpt-4o-mini"
    if 'language' in agentlab_data: parser.language = agentlab_data["language"]
    else: parser.language = "English"
    if 'num-papers-lit-review' in agentlab_data: parser.num_papers_lit_review = agentlab_data["num-papers-lit-review"]
    else: parser.num_papers_lit_review = 5
    if 'mlesolver-max-steps' in agentlab_data: parser.mlesolver_max_steps = agentlab_data["mlesolver-max-steps"]
    else: parser.mlesolver_max_steps = 3
    if 'papersolver-max-steps' in agentlab_data: parser.papersolver_max_steps = agentlab_data["papersolver-max-steps"]
    else: parser.papersolver_max_steps = 5
    if 'task-notes' in agentlab_data: parser.task_notes = agentlab_data["task-notes"]
    else: parser.task_notes = []
    if 'num-papers-to-write' in agentlab_data: parser.num_papers_to_write = agentlab_data["num-papers-to-write"]
    else: parser.num_papers_to_write = 100
    if 'parallel-labs' in agentlab_data: parser.parallel_labs = agentlab_data["parallel-labs"]
    else: parser.parallel_labs = False
    if 'num-parallel-labs' in agentlab_data: parser.num_parallel_labs = agentlab_data["num-parallel-labs"]
    else: parser.num_parallel_labs = 8
    if 'except-if-fail' in agentlab_data: parser.except_if_fail = agentlab_data["except-if-fail"]
    else: parser.except_if_fail = False
    if 'agentRxiv' in agentlab_data: parser.agentRxiv = agentlab_data["agentRxiv"]
    else: parser.agentRxiv = False
    if 'construct-agentRxiv' in agentlab_data: parser.construct_agentRxiv = agentlab_data["construct-agentRxiv"]
    else: parser.construct_agentRxiv = False
    if 'agentrxiv-papers' in agentlab_data: parser.agentrxiv_papers = agentlab_data["agentrxiv-papers"]
    else:  parser.agentrxiv_papers = 5

    if 'lab-index' in agentlab_data: parser.lab_index = agentlab_data["lab-index"]
    else: parser.lab_index = 0
    return parser


if __name__ == "__main__":
    user_args = parse_arguments()
    yaml_to_use = user_args.yaml_location
    args = parse_yaml(yaml_to_use)

    llm_backend = args.llm_backend
    human_mode =  args.copilot_mode.lower() == "true" if type(args.copilot_mode) == str else args.copilot_mode
    compile_pdf = args.compile_latex.lower() == "true" if type(args.compile_latex) == str else args.compile_latex
    load_previous = args.load_previous.lower() == "true" if type(args.load_previous) == str else args.load_previous
    parallel_labs = args.parallel_labs.lower() == "true" if type(args.parallel_labs) == str else args.parallel_labs
    except_if_fail = args.except_if_fail.lower() == "true" if type(args.except_if_fail) == str else args.except_if_fail
    agentRxiv = args.agentRxiv.lower() == "true" if type(args.agentRxiv) == str else args.agentRxiv
    construct_agentRxiv = args.construct_agentRxiv.lower() == "true" if type(args.construct_agentRxiv) == str else args.construct_agentRxiv
    lab_index = int(args.lab_index) if type(args.construct_agentRxiv) == str else args.lab_index
    research_dir_path = args.research_dir_path  # 使用配置中的研究目录路径
    
    # 如果启用了AgentRxiv，提醒用户确保app.py已经运行
    if agentRxiv:
        print("\n" + "="*60)
        print("注意: 您已启用AgentRxiv功能")
        print("请确保已通过以下命令启动app.py服务器:")
        print(f"python app.py --port {5000 + lab_index}")
        print("="*60 + "\n")
        
        # 创建一个临时AgentRxiv对象来检查服务器是否运行
        temp_agent_rxiv = AgentRxiv(lab_index)
        if not temp_agent_rxiv.check_server_running():
            user_continue = input("服务器似乎未运行。是否继续执行？(y/n): ").strip().lower()
            if user_continue != 'y':
                print("执行已取消。请先启动app.py服务器后再运行此脚本。")
                sys.exit(1)

    try: num_papers_to_write = int(args.num_papers_to_write.lower()) if type(args.num_papers_to_write) == str else args.num_papers_to_write
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: num_papers_lit_review = int(args.num_papers_lit_review.lower()) if type(args.num_papers_lit_review) == str else args.num_papers_lit_review
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: papersolver_max_steps = int(args.papersolver_max_steps.lower()) if type(args.papersolver_max_steps) == str else args.papersolver_max_steps
    except Exception: raise Exception("args.papersolver_max_steps must be a valid integer!")
    try: mlesolver_max_steps = int(args.mlesolver_max_steps.lower()) if type(args.mlesolver_max_steps) == str else args.mlesolver_max_steps
    except Exception: raise Exception("args.mlesolver_max_steps must be a valid integer!")
    if parallel_labs:
        num_parallel_labs = int(args.num_parallel_labs)
        print("="*20 , f"RUNNING {num_parallel_labs} LABS IN PARALLEL", "="*20)
    else: num_parallel_labs = 0

    api_key = (os.getenv('OPENAI_API_KEY') or args.api_key) if (hasattr(args, 'api_key') or os.getenv('OPENAI_API_KEY')) else None
    deepseek_api_key = (os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key) if (hasattr(args, 'deepseek_api_key') or os.getenv('DEEPSEEK_API_KEY')) else None
    if api_key is not None and os.getenv('OPENAI_API_KEY') is None: os.environ["OPENAI_API_KEY"] = args.api_key
    if deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None: os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key: raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")

    if human_mode or args.research_topic is None: research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else: research_topic = args.research_topic

    task_notes_LLM = list()
    task_notes = args.task_notes

    # 收集所有实际涉及的任务阶段
    phases_in_notes = set()

    for _task in task_notes:
        readable_phase = _task.replace("-", " ")
        phases_in_notes.add(readable_phase)
        for _note in task_notes[_task]:
            task_notes_LLM.append({"phases": [readable_phase], "note": _note})

    # 添加数据准备阶段的提示词，要求从网上下载轻量数据集
    task_notes_LLM.append({
        "phases": ["data preparation"],
        "note": "Always prefer to download lightweight datasets from online sources rather than using local datasets. Use datasets from Hugging Face, Kaggle, or UCI ML Repository that are small in size (preferably under 100MB). This ensures better reproducibility and avoids local file dependency issues. If using PyTorch or TensorFlow built-in datasets, choose the smallest appropriate version for the task."
    })

    # 如果指定语言不是英语，添加通用语言提示
    if args.language != "English":
        task_notes_LLM.append(
            {
                "phases": list(phases_in_notes),
                "note": f"You should always write in the following language to converse and to write the report: {args.language}"
            }
        )
    print(task_notes_LLM)


    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "results interpretation": llm_backend,
        "report writing":         llm_backend,
        "report refinement":      llm_backend,
    }
    if parallel_labs:
        remove_figures()
        # 检查app.py服务器是否运行
        print("\n" + "="*60)
        print("注意: 并行模式下启用了AgentRxiv功能")
        print("请确保已通过以下命令启动app.py服务器:")
        for i in range(num_parallel_labs):
            print(f"python app.py --port {5000 + i}")
        print("="*60 + "\n")
        
        GLOBAL_AGENTRXIV = AgentRxiv()  # 这里不再启动服务器，只初始化对象用于API访问
        remove_directory(f"{research_dir_path}")
        os.mkdir(os.path.join(".", f"{research_dir_path}"))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if not compile_pdf: raise Exception("PDF compilation must be used with agentRxiv!")
        def run_lab(parallel_lab_index):
            time_str = str()
            time_now = time.time()
            
            # 检查该实验室对应的端口上的服务器是否运行
            temp_agent_rxiv = AgentRxiv(parallel_lab_index)
            if not temp_agent_rxiv.check_server_running():
                print(f"[警告] 实验室 #{parallel_lab_index} 的AgentRxiv服务器未在端口 {5000 + parallel_lab_index} 运行")
                print(f"该实验室的执行可能会失败，如果需要使用AgentRxiv功能")
            
            for _paper_index in range(num_papers_to_write):
                lab_dir = os.path.join(research_dir_path, f"research_dir_lab{parallel_lab_index}_paper{_paper_index}")
                os.mkdir(lab_dir)
                os.mkdir(os.path.join(lab_dir, "src"))
                os.mkdir(os.path.join(lab_dir, "tex"))
                lab_instance = LaboratoryWorkflow(
                    parallelized=True,
                    research_topic=research_topic,
                    notes=task_notes_LLM,
                    agent_model_backbone=agent_models,
                    human_in_loop_flag=human_in_loop,
                    openai_api_key=api_key,
                    compile_pdf=compile_pdf,
                    num_papers_lit_review=num_papers_lit_review,
                    papersolver_max_steps=papersolver_max_steps,
                    mlesolver_max_steps=mlesolver_max_steps,
                    paper_index=_paper_index,
                    lab_index=parallel_lab_index,
                    except_if_fail=except_if_fail,
                    lab_dir=lab_dir,
                    agentRxiv=True,
                    agentrxiv_papers=args.agentrxiv_papers
                )
                lab_instance.perform_research()
                time_str += str(time.time() - time_now) + " | "
                with open(f"agent_times_{parallel_lab_index}.txt", "w") as f:
                    f.write(time_str)
                time_now = time.time()

        with ThreadPoolExecutor(max_workers=num_parallel_labs) as executor:
            futures = [executor.submit(run_lab, lab_idx) for lab_idx in range(num_parallel_labs)]
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: print(f"Error in lab: {e}")

    else:
        # remove previous files
        remove_figures()
        if agentRxiv: GLOBAL_AGENTRXIV = AgentRxiv(lab_index)  # 这里不再启动服务器，只初始化对象用于API访问
        if not agentRxiv:
            remove_directory(f"{research_dir_path}")
            os.mkdir(os.path.join(".", f"{research_dir_path}"))
        # make src and research directory
        if not os.path.exists("state_saves"): os.mkdir(os.path.join(".", "state_saves"))
        time_str = str()
        time_now = time.time()
        for _paper_index in range(num_papers_to_write):
            lab_direct = f"{research_dir_path}/research_dir_{_paper_index}_lab_{lab_index}"
            os.mkdir(os.path.join(".", lab_direct))
            os.mkdir(os.path.join(os.path.join(".", lab_direct), "src"))
            os.mkdir(os.path.join(os.path.join(".", lab_direct), "tex"))
            lab = LaboratoryWorkflow(
                research_topic=research_topic,
                notes=task_notes_LLM,
                agent_model_backbone=agent_models,
                human_in_loop_flag=human_in_loop,
                openai_api_key=api_key,
                compile_pdf=compile_pdf,
                num_papers_lit_review=num_papers_lit_review,
                papersolver_max_steps=papersolver_max_steps,
                mlesolver_max_steps=mlesolver_max_steps,
                paper_index=_paper_index,
                except_if_fail=except_if_fail,
                agentRxiv=False,
                lab_index=lab_index,
                lab_dir=os.path.join(".", lab_direct)
            )
            lab.perform_research()
            time_str += str(time.time() - time_now) + " | "
            with open(f"agent_times_{lab_index}.txt", "w") as f:
                f.write(time_str)
            time_now = time.time()

"""
@@@@@@@@@@@@@@@ CHECKLIST @@@@@@@@@@@@@@@ 
Practical:
----------
- Make a better config system (YAML?)

Advancements:
-------------
- Make the ability to have agents build on top of their own research
- Run agent labs in parallel (asynch) 

"""
