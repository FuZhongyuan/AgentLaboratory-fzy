import random, time, uuid
from datetime import datetime, timedelta, timezone
import threading
import yaml
import subprocess
import sys
import pickle
import json

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session, Response, stream_with_context
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil

# 移除直接导入，改为在需要的函数中导入
AI_LAB_AVAILABLE = True
try:
    # 测试是否可以导入，但不在模块顶层导入
    import ai_lab_repo
    AI_LAB_AVAILABLE = True
    # 从ai_lab_repo导入DEFAULT_LLM_BACKBONE
    from ai_lab_repo import DEFAULT_LLM_BACKBONE
except ImportError:
    AI_LAB_AVAILABLE = False
    print("警告：无法导入AI实验室，研究功能将不可用")
    # 如果无法导入，定义默认值
    DEFAULT_LLM_BACKBONE = "o4-mini-yunwu"

app = Flask(__name__)
app.config['SECRET_KEY'] = "sk-VdXR5cG1MtxNmgm9yosUgAQvy1Xcdx5U8bOOWfRQA0Rr9Cob"
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['USER_DATA_FOLDER'] = 'user_data/'  # 用户数据主目录
app.config['RESEARCH_CONFIG_FOLDER'] = 'research_configs/'  # 研究配置目录

# 确保数据库目录存在
db_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(db_dir, 'Database')
os.makedirs(db_path, exist_ok=True)

# 使用绝对路径设置数据库URI
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(db_path, "papers.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 会话有效期24小时

# 创建必要的目录
for dir_path in [app.config['UPLOAD_FOLDER'], app.config['USER_DATA_FOLDER'], app.config['RESEARCH_CONFIG_FOLDER']]:
    os.makedirs(dir_path, exist_ok=True)

db = SQLAlchemy(app)

class Paper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    text = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.String(36), nullable=True)  # 用户ID，关联到会话

class User(db.Model):
    id = db.Column(db.String(36), primary_key=True)  # UUID作为用户ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

# 确保存在用户数据目录
os.makedirs(app.config['USER_DATA_FOLDER'], exist_ok=True)

def update_papers_from_uploads():
    """从上传文件夹更新论文数据库"""
    max_retries = 5
    
    for retry in range(max_retries):
        try:
            uploads_dir = app.config['UPLOAD_FOLDER']
            pdf_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')]
            print(f"在上传文件夹中找到 {len(pdf_files)} 个PDF文件")
            
            # 获取当前用户ID（如果在会话中）
            user_id = session.get('user_id')
            
            for filename in pdf_files:
                # 检查文件是否已在数据库中
                if Paper.query.filter_by(filename=filename).first():
                    continue
                
                print(f"处理文件: {filename}")
                file_path = os.path.join(uploads_dir, filename)
                
                # 提取PDF文本
                extracted_text = extract_pdf_text(file_path)
                if not extracted_text.strip():
                    print(f"警告: 无法从 {filename} 提取文本")
                    continue
                
                print(f"从 {filename} 提取了 {len(extracted_text)} 个字符")
                
                # 创建新的论文记录
                new_paper = Paper(filename=filename, text=extracted_text, user_id=user_id)
                db.session.add(new_paper)
            
            db.session.commit()
            return
            
        except Exception as e:
            print(f"更新论文时发生错误: {e}")
            if retry < max_retries - 1:
                wait_time = random.randint(5, 15)
                print(f"将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"达到最大重试次数 ({max_retries})，放弃更新")
                return

def extract_pdf_text(file_path):
    """从PDF文件提取文本"""
    extracted_text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
    except Exception as e:
        print(f'处理PDF文件时出错 {os.path.basename(file_path)}: {e}')
    
    return extracted_text

# 用户会话管理
@app.before_request
def check_user_session():
    """确保每个请求都有用户会话"""
    if 'user_id' not in session:
        # 创建新用户会话
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
        session.permanent = True
        
        # 创建用户记录
        new_user = User(id=user_id)
        db.session.add(new_user)
        db.session.commit()
        
        # 创建用户数据目录
        user_dir = os.path.join(app.config['USER_DATA_FOLDER'], user_id)
        os.makedirs(user_dir, exist_ok=True)
    else:
        # 更新用户最后活动时间
        user = User.query.get(session['user_id'])
        if user:
            user.last_active = datetime.utcnow()
            db.session.commit()

# 获取当前用户的数据目录
def get_user_data_dir():
    """获取当前用户的数据目录路径"""
    user_id = session.get('user_id')
    if user_id:
        user_dir = os.path.join(app.config['USER_DATA_FOLDER'], user_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    return None

# 全局变量，用于延迟初始化
model = None

# Load a pre-trained sentence transformer model
# print("准备加载句向量模型...")
# 将模型初始化移到函数中

@app.route('/update', methods=['GET'])
def update_on_demand():
    update_papers_from_uploads()
    return jsonify({"message": "Uploads folder processed successfully."})

@app.route('/')
def index():
    update_papers_from_uploads()
    # 只显示当前用户的论文或公共论文
    user_id = session.get('user_id')
    papers = Paper.query.filter((Paper.user_id == user_id) | (Paper.user_id == None)).all()
    return render_template('index.html', papers=papers)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            
            # 获取用户目录
            user_dir = get_user_data_dir()
            if not user_dir:
                flash('User session error')
                return redirect(url_for('index'))
                
            # 保存到用户目录内的uploads子文件夹
            user_uploads_dir = os.path.join(user_dir, 'uploads')
            os.makedirs(user_uploads_dir, exist_ok=True)
            file_path = os.path.join(user_uploads_dir, filename)
            file.save(file_path)
            
            extracted_text = ""
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
            except Exception as e:
                flash(f'Error processing PDF: {e}')
                
            # 添加用户ID
            new_paper = Paper(filename=filename, text=extracted_text, user_id=session['user_id'])
            db.session.add(new_paper)
            db.session.commit()
            flash('File uploaded and processed successfully!')
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/search')
def search():
    global model
    
    query = request.args.get('q', '')
    if query:
        user_id = session.get('user_id')
        # 只搜索当前用户的论文或公共论文
        papers = Paper.query.filter((Paper.user_id == user_id) | (Paper.user_id == None)).all()
        
        # 检查是否有论文和模型
        if not papers:
            return render_template('search.html', papers=[], query=query)
            
        # 检查模型是否已初始化
        if model is None:
            try:
                model_dir = './hugging-face/m3e-base'
                if not os.path.exists(model_dir):
                    model_name = 'moka-ai/m3e-base'
                else:
                    model_name = model_dir
                model = SentenceTransformer(model_name)
            except Exception as e:
                flash(f'搜索功能暂时不可用: {str(e)}')
                return render_template('search.html', papers=[], query=query)
                
        try:
            query_embedding = model.encode([query])
            paper_texts = [paper.text for paper in papers if paper.text]
            if not paper_texts:
                return render_template('search.html', papers=[], query=query)
            paper_embeddings = model.encode(paper_texts)
            similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
            papers_with_scores = list(zip([p for p in papers if p.text], similarities))
            papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
            return render_template('search.html', papers=papers_sorted, query=query)
        except Exception as e:
            flash(f'搜索过程中发生错误: {str(e)}')
            return render_template('search.html', papers=[], query=query)
            
    return render_template('search.html', papers=[], query=query)

@app.route('/api/search')
def api_search():
    global model
    
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    user_id = session.get('user_id')
    # 只搜索当前用户的论文或公共论文
    papers = Paper.query.filter((Paper.user_id == user_id) | (Paper.user_id == None)).all()
    if not papers:
        return jsonify({'query': query, 'results': []})
    
    # 检查模型是否已初始化
    if model is None:
        try:
            model_dir = './hugging-face/m3e-base'
            if not os.path.exists(model_dir):
                model_name = 'moka-ai/m3e-base'
            else:
                model_name = model_dir
            model = SentenceTransformer(model_name)
        except Exception as e:
            return jsonify({'error': f'搜索功能暂时不可用: {str(e)}'}), 500
    
    try:
        query_embedding = model.encode([query])
        paper_texts = [paper.text for paper in papers if paper.text]
        if not paper_texts:
            return jsonify({'query': query, 'results': []})
            
        paper_embeddings = model.encode(paper_texts)
        similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        papers_with_scores = list(zip([p for p in papers if p.text], similarities))
        papers_sorted = sorted(papers_with_scores, key=lambda x: x[1], reverse=True)
            
        results = []
        for paper, score in papers_sorted:
                # 获取用户特定的文件路径
                if paper.user_id:
                    user_uploads = os.path.join(app.config['USER_DATA_FOLDER'], paper.user_id, 'uploads')
                    pdf_url = url_for('user_file', user_id=paper.user_id, filename=paper.filename, _external=True)
                else:
                    pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
                            
                    results.append({
                        'id': paper.id,
                        'filename': paper.filename,
                        'similarity': float(score),
                        'pdf_url': pdf_url
                    })
                return jsonify({'query': query, 'results': results})
    except Exception as e:
        return jsonify({'error': f'搜索过程中发生错误: {str(e)}'}), 500

# 公共文件访问
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

# 用户特定文件访问
@app.route('/user_data/<user_id>/uploads/<path:filename>')
def user_file(user_id, filename):
    # 只允许当前用户访问自己的文件
    if session.get('user_id') == user_id:
        user_uploads = os.path.join(app.config['USER_DATA_FOLDER'], user_id, 'uploads')
        return send_from_directory(user_uploads, filename)
    flash('Access denied')
    return redirect(url_for('index'))

@app.route('/view/<int:paper_id>')
def view_pdf(paper_id):
    paper = Paper.query.get_or_404(paper_id)
    
    # 检查当前用户是否有权限查看
    if paper.user_id and paper.user_id != session.get('user_id'):
        flash('Access denied')
        return redirect(url_for('index'))
        
    # 构建适当的PDF URL
    if paper.user_id:
        pdf_url = url_for('user_file', user_id=paper.user_id, filename=paper.filename, _external=True)
    else:
        pdf_url = url_for('uploaded_file', filename=paper.filename, _external=True)
        
    return render_template('view.html', paper=paper, pdf_url=pdf_url)

# 添加任务模型
class ResearchTask(db.Model):
    id = db.Column(db.String(36), primary_key=True)  # 使用UUID作为主键
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    config_path = db.Column(db.String(255), nullable=True)
    result_path = db.Column(db.String(255), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    report_language = db.Column(db.String(20), default='English')  # 报告最终语言

# 添加研究状态模型
class ResearchState(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(36), db.ForeignKey('research_task.id'), nullable=False)
    phase = db.Column(db.String(50), nullable=False)
    state_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 建立与任务的关系
    task = db.relationship('ResearchTask', backref=db.backref('states', lazy=True))

# 全局状态保存回调函数
def save_state_callback(task_id, phase, state_path):
    """
    保存研究状态的回调函数
    
    参数:
        task_id (str): 任务ID
        phase (str): 当前研究阶段
        state_path (str): 状态文件路径
    """
    with app.app_context():
        try:
            # 检查是否已存在相同阶段的状态记录
            existing_state = ResearchState.query.filter_by(
                task_id=task_id, 
                phase=phase
            ).first()
            
            if existing_state:
                # 更新现有记录
                existing_state.state_path = state_path
                existing_state.created_at = datetime.utcnow()
            else:
                # 创建新记录
                new_state = ResearchState(
                    task_id=task_id,
                    phase=phase,
                    state_path=state_path
                )
                db.session.add(new_state)
            
            db.session.commit()
            print(f"已保存研究状态: 任务 {task_id}, 阶段 {phase}")
        except Exception as e:
            db.session.rollback()
            print(f"保存研究状态时出错: {e}")

# 全局变量，用于存储task_id
_TASK_ID_FOR_CALLBACK = None

# 全局回调函数，用于解决pickle序列化问题
def global_state_callback(phase, path):
    """
    全局回调函数，将通过state_callback_wrapper被间接调用
    
    参数:
        phase (str): 研究阶段
        path (str): 状态文件路径
    """
    return save_state_callback(_TASK_ID_FOR_CALLBACK, phase, path)

# 全局回调函数包装器，用于解决pickle序列化问题
def state_callback_wrapper(task_id):
    """
    创建一个可序列化的回调函数
    
    参数:
        task_id (str): 任务ID
    
    返回:
        function: 可序列化的回调函数
    """
    global _TASK_ID_FOR_CALLBACK
    _TASK_ID_FOR_CALLBACK = task_id
    return global_state_callback

# 运行研究任务的函数
def run_research_task(task_id, user_id, topic, config_path):
    """
    后台运行研究任务
    """
    if not AI_LAB_AVAILABLE:
        update_task_status(task_id, 'failed', error_message="AI实验室模块不可用")
        return
    
    try:
        # 延迟导入LaboratoryWorkflow，避免循环导入问题
        from ai_lab_repo import LaboratoryWorkflow, AgentRxiv
        import ai_lab_repo
        
        # 用户数据目录
        user_dir = os.path.join(app.config['USER_DATA_FOLDER'], user_id)
        research_dir = os.path.join(user_dir, f"research_{task_id}")
        os.makedirs(research_dir, exist_ok=True)
        
        # 创建研究相关目录
        os.makedirs(os.path.join(research_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(research_dir, "tex"), exist_ok=True)
        os.makedirs(os.path.join(research_dir, "state_saves"), exist_ok=True)
        
        # 加载配置
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取目标报告语言
        target_language = config.get('language', app.config.get('DEFAULT_LANGUAGE', 'English'))
        
        # 存储目标语言到任务记录
        with app.app_context():
            task = ResearchTask.query.get(task_id)
            if task:
                task.report_language = target_language
                db.session.commit()
        
        # 更新研究主题 - 不再在主题前添加语言标记，保持英文交互
        clean_topic = topic
        if '[' in topic and ']' in topic:
            # 移除可能已有的语言标记
            clean_topic = topic[topic.index(']')+1:].strip()
        
        # 更新配置
        config['research-topic'] = clean_topic
        config['research-dir-path'] = research_dir
        
        # 写入更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
            
        # 获取OpenAI API密钥（从环境变量或配置中）
        api_key = config.get('api-key')
        if not api_key or api_key == 'OPENAI-API-KEY-HERE':
            api_key = os.environ.get('OPENAI_API_KEY', "sk-VdXR5cG1MtxNmgm9yosUgAQvy1Xcdx5U8bOOWfRQA0Rr9Cob")
        
        # 创建工作流实例的基本参数
        workflow_params = {
            'research_topic': clean_topic,
            'openai_api_key': api_key,
            'lab_dir': research_dir,
            'language': target_language,
            # 使用可序列化的回调函数
            'state_callback': state_callback_wrapper(task_id),
        }
        
        # 辅助函数：从配置中提取参数，如果不存在则使用默认值
        def get_param(config_key, param_key, default_config_key=None):
            if config_key in config:
                return config[config_key]
            elif default_config_key and default_config_key in app.config:
                return app.config[default_config_key]
            return None
        
        # 添加可选参数
        param_mappings = [
            ('num-papers-lit-review', 'num_papers_lit_review', 'DEFAULT_NUM_PAPERS_LIT_REVIEW'),
            ('agentrxiv-papers', 'agentrxiv_papers', 'DEFAULT_AGENTRXIV_PAPERS'),
            ('compile-latex', 'compile_pdf', 'DEFAULT_COMPILE_LATEX'),
            ('mlesolver-max-steps', 'mlesolver_max_steps', 'DEFAULT_MLESOLVER_MAX_STEPS'),
            ('datasolver-max-steps', 'datasolver_max_steps', 'DEFAULT_DATASOLVER_MAX_STEPS'),
            ('papersolver-max-steps', 'papersolver_max_steps', 'DEFAULT_PAPERSOLVER_MAX_STEPS'),
            ('lab-index', 'lab_index', 'DEFAULT_LAB_INDEX')
        ]
        
        for config_key, param_key, default_key in param_mappings:
            value = get_param(config_key, param_key, default_key)
            if value is not None:
                workflow_params[param_key] = value
        
        # 处理任务笔记
        if 'task-notes' in config:
            task_notes_LLM = []
            task_notes = config['task-notes']
            phases_in_notes = set()
            
            # 收集所有任务阶段并转换笔记格式
            for task_name, notes in task_notes.items():
                readable_phase = task_name.replace("-", " ")
                phases_in_notes.add(readable_phase)
                for note in notes:
                    task_notes_LLM.append({"phases": [readable_phase], "note": note})
            
            # 添加数据准备阶段的提示词，要求从网上下载轻量数据集
            task_notes_LLM.append({
                "phases": ["data preparation"],
                "note": "Always prefer to download lightweight datasets from online sources rather than using local datasets. Use datasets from Hugging Face, Kaggle, or UCI ML Repository that are small in size (preferably under 500MB). This ensures better reproducibility and avoids local file dependency issues. If using PyTorch or TensorFlow built-in datasets, choose the smallest appropriate version for the task."
            })
            
            # 添加语言提示 - 始终使用英文进行交互
            if phases_in_notes:
                task_notes_LLM.append({
                    "phases": list(phases_in_notes),
                    "note": "You should always write in English to converse and write the initial report."
                })
                
            workflow_params['notes'] = task_notes_LLM
        
        # 设置human-in-loop标志
        research_phases = [
            "literature review", "plan formulation", "data preparation", 
            "running experiments", "results interpretation", 
            "report writing", "report translation", "report refinement"
        ]
        
        human_in_loop_flag = {phase: False for phase in research_phases}
        
        # 如果配置中有copilot-mode，则设置human-in-loop
        if config.get('copilot-mode', app.config.get('DEFAULT_COPILOT_MODE', False)):
            human_in_loop_flag = {phase: True for phase in research_phases}
                
        workflow_params['human_in_loop_flag'] = human_in_loop_flag
        
        # 设置agent模型
        llm_backend = config.get('llm-backend', app.config.get('DEFAULT_LLM_BACKBONE', 'o4-mini-yunwu'))
        workflow_params['agent_model_backbone'] = {phase: llm_backend for phase in research_phases}
        
        # 设置AgentRxiv
        agentRxiv = config.get('agentRxiv', app.config.get('DEFAULT_AGENTRXIV', False))
        workflow_params['agentRxiv'] = agentRxiv
        
        # 确保全局AgentRxiv已初始化
        if agentRxiv and not ai_lab_repo.GLOBAL_AGENTRXIV:
            lab_index = workflow_params.get('lab_index', 0)
            ai_lab_repo.GLOBAL_AGENTRXIV = AgentRxiv(lab_index=lab_index,model=llm_backend,api_key=api_key)
            print(f"为任务 {task_id} 初始化AgentRxiv")
        
        # 创建工作流实例并执行研究
        workflow = LaboratoryWorkflow(**workflow_params)
        workflow.perform_research()
        
        # 更新任务状态
        results_path = os.path.join(research_dir, "report.txt")
        update_task_status(task_id, 'completed', results_path)
        
    except Exception as e:
        import traceback
        error_msg = f"研究任务执行错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        update_task_status(task_id, 'failed', error_message=error_msg)

# 任务状态更新通知
task_update_listeners = {}

# SSE接口：监听任务状态更新
@app.route('/api/task_updates', methods=['GET'])
def task_updates():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
    
    def event_stream():
        # 为当前用户创建一个唯一的监听器ID
        listener_id = str(uuid.uuid4())
        
        # 创建该用户的消息队列
        if user_id not in task_update_listeners:
            task_update_listeners[user_id] = {}
        
        # 初始化队列
        task_update_listeners[user_id][listener_id] = []
        
        # 发送初始连接事件
        yield f"id: {listener_id}\n"
        yield f"event: connected\n"
        yield f"data: {json.dumps({'type': 'connected', 'message': '连接已建立'})}\n\n"
        
        try:
            # 发送重连指令和心跳间隔
            yield f"retry: 10000\n\n"  # 如果连接断开，10秒后自动重连
            
            heartbeat_count = 0
            while True:
                # 检查队列中是否有消息
                if task_update_listeners[user_id][listener_id]:
                    update = task_update_listeners[user_id][listener_id].pop(0)
                    yield f"event: task_update\n"
                    yield f"data: {json.dumps(update)}\n\n"
                else:
                    # 发送心跳包以保持连接
                    heartbeat_count += 1
                    yield f"event: heartbeat\n"
                    yield f"data: {json.dumps({'type': 'heartbeat', 'count': heartbeat_count})}\n\n"
                
                # 等待一小段时间
                time.sleep(2)
        finally:
            # 清理
            if user_id in task_update_listeners and listener_id in task_update_listeners[user_id]:
                del task_update_listeners[user_id][listener_id]
                if not task_update_listeners[user_id]:
                    del task_update_listeners[user_id]
                print(f"SSE连接已关闭: {user_id}/{listener_id}")
    
    return Response(stream_with_context(event_stream()), 
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform", 
                        "X-Accel-Buffering": "no",
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Connection": "keep-alive"
                    })

def update_task_status(task_id, status, result_path=None, error_message=None):
    """
    更新研究任务的状态
    
    参数:
        task_id (str): 任务ID
        status (str): 任务状态 (pending, running, completed, failed)
        result_path (str, optional): 结果文件路径
        error_message (str, optional): 错误信息
    """
    # 创建应用上下文
    with app.app_context():
        try:
            task = ResearchTask.query.get(task_id)
            if not task:
                print(f"警告: 找不到任务ID {task_id}")
                return
                
            # 更新任务状态
            task.status = status
            
            # 如果任务已完成或失败，记录完成时间
            if status in ['completed', 'failed']:
                task.completed_at = datetime.utcnow()
                
            # 更新结果路径和错误信息（如果提供）
            if result_path:
                task.result_path = result_path
            if error_message:
                task.error_message = error_message
                
            # 保存更改
            db.session.commit()
            print(f"任务 {task_id} 状态已更新为 {status}")
            
            # 通知前端任务状态已更新
            user_id = task.user_id
            if user_id in task_update_listeners:
                update_info = {
                    'type': 'task_update',
                    'task_id': task_id,
                    'status': status,
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                # 向该用户的所有连接发送更新
                for listener_id in list(task_update_listeners[user_id].keys()):
                    task_update_listeners[user_id][listener_id].append(update_info)
                    
        except Exception as e:
            print(f"更新任务状态时出错: {e}")
            try:
                db.session.rollback()
            except:
                pass

# 获取配置模板
@app.route('/api/config_template/<template_name>', methods=['GET'])
def get_config_template(template_name):
    """获取配置模板的内容"""
    try:
        # 安全检查模板名称
        if '..' in template_name or not template_name.endswith('.yaml'):
            return jsonify({'error': '无效的模板名称'}), 400
        
        template_path = os.path.join('experiment_configs', template_name)
        if not os.path.exists(template_path):
            return jsonify({'error': '模板不存在'}), 404
            
        import yaml
        with open(template_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        return jsonify({'success': True, 'config': config_data})
    except Exception as e:
        return jsonify({'error': f'读取配置模板失败: {str(e)}'}), 500

# 新接口：启动研究任务
@app.route('/api/start_research', methods=['POST'])
def start_research():
    if not request.is_json:
        return jsonify({'error': 'Missing JSON data'}), 400
        
    data = request.get_json()
    topic = data.get('topic')
    config_template = data.get('config_template')  # 保留此参数用于兼容旧版API调用
    custom_config = data.get('custom_config')
    template_custom_config = data.get('template_custom_config')
    paper_id = data.get('paper_id')  # 可选，用于基于特定论文的研究
    
    if not topic:
        return jsonify({'error': 'Missing research topic'}), 400
        
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
        
    try:
        # 创建任务记录
        task_id = str(uuid.uuid4())
        
        # 准备配置文件
        user_config_dir = os.path.join(app.config['USER_DATA_FOLDER'], user_id, 'configs')
        os.makedirs(user_config_dir, exist_ok=True)
        
        # 配置文件路径
        user_config_path = os.path.join(user_config_dir, f"research_config_{task_id}.yaml")
        
        # 辅助函数：处理配置并写入文件
        def process_config(config):
            # 设置研究主题
            config['research-topic'] = topic
            
            # 确保API密钥存在
            if 'api-key' not in config or not config['api-key'] or config['api-key'] == 'OPENAI-API-KEY-HERE':
                # 尝试从环境变量获取API密钥
                env_api_key = os.environ.get('OPENAI_API_KEY')
                if env_api_key:
                    print(f"使用环境变量中的API密钥")
                    config['api-key'] = env_api_key
                else:
                    print(f"未找到API密钥，使用默认密钥")
                    config['api-key'] = "sk-VdXR5cG1MtxNmgm9yosUgAQvy1Xcdx5U8bOOWfRQA0Rr9Cob"
            else:
                print(f"使用用户提供的API密钥")
            
            # 写入配置
            with open(user_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            return config
        
        # 处理配置来源
        import yaml
        config = None
        
        if template_custom_config:
            # 使用模板作为基础并自定义
            config = process_config(template_custom_config)
        elif config_template:
            # 兼容旧版API：将模板转换为自定义配置
            template_path = os.path.join('experiment_configs', config_template)
            if not os.path.exists(template_path):
                return jsonify({'error': '配置模板不存在'}), 404
                
            # 读取模板配置
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config = process_config(config)
        elif custom_config:
            # 使用自定义配置
            config = process_config(custom_config)
        else:
            # 默认使用MATH_agentlab.yaml作为模板
            template_path = os.path.join('experiment_configs', 'MATH_agentlab.yaml')
            
            # 读取模板配置
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config = process_config(config)
                
        # 如果指定了论文ID，在配置中添加论文引用
        if paper_id:
            paper = Paper.query.get(paper_id)
            if paper and config:
                try:
                    # 添加论文到任务备注
                    if 'task-notes' not in config:
                        config['task-notes'] = {}
                    
                    if 'literature-review' not in config['task-notes']:
                        config['task-notes']['literature-review'] = []
                    
                    # 创建论文引用备注
                    paper_note = f'参考论文: {paper.filename}'
                    if paper_note not in config['task-notes']['literature-review']:
                        config['task-notes']['literature-review'].append(paper_note)
                    
                    # 重新写入配置文件
                    with open(user_config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f)
                except Exception as e:
                    print(f"添加论文引用失败: {e}")
            
        # 创建任务记录
        new_task = ResearchTask(
            id=task_id,
            user_id=user_id,
            topic=topic,
            config_path=user_config_path,
            status='running'  # 直接设置为运行中状态，而不是默认的等待中
        )
        db.session.add(new_task)
        db.session.commit()
        
        # 启动研究任务线程
        task_thread = threading.Thread(
            target=run_research_task,
            args=(task_id, user_id, topic, user_config_path)
        )
        task_thread.daemon = True
        task_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Research task started',
            'user_id': user_id,
            'topic': topic,
            'task_id': task_id
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start research: {str(e)}'}), 500

# 获取任务状态
@app.route('/api/research_status/<task_id>', methods=['GET'])
def get_research_status(task_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
        
    task = ResearchTask.query.get(task_id)
    if not task or task.user_id != user_id:
        return jsonify({'error': 'Task not found'}), 404
        
    response = {
        'task_id': task.id,
        'topic': task.topic,
        'status': task.status,
        'created_at': task.created_at.isoformat(),
        'completed_at': task.completed_at.isoformat() if task.completed_at else None
    }
    
    if task.status == 'completed' and task.result_path:
        response['result_url'] = url_for('get_research_result', task_id=task_id, _external=True)
    
    if task.status == 'failed' and task.error_message:
        response['error'] = task.error_message
        
    return jsonify(response)

# 获取用户的所有研究任务
@app.route('/api/research_tasks', methods=['GET'])
def get_user_tasks():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
        
    tasks = ResearchTask.query.filter_by(user_id=user_id).order_by(ResearchTask.created_at.desc()).all()
    
    response = []
    for task in tasks:
        task_data = {
            'task_id': task.id,
            'topic': task.topic,
            'status': task.status,
            'created_at': task.created_at.isoformat(),
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        }
        response.append(task_data)
        
    return jsonify({'tasks': response})

# 获取研究结果
@app.route('/api/research_result/<task_id>', methods=['GET'])
def get_research_result(task_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
        
    task = ResearchTask.query.get(task_id)
    if not task or task.user_id != user_id:
        return jsonify({'error': 'Task not found'}), 404
        
    if task.status != 'completed' or not task.result_path:
        return jsonify({'error': 'Research results not available'}), 404
        
    try:
        # 获取研究报告文本
        with open(task.result_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
            
        # 返回结果
        response_data = {
            'task_id': task_id,
            'report': report_text,
            'language': task.report_language
        }
        
        # 获取生成的图像文件
        research_dir = os.path.dirname(task.result_path)
        image_files = []
        for root, dirs, files in os.walk(research_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, start=app.config['USER_DATA_FOLDER'])
                    image_url = url_for('research_file', file_path=relative_path, _external=True)
                    image_files.append({
                        'name': file,
                        'url': image_url
                    })
        
        response_data['images'] = image_files
        
        # 添加PDF下载链接，优先查找带语言标识的 PDF，防止语言版本冲突
        pdf_dir = os.path.join(research_dir, 'tex')
        lang_suffix = (task.report_language or 'English').replace(' ', '_')
        pdf_candidates = [f"temp_{lang_suffix}.pdf", "temp.pdf"]
        pdf_path = None
        for candidate in pdf_candidates:
            candidate_path = os.path.join(pdf_dir, candidate)
            if os.path.exists(candidate_path):
                pdf_path = candidate_path
                break

        if pdf_path:
            response_data['pdf_url'] = url_for(
                'research_file',
                file_path=os.path.relpath(pdf_path, start=app.config['USER_DATA_FOLDER']),
                _external=True
            )
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'Error retrieving research results: {str(e)}'}), 500

# 研究文件访问
@app.route('/research_files/<path:file_path>')
def research_file(file_path):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 401
    
    # 确保文件路径属于当前用户
    if not file_path.startswith(user_id + '/'):
        return jsonify({'error': 'Access denied'}), 403
        
    # 获取文件的完整路径
    full_path = os.path.join(app.config['USER_DATA_FOLDER'], file_path)
    
    # 获取文件所在目录
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    
    # 确定MIME类型
    mime_type = 'application/octet-stream'  # 默认
    if filename.lower().endswith('.pdf'):
        mime_type = 'application/pdf'
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        mime_type = f'image/{os.path.splitext(filename)[1][1:].lower()}'
    elif filename.lower().endswith('.txt'):
        mime_type = 'text/plain'
    
    return send_from_directory(directory, filename, mimetype=mime_type)

# 定期清理旧任务的定时任务
def cleanup_old_tasks():
    """清理超过30天的旧任务"""
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    with app.app_context():
        old_tasks = ResearchTask.query.filter(
            ResearchTask.created_at < thirty_days_ago,
            ResearchTask.status.in_(['completed', 'failed'])
        ).all()
        
        for task in old_tasks:
            # 删除任务相关文件
            try:
                if task.result_path and os.path.exists(task.result_path):
                    research_dir = os.path.dirname(task.result_path)
                    shutil.rmtree(research_dir, ignore_errors=True)
                
                if task.config_path and os.path.exists(task.config_path):
                    os.remove(task.config_path)
            except Exception as e:
                print(f"清理任务文件失败: {str(e)}")
                
            # 删除任务记录
            db.session.delete(task)
            
        db.session.commit()
        print(f"已清理 {len(old_tasks)} 个旧任务")

# 检查并关闭占用端口的进程
def check_and_kill_process(port=5000):
    """
    检查指定端口是否被占用，如果占用则尝试关闭该进程
    
    参数:
        port (int): 要检查的端口号
    """
    try:
        # 根据操作系统选择合适的命令
        if sys.platform.startswith('win'):
            check_port_windows(port)
        else:
            check_port_unix(port)
    except subprocess.CalledProcessError:
        # 没有找到占用端口的进程
        pass
    except Exception as e:
        print(f"检查端口占用时发生错误: {e}")

def check_port_windows(port):
    """Windows系统检查并尝试释放端口"""
    cmd = f'netstat -ano | findstr :{port}'
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
        if not output:
            return
            
        print(f"端口 {port} 被占用，正在尝试识别进程...")
        for line in output.splitlines():
            if f':{port}' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"发现占用端口 {port} 的进程 PID: {pid}")
                    # 注释掉实际终止进程的代码，仅显示信息
                    # subprocess.check_output(f'taskkill /F /PID {pid}', shell=True)
                    print(f"如需终止该进程，请手动运行: taskkill /F /PID {pid}")
    except subprocess.CalledProcessError:
        # 命令执行失败，可能是端口未被占用
        pass

def check_port_unix(port):
    """Unix/Linux/Mac系统检查并尝试释放端口"""
    cmd = f'lsof -i :{port} | grep LISTEN'
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
        if not output:
            return
            
        print(f"端口 {port} 被占用，正在尝试识别进程...")
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                pid = parts[1]
                print(f"发现占用端口 {port} 的进程 PID: {pid}")
                # 注释掉实际终止进程的代码，仅显示信息
                        # subprocess.check_output(f'kill -9 {pid}', shell=True)
                print(f"如需终止该进程，请手动运行: kill -9 {pid}")
    except subprocess.CalledProcessError:
        # 命令执行失败，可能是端口未被占用
        pass

# 在应用启动时设置定期清理任务（生产环境中应使用celery或类似工具）
def run_app(port=5000, config_file=None):
    global model
    
    # 检查并清理端口占用
    check_and_kill_process(port)
    
    # 创建必要的目录
    os.makedirs(app.config['USER_DATA_FOLDER'], exist_ok=True)
    
    # 初始化数据库
    with app.app_context():
        try:
            # 确保数据库目录存在
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            if db_uri.startswith('sqlite:///'):
                db_file_path = db_uri.replace('sqlite:///', '')
                db_dir = os.path.dirname(db_file_path)
                os.makedirs(db_dir, exist_ok=True)

            # 确保目录有写入权限
            if not os.access(db_dir, os.W_OK):
                os.chmod(db_dir, 0o755)  # 设置目录权限为 rwxr-xr-x
                print(f"目录 {db_dir} 权限已设置为可写")

            db.create_all()
            print("数据库初始化成功")
        except Exception as e:
            print(f"数据库初始化错误: {e}")
    
    # 加载配置文件
    if config_file:
        load_config_from_yaml(config_file)
    
    # 初始化模型
    initialize_sentence_model()
    
    # 初始化AgentRxiv全局变量
    initialize_agentrxiv(port)
    
    # 设置定期清理任务
    setup_cleanup_task()
        
    # 启动应用
    print(f"启动Web服务器，监听端口 {port}")
    app.run(debug=False, host='0.0.0.0', port=port)

def load_config_from_yaml(config_file):
    """从YAML文件加载配置"""
    try:
        print(f"加载配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 应用配置中的默认值
        app.config.update({
            'DEFAULT_LLM_BACKEND': config.get('llm-backend', 'o4-mini-yunwu'),
            'DEFAULT_NUM_PAPERS_LIT_REVIEW': ('num-papers-lit-review', 5),
            'DEFAULT_AGENTRXIV_PAPERS': ('agentrxiv-papers', 5),
            'DEFAULT_COMPILE_LATEX': ('compile-latex', True),
            'DEFAULT_MLESOLVER_MAX_STEPS': ('mlesolver-max-steps', 3),
            'DEFAULT_DATASOLVER_MAX_STEPS': ('datasolver-max-steps', 3),
            'DEFAULT_PAPERSOLVER_MAX_STEPS': ('papersolver-max-steps', 5),
            'DEFAULT_LAB_INDEX': ('lab-index', 0),
            'DEFAULT_COPILOT_MODE': config.get('copilot-mode', False),
            'DEFAULT_AGENTRXIV': config.get('agentRxiv', False),
            'DEFAULT_LLM_BACKBONE': DEFAULT_LLM_BACKBONE,
            'DEFAULT_TASK_NOTES': config.get('task-notes', {})
        })
            
        # 设置API密钥
        api_keys = {
            'OPENAI_API_KEY': 'api-key',
            'DEEPSEEK_API_KEY': 'deepseek-api-key'
        }
        
        for env_key, config_key in api_keys.items():
            if config_key in config and not os.environ.get(env_key):
                os.environ[env_key] = config[config_key]
                print(f"已从配置文件设置{env_key}")
                
        print("配置文件加载成功")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("将使用默认配置")

def initialize_sentence_model():
    """初始化句向量模型"""
    global model
    try:
        # 检查模型目录是否存在
        model_dir = './hugging-face/m3e-base'
        model_name = model_dir if os.path.exists(model_dir) else 'moka-ai/m3e-base'
        
        if model_name != model_dir:
            print(f"警告: 模型目录 {model_dir} 不存在，尝试直接从Hugging Face加载")
            
        print(f"加载句向量模型: {model_name}")
        model = SentenceTransformer(model_name)
        print("句向量模型加载成功!")
    except Exception as e:
        print(f"加载句向量模型失败: {e}")
        print("继续运行，但搜索功能将无法使用")
    
def initialize_agentrxiv(port):
    """初始化AgentRxiv全局变量"""
    if AI_LAB_AVAILABLE:
        try:
            from ai_lab_repo import AgentRxiv
            import ai_lab_repo
            ai_lab_repo.GLOBAL_AGENTRXIV = AgentRxiv(
                lab_index=app.config.get('DEFAULT_LAB_INDEX', 0), 
                port=port,
                model=app.config.get('DEFAULT_LLM_BACKBONE', 'o4-mini-yunwu'),
                api_key=app.config.get('OPENAI_API_KEY')
            )
            print(f"AgentRxiv已初始化，使用端口 {port}")
        except Exception as e:
            print(f"初始化AgentRxiv失败: {e}")

def setup_cleanup_task():
    """设置定期清理任务"""
    if not app.debug:
        import threading
        import time
        
        def cleanup_thread():
            while True:
                try:
                    cleanup_old_tasks()
                    print("完成旧任务清理")
                except Exception as e:
                    print(f"清理任务错误: {e}")
                # 每天运行一次
                time.sleep(86400)
                
        thread = threading.Thread(target=cleanup_thread)
        thread.daemon = True
        thread.start()

# 获取研究状态
@app.route('/api/research_state/<task_id>', methods=['GET'])
def get_research_state(task_id):
    """获取研究任务的状态历史"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': '用户会话未找到'}), 401
        
    # 检查任务是否属于当前用户
    task = ResearchTask.query.get(task_id)
    if not task or task.user_id != user_id:
        return jsonify({'error': '任务未找到'}), 404
    
    # 获取任务的所有状态记录
    states = ResearchState.query.filter_by(task_id=task_id).order_by(ResearchState.created_at.desc()).all()
    
    # 构建响应
    states_data = []
    for state in states:
        states_data.append({
            'phase': state.phase,
            'created_at': state.created_at.isoformat(),
            'state_path': state.state_path
        })
    
    return jsonify({
        'task_id': task_id,
        'topic': task.topic,
        'status': task.status,
        'states': states_data
    })

# 暂停研究任务
@app.route('/api/pause_research/<task_id>', methods=['POST'])
def pause_research(task_id):
    """暂停研究任务"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': '用户会话未找到'}), 401
        
    # 检查任务是否属于当前用户
    task = ResearchTask.query.get(task_id)
    if not task or task.user_id != user_id:
        return jsonify({'error': '任务未找到'}), 404
    
    # 只能暂停正在运行的任务
    if task.status != 'running':
        return jsonify({'error': '只能暂停正在运行的任务'}), 400
    
    # 更新任务状态为暂停
    update_task_status(task_id, 'paused')
    
    return jsonify({
        'success': True,
        'message': '研究任务已暂停',
        'task_id': task_id
    })

# 从保存的状态继续研究任务
@app.route('/api/resume_research/<task_id>', methods=['POST'])
def resume_research(task_id):
    """从保存的状态继续研究任务"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': '用户会话未找到'}), 401
        
    # 检查任务是否属于当前用户
    task = ResearchTask.query.get(task_id)
    if not task or task.user_id != user_id:
        return jsonify({'error': '任务未找到'}), 404
    
    # 只能继续已暂停的任务
    if task.status != 'paused':
        return jsonify({'error': '只能继续已暂停的任务'}), 400
    
    # 获取最新的状态记录
    latest_state = ResearchState.query.filter_by(task_id=task_id).order_by(ResearchState.created_at.desc()).first()
    
    if not latest_state or not latest_state.state_path or not os.path.exists(latest_state.state_path):
        return jsonify({'error': '找不到有效的保存状态'}), 404
    
    # 启动继续研究的线程
    thread = threading.Thread(
        target=continue_research_task,
        args=(task_id, user_id, latest_state.state_path, latest_state.phase)
    )
    thread.daemon = True
    thread.start()
    
    # 更新任务状态为运行中
    update_task_status(task_id, 'running')
    
    return jsonify({
        'success': True,
        'message': '研究任务已继续',
        'task_id': task_id,
        'resumed_from_phase': latest_state.phase
    })

# 继续研究任务的函数
def continue_research_task(task_id, user_id, state_path, phase):
    """
    从保存的状态继续研究任务
    
    参数:
        task_id (str): 任务ID
        user_id (str): 用户ID
        state_path (str): 状态文件路径
        phase (str): 恢复的阶段
    """
    if not AI_LAB_AVAILABLE:
        update_task_status(task_id, 'failed', error_message="AI实验室模块不可用")
        return
    
    try:
        # 延迟导入LaboratoryWorkflow，避免循环导入问题
        from ai_lab_repo import LaboratoryWorkflow, AgentRxiv
        import ai_lab_repo
        
        # 加载保存的状态
        with open(state_path, 'rb') as f:
            workflow = pickle.load(f)
        
        print(f"已从阶段 {phase} 的状态恢复研究任务 {task_id}")
        
        # 获取任务配置路径
        task = ResearchTask.query.get(task_id)
        if not task:
            print(f"警告: 找不到任务ID {task_id}")
            return
        
        # 用户数据目录
        user_dir = os.path.join(app.config['USER_DATA_FOLDER'], user_id)
        research_dir = os.path.join(user_dir, f"research_{task_id}")
        
        # 更新工作流参数
        workflow.state_callback = state_callback_wrapper(task_id)
        workflow.lab_dir = research_dir
        
        # 继续执行研究
        workflow.perform_research()
        
        # 更新任务状态
        results_path = os.path.join(research_dir, "report.txt")
        update_task_status(task_id, 'completed', results_path)
        
    except Exception as e:
        import traceback
        error_msg = f"继续研究任务时出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        update_task_status(task_id, 'failed', error_message=error_msg)

if __name__ == '__main__':
    # 添加命令行参数解析，整合ai_lab_repo.py的逻辑
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentLaboratory Web Server")
    parser.add_argument(
        '--port',
        type=int,
        default=6000,
        help='Web服务器监听端口'
    )
    parser.add_argument(
        '--yaml-location',
        type=str,
        default="experiment_configs/CIFAR_CNN.yaml",
        help='YAML配置文件路径，用于加载默认配置'
    )
    
    args = parser.parse_args()
    
    run_app(port=args.port, config_file=args.yaml_location)