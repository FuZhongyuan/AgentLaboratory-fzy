/**
 * AI实验室前端交互脚本
 */

document.addEventListener('DOMContentLoaded', function() {
    // 初始化导航栏交互
    initNavigation();
    
    // 初始化研究任务相关功能
    initResearchFeatures();
    
    // 初始化配置相关功能
    initConfigurationFeatures();
});

/**
 * 初始化导航栏交互
 */
function initNavigation() {
    // 开始研究按钮
    const startResearchBtn = document.getElementById('start-research-btn');
    if (startResearchBtn) {
        startResearchBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // 检查是否有任务正在进行
            checkActiveTasks().then(hasTasks => {
                if (hasTasks) {
                    // 如果有任务，显示任务列表
                    const tasksModal = new bootstrap.Modal(document.getElementById('tasksModal'));
                    loadTasks();
                    tasksModal.show();
                } else {
                    // 如果没有任务，显示新建研究对话框
                    const researchModal = new bootstrap.Modal(document.getElementById('researchModal'));
                    researchModal.show();
                }
            });
        });
    }
}

/**
 * 初始化研究任务相关功能
 */
function initResearchFeatures() {
    // 提交研究表单
    const submitResearchBtn = document.getElementById('submit-research');
    if (submitResearchBtn) {
        submitResearchBtn.addEventListener('click', function() {
            const topic = document.getElementById('research-topic').value;
            
            if (!topic) {
                alert('请输入研究主题');
                return;
            }
            
            // 禁用按钮，显示加载状态
            submitResearchBtn.disabled = true;
            submitResearchBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 处理中...';
            
            // 构建选项
            const options = {};
            
            // 如果有Paper ID参数，则添加到选项中（从search页面或view页面中传递过来）
            const paperIdInput = document.getElementById('paper-id-input');
            if (paperIdInput && paperIdInput.value) {
                options.paper_id = paperIdInput.value;
            }
            
            // 发送API请求
            startResearch(topic, options)
                .then(data => {
                    // 重置表单
                    document.getElementById('research-form').reset();
                    
                    // 隐藏模态框
                    const researchModal = bootstrap.Modal.getInstance(document.getElementById('researchModal'));
                    if (researchModal) {
                        researchModal.hide();
                    }
                    
                    // 显示任务列表
                    const tasksModal = new bootstrap.Modal(document.getElementById('tasksModal'));
                    loadTasks();
                    tasksModal.show();
                    
                    // 显示成功消息
                    showAlert('研究任务已成功启动！您可以在任务列表中查看进度。', 'success');
                })
                .catch(error => {
                    console.error('Error:', error);
                    showAlert('启动研究任务失败: ' + error, 'danger');
                })
                .finally(() => {
                    // 恢复按钮状态
                    submitResearchBtn.disabled = false;
                    submitResearchBtn.innerHTML = '开始研究';
                });
        });
    }
}

/**
 * 初始化配置相关功能
 */
function initConfigurationFeatures() {
    // 切换配置类型（模板自定义/完全自定义）
    const useTemplateCustomRadio = document.getElementById('use-template-custom');
    const useCustomRadio = document.getElementById('use-custom');
    const templateSection = document.getElementById('template-selection');
    const customSection = document.getElementById('custom-config-section');
    
    if (useTemplateCustomRadio && useCustomRadio) {
        // 模板作为基础并自定义切换事件
        useTemplateCustomRadio.addEventListener('change', function() {
            if (this.checked) {
                templateSection.classList.remove('d-none');
                customSection.classList.remove('d-none');
                
                // 加载选中的模板到自定义表单
                const selectedTemplate = document.getElementById('config-template').value;
                fetchConfigTemplate(selectedTemplate);
            }
        });
        
        // 完全自定义配置切换事件
        useCustomRadio.addEventListener('change', function() {
            if (this.checked) {
                templateSection.classList.add('d-none');
                customSection.classList.remove('d-none');
            }
        });
    }
    
    // 处理配置模板选择变更
    const configTemplateSelect = document.getElementById('config-template');
    if (configTemplateSelect) {
        configTemplateSelect.addEventListener('change', function() {
            const selectedTemplate = this.value;
            // 如果当前是"使用模板作为基础并自定义"模式，则加载模板到表单
            if (useTemplateCustomRadio && useTemplateCustomRadio.checked) {
                fetchConfigTemplate(selectedTemplate);
            }
        });
    }
    
    // 切换高级选项的显示状态
    const toggleAdvancedBtn = document.getElementById('toggle-advanced-options');
    const advancedOptions = document.getElementById('advanced-options');
    
    if (toggleAdvancedBtn && advancedOptions) {
        toggleAdvancedBtn.addEventListener('click', function() {
            const isHidden = advancedOptions.classList.contains('d-none');
            
            if (isHidden) {
                advancedOptions.classList.remove('d-none');
                this.innerHTML = '隐藏任务备注设置 <i class="bi bi-chevron-up"></i>';
            } else {
                advancedOptions.classList.add('d-none');
                this.innerHTML = '显示任务备注设置 <i class="bi bi-chevron-down"></i>';
            }
        });
    }
    
    // 初始加载默认配置
    if (configTemplateSelect && useTemplateCustomRadio && useTemplateCustomRadio.checked) {
        fetchConfigTemplate(configTemplateSelect.value);
    }
}

/**
 * 获取配置模板内容
 * @param {string} templateName - 模板文件名
 */
function fetchConfigTemplate(templateName) {
    fetch(`/api/config_template/${templateName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('无法加载配置模板');
            }
            return response.json();
        })
        .then(data => {
            populateConfigForm(data.config);
        })
        .catch(error => {
            console.error('加载配置模板失败:', error);
            showAlert('加载配置模板失败: ' + error.message, 'danger');
        });
}

/**
 * 使用配置数据填充表单
 * @param {object} config - 配置数据对象
 */
function populateConfigForm(config) {
    // 填充基本配置
    if (document.getElementById('language-select')) {
        document.getElementById('language-select').value = config.language || '中文';
    }
    
    // 清空API密钥字段，不自动填充API密钥
    if (document.getElementById('api-key')) {
        document.getElementById('api-key').value = '';
    }
    
    if (document.getElementById('llm-backend')) {
        document.getElementById('llm-backend').value = config['llm-backend'] || 'o4-mini-yunwu';
    }
    
    if (document.getElementById('lit-review-papers')) {
        document.getElementById('lit-review-papers').value = config['num-papers-lit-review'] || 5;
    }
    
    if (document.getElementById('agentrxiv-papers')) {
        document.getElementById('agentrxiv-papers').value = config['agentrxiv-papers'] || 5;
    }
    
    if (document.getElementById('papers-to-write')) {
        document.getElementById('papers-to-write').value = config['num-papers-to-write'] || 1;
    }
    
    if (document.getElementById('mlesolver-steps')) {
        document.getElementById('mlesolver-steps').value = config['mlesolver-max-steps'] || 3;
    }
    
    if (document.getElementById('papersolver-steps')) {
        document.getElementById('papersolver-steps').value = config['papersolver-max-steps'] || 1;
    }
    
    // 填充开关配置
    if (document.getElementById('copilot-mode')) {
        document.getElementById('copilot-mode').checked = config['copilot-mode'] !== false;
    }
    
    if (document.getElementById('compile-latex')) {
        document.getElementById('compile-latex').checked = config['compile-latex'] === true;
    }
    
    // 填充任务备注
    const taskNotes = config['task-notes'] || {};
    let hasNotes = false;
    
    // 处理各种任务备注
    const noteFields = [
        {id: 'literature-review-notes', key: 'literature-review'},
        {id: 'plan-formulation-notes', key: 'plan-formulation'},
        {id: 'data-preparation-notes', key: 'data-preparation'},
        {id: 'running-experiments-notes', key: 'running-experiments'},
        {id: 'results-interpretation-notes', key: 'results-interpretation'},
        {id: 'report-writing-notes', key: 'report-writing'}
    ];
    
    // 检查并填充每个备注字段
    noteFields.forEach(field => {
        const element = document.getElementById(field.id);
        if (element) {
            const notes = taskNotes[field.key] || [];
            if (notes.length > 0) {
                element.value = notes.join('\n');
                hasNotes = true;
                
                // 如果有备注，自动展开对应的折叠面板
                const collapseId = field.id.replace('-notes', 'Collapse');
                const collapseElement = document.getElementById(collapseId);
                if (collapseElement) {
                    const bsCollapse = new bootstrap.Collapse(collapseElement, {toggle: false});
                    bsCollapse.show();
                }
            } else {
                element.value = '';
            }
        }
    });
    
    // 如果有任何备注，自动显示任务备注设置区域
    if (hasNotes) {
        const advancedOptions = document.getElementById('advanced-options');
        const toggleAdvancedBtn = document.getElementById('toggle-advanced-options');
        
        if (advancedOptions && toggleAdvancedBtn) {
            advancedOptions.classList.remove('d-none');
            toggleAdvancedBtn.innerHTML = '隐藏任务备注设置 <i class="bi bi-chevron-up"></i>';
        }
    }
}

/**
 * 从表单中收集自定义配置数据
 * @returns {object} - 配置数据对象
 */
function collectCustomConfig() {
    const config = {
        'language': document.getElementById('language-select').value,
        'llm-backend': document.getElementById('llm-backend').value,
        'lit-review-backend': document.getElementById('llm-backend').value,
        'num-papers-lit-review': parseInt(document.getElementById('lit-review-papers').value) || 5,
        'agentrxiv-papers': parseInt(document.getElementById('agentrxiv-papers').value) || 5,
        'num-papers-to-write': parseInt(document.getElementById('papers-to-write').value) || 1,
        'mlesolver-max-steps': parseInt(document.getElementById('mlesolver-steps').value) || 3,
        'papersolver-max-steps': parseInt(document.getElementById('papersolver-steps').value) || 1,
        'copilot-mode': document.getElementById('copilot-mode').checked,
        'compile-latex': document.getElementById('compile-latex').checked,
        'parallel-labs': false,
        'lab-index': 1,
        'load-existing': false,
        'except-if-fail': false
    };
    
    // 添加API密钥（如果用户提供了）
    const apiKey = document.getElementById('api-key').value;
    if (apiKey && apiKey.trim() !== '') {
        config['api-key'] = apiKey.trim();
    }
    
    // 收集任务备注
    config['task-notes'] = {};
    
    // 处理文献综述备注
    const literatureReviewNotes = document.getElementById('literature-review-notes').value;
    if (literatureReviewNotes) {
        config['task-notes']['literature-review'] = literatureReviewNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    // 处理计划制定备注
    const planFormulationNotes = document.getElementById('plan-formulation-notes').value;
    if (planFormulationNotes) {
        config['task-notes']['plan-formulation'] = planFormulationNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    // 处理数据准备备注
    const dataPreparationNotes = document.getElementById('data-preparation-notes').value;
    if (dataPreparationNotes) {
        config['task-notes']['data-preparation'] = dataPreparationNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    // 处理运行实验备注
    const runningExperimentsNotes = document.getElementById('running-experiments-notes').value;
    if (runningExperimentsNotes) {
        config['task-notes']['running-experiments'] = runningExperimentsNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    // 处理结果解释备注
    const resultsInterpretationNotes = document.getElementById('results-interpretation-notes').value;
    if (resultsInterpretationNotes) {
        config['task-notes']['results-interpretation'] = resultsInterpretationNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    // 处理报告写作备注
    const reportWritingNotes = document.getElementById('report-writing-notes').value;
    if (reportWritingNotes) {
        config['task-notes']['report-writing'] = reportWritingNotes.split('\n').filter(note => note.trim() !== '');
    }
    
    return config;
}

/**
 * 启动新的研究任务
 * @param {string} topic - 研究主题
 * @param {object} options - 其他选项
 * @returns {Promise} - 返回Promise对象
 */
function startResearch(topic, options = {}) {
    // 处理配置选项
    const useTemplateCustom = document.getElementById('use-template-custom').checked;
    const useCustom = document.getElementById('use-custom').checked;
    let configData = {};
    
    if (useTemplateCustom) {
        // 使用模板作为基础并自定义
        configData = {
            template_custom_config: collectCustomConfig()
        };
    } else if (useCustom) {
        // 使用自定义配置
        configData = {
            custom_config: collectCustomConfig()
        };
    }
    
    return new Promise((resolve, reject) => {
        fetch('/api/start_research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                topic: topic,
                ...configData,
                ...options
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('服务器响应错误');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                resolve(data);
            } else {
                reject(data.error || '未知错误');
            }
        })
        .catch(error => {
            reject(error.message || '请求失败');
        });
    });
}

/**
 * 检查是否有活跃的研究任务
 * @returns {Promise<boolean>} - 返回是否有任务的Promise
 */
function checkActiveTasks() {
    return new Promise((resolve, reject) => {
        fetch('/api/research_tasks')
            .then(response => response.json())
            .then(data => {
                if (data.tasks && data.tasks.length > 0) {
                    resolve(true);
                } else {
                    resolve(false);
                }
            })
            .catch(error => {
                console.error('Error checking tasks:', error);
                resolve(false);
            });
    });
}

/**
 * 加载用户的研究任务列表
 */
function loadTasks() {
    const tableBody = document.getElementById('tasks-table-body');
    const noTasksMessage = document.getElementById('no-tasks-message');
    
    if (!tableBody || !noTasksMessage) return;
    
    // 清空表格
    tableBody.innerHTML = '<tr><td colspan="4" class="text-center"><span class="spinner-border" role="status"></span> 加载中...</td></tr>';
    
    // 获取任务列表
    fetch('/api/research_tasks')
        .then(response => response.json())
        .then(data => {
            tableBody.innerHTML = '';
            
            if (data.tasks && data.tasks.length > 0) {
                noTasksMessage.classList.add('d-none');
                
                // 按创建时间降序排序
                const sortedTasks = data.tasks.sort((a, b) => {
                    return new Date(b.created_at) - new Date(a.created_at);
                });
                
                // 添加任务到表格
                sortedTasks.forEach(task => {
                    const row = document.createElement('tr');
                    
                    // 创建日期对象
                    const createdDate = new Date(task.created_at);
                    
                    // 状态标签样式
                    let statusBadgeClass = 'badge ';
                    switch(task.status) {
                        case 'pending':
                            statusBadgeClass += 'badge-pending';
                            break;
                        case 'running':
                            statusBadgeClass += 'badge-running';
                            break;
                        case 'completed':
                            statusBadgeClass += 'badge-completed';
                            break;
                        case 'failed':
                            statusBadgeClass += 'badge-failed';
                            break;
                        case 'paused':
                            statusBadgeClass += 'badge-paused';
                            break;
                        default:
                            statusBadgeClass += 'bg-secondary';
                    }
                    
                    // 状态文本
                    let statusText = {
                        'pending': '等待中',
                        'running': '进行中',
                        'completed': '已完成',
                        'failed': '失败',
                        'paused': '已暂停'
                    }[task.status] || task.status;
                    
                    // 构建操作按钮
                    let actionButtons = '';
                    
                    // 查看结果按钮（仅完成状态可用）
                    if (task.status === 'completed') {
                        actionButtons += `<button class="btn btn-sm btn-primary view-result me-1" data-task-id="${task.task_id}">查看结果</button>`;
                    } else {
                        actionButtons += `<button class="btn btn-sm btn-secondary me-1" ${task.status === 'failed' ? '' : 'disabled'}>查看结果</button>`;
                    }
                    
                    // 暂停按钮（仅运行状态可用）
                    if (task.status === 'running') {
                        actionButtons += `<button class="btn btn-sm btn-warning pause-task me-1" data-task-id="${task.task_id}">暂停</button>`;
                    }
                    
                    // 继续按钮（仅暂停状态可用）
                    if (task.status === 'paused') {
                        actionButtons += `<button class="btn btn-sm btn-success resume-task" data-task-id="${task.task_id}">继续</button>`;
                    }
                    
                    // 构建行内容
                    row.innerHTML = `
                        <td>${task.topic}</td>
                        <td><span class="${statusBadgeClass}">${statusText}</span></td>
                        <td>${createdDate.toLocaleString()}</td>
                        <td>${actionButtons}</td>
                    `;
                    
                    tableBody.appendChild(row);
                });
                
                // 添加查看结果按钮事件
                document.querySelectorAll('.view-result').forEach(button => {
                    button.addEventListener('click', function() {
                        const taskId = this.getAttribute('data-task-id');
                        viewTaskResult(taskId);
                    });
                });
                
                // 添加暂停任务按钮事件
                document.querySelectorAll('.pause-task').forEach(button => {
                    button.addEventListener('click', function() {
                        const taskId = this.getAttribute('data-task-id');
                        pauseResearchTask(taskId);
                    });
                });
                
                // 添加继续任务按钮事件
                document.querySelectorAll('.resume-task').forEach(button => {
                    button.addEventListener('click', function() {
                        const taskId = this.getAttribute('data-task-id');
                        resumeResearchTask(taskId);
                    });
                });
            } else {
                noTasksMessage.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error loading tasks:', error);
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">加载任务失败</td></tr>';
        });
}

/**
 * 暂停研究任务
 * @param {string} taskId - 任务ID
 */
function pauseResearchTask(taskId) {
    if (!confirm('确定要暂停此研究任务吗？')) {
        return;
    }
    
    fetch(`/api/pause_research/${taskId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('服务器响应错误');
        }
        return response.json();
    })
    .then(data => {
        showAlert('研究任务已成功暂停', 'success');
        loadTasks(); // 重新加载任务列表
    })
    .catch(error => {
        console.error('Error pausing task:', error);
        showAlert('暂停任务失败: ' + error, 'danger');
    });
}

/**
 * 继续研究任务
 * @param {string} taskId - 任务ID
 */
function resumeResearchTask(taskId) {
    if (!confirm('确定要继续此研究任务吗？')) {
        return;
    }
    
    fetch(`/api/resume_research/${taskId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('服务器响应错误');
        }
        return response.json();
    })
    .then(data => {
        showAlert(`研究任务已成功继续，从阶段 ${data.resumed_from_phase} 恢复`, 'success');
        loadTasks(); // 重新加载任务列表
    })
    .catch(error => {
        console.error('Error resuming task:', error);
        showAlert('继续任务失败: ' + error, 'danger');
    });
}

/**
 * 查看任务结果
 * @param {string} taskId - 任务ID
 */
function viewTaskResult(taskId) {
    const resultContent = document.getElementById('result-content');
    const downloadPdfBtn = document.getElementById('download-pdf');
    
    if (!resultContent || !downloadPdfBtn) return;
    
    // 显示加载状态
    resultContent.innerHTML = '<div class="text-center p-5"><span class="spinner-border" role="status"></span><p class="mt-3">正在加载研究结果...</p></div>';
    
    // 显示结果模态框
    const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
    resultModal.show();
    
    // 获取任务结果
    fetch(`/api/research_result/${taskId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('获取结果失败');
            }
            return response.json();
        })
        .then(data => {
            // 使用marked.js渲染Markdown
            let reportHtml = marked.parse(data.report || '');
            
            // 添加图片
            let imagesHtml = '';
            if (data.images && data.images.length > 0) {
                imagesHtml = '<h3>生成的图像</h3><div class="row">';
                data.images.forEach(image => {
                    imagesHtml += `
                        <div class="col-md-6 col-lg-4 mb-3">
                            <div class="card">
                                <img src="${image.url}" class="card-img-top" alt="${image.name}">
                                <div class="card-body">
                                    <p class="card-text">${image.name}</p>
                                    <a href="${image.url}" class="btn btn-sm btn-outline-primary" download>下载</a>
                                </div>
                            </div>
                        </div>
                    `;
                });
                imagesHtml += '</div>';
            }
            
            // 设置PDF下载链接
            if (data.pdf_url) {
                downloadPdfBtn.href = data.pdf_url;
                downloadPdfBtn.classList.remove('d-none');
            } else {
                downloadPdfBtn.classList.add('d-none');
            }
            
            // 更新结果内容
            resultContent.innerHTML = `
                <div class="research-report mb-4">
                    <h2>研究报告</h2>
                    <div class="report-content">
                        ${reportHtml}
                    </div>
                </div>
                ${imagesHtml}
            `;
        })
        .catch(error => {
            console.error('Error viewing result:', error);
            resultContent.innerHTML = `
                <div class="alert alert-danger">
                    <h4>加载结果失败</h4>
                    <p>${error.message || '未知错误'}</p>
                </div>
            `;
        });
}

/**
 * 显示提示消息
 * @param {string} message - 消息内容
 * @param {string} type - 消息类型 (success, danger, warning, info)
 */
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // 添加到页面
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // 5秒后自动关闭
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    }
} 