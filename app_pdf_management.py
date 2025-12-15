import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime
import mysql.connector
import requests
import PyPDF2
import tiktoken
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="pdf_management", page_icon="📕", layout="wide")

# --- 2. 核心工具库 (基础设施) ---

# A. 初始化 API
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("❌ 未找到 API Key，请在 secrets.toml 中配置")
    st.stop()

client = openai.Client(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)


# B. 数据库连接
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["tidb"]["host"],
        port=st.secrets["tidb"]["port"],
        user=st.secrets["tidb"]["user"],
        password=st.secrets["tidb"]["password"],
        database=st.secrets["tidb"]["database"]
    )


# C. 数据库写入 (路由分发)
def save_to_db(table_name, data_dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if table_name == "paper_notes":
            # 处理列表转字符串
            tags_str = ",".join(data_dict.get('tags', []))
            file_path = data_dict.get('file_path', '')
            summary = data_dict.get('summary', '')  # 获取智能摘要

            sql = "INSERT INTO paper_notes (paper_name, question, answer, tags, file_path, summary, log_time) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (data_dict['paper_name'], data_dict['question'], data_dict['answer'], tags_str, file_path, summary,
                   current_time)
        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ TiDB 写入失败: {e}")
        return False


def load_from_db(table_name):
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM {table_name} ORDER BY log_time DESC"  # 默认倒序
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        # st.error(f"读取数据失败: {e}") # 生产环境可以注释掉以免干扰
        return pd.DataFrame()


# D. 飞书同步
def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    req = {
        "app_id": st.secrets["feishu"]["app_id"],
        "app_secret": st.secrets["feishu"]["app_secret"]
    }
    resp = requests.post(url, json=req).json()
    return resp.get("tenant_access_token")


def save_to_feishu(type_key, data):
    try:
        token = get_feishu_token()
        if not token: return False
        app_token = st.secrets["feishu"]["app_token"]
        if type_key == "paper":
            table_id = st.secrets["feishu"]["paper_table_id"]
            tags_str = ",".join(data.get('tags', []))
            fields = {
                "文献名": data['paper_name'],
                "问题": data['question'],
                "AI解读": data['answer'],
                "标签": tags_str,
                "精简摘要": data.get('summary', ''),
                "记录时间": int(datetime.now().timestamp() * 1000)
            }

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"fields": fields}
        resp = requests.post(url, headers=headers, json=payload).json()
        return resp.get("code") == 0
    except Exception as e:
        st.error(f"❌ 飞书同步失败: {e}")
        return False


# E. 本地文件管理
def save_uploaded_file(uploaded_file):
    library_dir = "paper_library"
    if not os.path.exists(library_dir):
        os.makedirs(library_dir)
    file_path = os.path.join(library_dir, uploaded_file.name)
    # 4. 【新增功能】检查文件是否已经存在
    if os.path.exists(file_path):
        # 如果存在，直接返回路径，并标记 is_new = False
        return file_path, False

    # 5. 如果不存在，才进行写入操作
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 返回路径，并标记 is_new = True (代表是新存的)
    return file_path, True



# F. Token 计数
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# --- 功能模块 文献阅读：
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for i, page in enumerate(pdf_reader.pages):
        c = page.extract_text()
        if c: text += f"\n\n--- [第 {i + 1} 页] ---\n\n{c}"
    return text


def render_med_reader():
    st.header("📄 AI 文献阅读助手 (Pro)")
    st.caption("RAG 阅读 | 标签管理 | 存算分离架构")
    if "init_done" not in st.session_state:
        st.session_state.chat_history = []
        default_tags = []
        # 从数据库捞标签
        df_history = load_from_db("paper_notes")
        db_tags = set()
        if not df_history.empty and 'tags' in df_history.columns:
            for tag_str in df_history['tags']:
                if tag_str:
                    for p in tag_str.split(","):
                        if p.strip(): db_tags.add(p.strip())

        all_tags = list(set(default_tags + list(db_tags)))
        all_tags.sort()
        st.session_state.all_tags = all_tags
        st.session_state.init_done = True

    # 1. 文件上传区
    with st.sidebar:
        st.markdown("### 📥 文献上传")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            # 自动保存到本地书架
            saved_path = save_uploaded_file(uploaded_file)
            st.session_state.current_file_path = saved_path

            # 清空旧会话
            if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
                st.session_state.chat_history = []
                st.session_state.last_file = uploaded_file.name
                st.toast("新文献已加载，记忆重置")

            # 提取文本
            paper_text = extract_text_from_pdf(uploaded_file)
            tokens = count_tokens(paper_text)
            st.success(f"已解析: {len(paper_text)} 字符")
            st.caption(f"Token 估算: {tokens}")
            if len(paper_text) > 2000:
                # 如果文章很长，显示头尾
                preview_content = paper_text[:1000] + "\n\n... (中间内容已省略) ...\n\n" + paper_text[-1000:]
            else:
                # 如果文章本身就不长，直接显示全部
                preview_content = paper_text
            with st.expander("点击展开查看文档预览"):
                st.markdown(preview_content)
    # 2. 聊天交互区
    if uploaded_file and 'paper_text' in locals():
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        query = st.chat_input("关于这篇论文，你想问什么？")
        if query:
            with st.chat_message("user"):
                st.write(query)
            st.session_state.chat_history.append({"role": "user", "content": query})

            # 构造带缓存的消息链
            messages = [
                           {"role": "system",
                            "content": f"""
                            你是一个严谨的医学科研助手。
                             1. 请基于我提供的【论文内容】回答问题。
                             2. **必须引用原文**：在回答的关键观点后，请标注出处，例如 (见第 3 页)。
                             3. 如果论文中没有相关信息，请直接回答“文中未提及”，不要编造。
                             4. 保持回答的逻辑性，使用 Markdown 格式（如列表、粗体）。
                            【论文全文】：
                            {paper_text}
                            """},
                       ]
            messages.extend(st.session_state.chat_history)
            with st.chat_message("assistant"):
                with st.spinner("AI 思考中..."):
                    try:
                        resp = client.chat.completions.create(
                            model="deepseek-chat", messages=messages, temperature=0.1
                        )
                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})

                        # 费用统计
                        if resp.usage:
                            prompt_tokens = resp.usage.prompt_tokens  # 提问消耗 (PDF + 问题)
                            completion_tokens = resp.usage.completion_tokens  # 回答消耗 (AI 写的字)
                            # 缓存命中的 Token 数量 (Cache Hit)
                            cached_tokens = resp.usage.prompt_cache_hit_tokens
                            # 实际扣费的 Token 数量 (Cache Miss)
                            miss_tokens = resp.usage.prompt_cache_miss_tokens
                            total = resp.usage.total_tokens

                            st.caption(f"""
                            💰 **DeepSeek 缓存统计**:
                            - 📥 阅读 (Input): `{prompt_tokens}` Tokens
                            - ✅ 命中缓存: `{cached_tokens}` Tokens 
                            - 🆕 新增读取: `{miss_tokens}` Tokens 
                            - 📤 思考 (Output): `{completion_tokens}` Tokens
                            - 💰 总计 (Total): `{total}` Tokens
                            """)
                    except Exception as e:
                        st.error(f"Error: {e}")

    #  3. 笔记保存区 (升级版：支持自定义标签)
        st.divider()
        if len(st.session_state.chat_history) > 0:
            with st.expander("💾 保存当前对话到笔记", expanded=True):
                with st.form("save_note"):
                    # 获取最近一次问答
                    last_q = st.session_state.chat_history[-2]['content']
                    last_a = st.session_state.chat_history[-1]['content']
                    p_name = st.session_state.get("last_file", "未知")

                    # 标签系统
                    sel_tags = st.multiselect("已有标签", options=st.session_state.all_tags)
                    new_tags = st.text_input("新增标签", placeholder="例如：罕见病, 基因编辑")

                    if st.form_submit_button("✅ 确认归档"):
                        # 合并标签
                        custom = [t.strip() for t in new_tags.split(",") if t.strip()]
                        final_tags = list(set(sel_tags + custom))

                        # 学习新标签
                        for t in custom:
                            if t not in st.session_state.all_tags:
                                st.session_state.all_tags.append(t)

                         # 生成 One-Liner 摘要
                        summary_text = ""
                        with st.spinner("正在生成精简摘要..."):
                            try:
                                sum_resp = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[{"role": "user",
                                                   "content": f"请为以下问答生成一个20字以内的核心结论摘要，不要标点：\n问：{last_q}\n答：{last_a}"}],
                                            temperature=0.1
                                        )
                                summary_text = sum_resp.choices[0].message.content.strip()
                            except:
                                summary_text = "摘要生成失败"
                                # 存库
                            data = {
                                "paper_name": p_name, "question": last_q, "answer": last_a,
                                "tags": final_tags, "summary": summary_text,
                                "file_path": st.session_state.get("current_file_path", "")
                                }

                            if save_to_db("paper_notes", data) and save_to_feishu("paper", data):
                                st.success(f"已归档！摘要: {summary_text}")

                            else:
                                 st.error("保存失败")

    # 4. 知识库浏览区 (分栏 + 交互 + 下载)
    st.header("📚 科研知识库")
    df = load_from_db("paper_notes")

    if not df.empty:
        # 数据清洗
        if 'tags' not in df.columns: df['tags'] = ""
        df['tags'] = df['tags'].fillna("")
        if 'summary' not in df.columns: df['summary'] = ""
        df['summary'] = df['summary'].fillna("无摘要")

        # 标签筛选器
        all_db_tags = set(t for s in df['tags'] for t in s.split(",") if t)
        with st.expander("🔍 筛选与导出"):
            col_f1, col_f2 = st.columns(2)
            filter_tags = col_f1.multiselect("按标签筛选", list(all_db_tags))

            # 导出 CSV
            csv = df.to_csv(index=False).encode('utf-8-sig')
            col_f2.download_button("📤 导出备份 (CSV)", csv, "medical_notes.csv", "text/csv")

        # 应用筛选
        if filter_tags:
            df = df[df['tags'].apply(lambda x: any(t in x.split(",") for t in filter_tags))]

        # 分栏布局
        col_list, col_detail = st.columns([2, 3])

        with col_list:
            st.caption(f"共 {len(df)} 条笔记")
            # 交互式表格
            event = st.dataframe(
                df,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "summary": st.column_config.TextColumn("📌 核心结论", width="large"),
                    "tags": st.column_config.TextColumn("标签", width="medium"),
                    "paper_name": st.column_config.Column(hidden=True),
                    "question": st.column_config.Column(hidden=True),
                    "answer": st.column_config.Column(hidden=True),
                    "file_path": st.column_config.Column(hidden=True)
                },
                use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun", height=600
            )

        with col_detail:
            if len(event.selection.rows) > 0:
                row = df.iloc[event.selection.rows[0]]
                with st.container(border=True):
                    # 详情卡片
                    st.markdown(f"### 📄 {row['paper_name']}")
                    if row['tags']:
                        # 渲染标签
                        tags_html = " ".join([f"`{t}`" for t in row['tags'].split(",") if t])
                        st.markdown(f"🏷️ {tags_html}")

                    st.divider()
                    st.info(f"📌 **摘要**: {row['summary']}")
                    st.markdown("#### ❓ 问题");
                    st.write(row['question'])
                    st.markdown("#### 🤖 解读");
                    st.markdown(row['answer'])

                    st.divider()
                    # 原始文件下载
                    if row['file_path'] and os.path.exists(row['file_path']):
                        with open(row['file_path'], "rb") as f:
                            st.download_button("📥 打开/下载原始 PDF", f, file_name=row['paper_name'])
                    else:
                        st.caption("⚠️ 原始文件未在本地找到 (仅显示云端笔记)")
            else:
                st.info("👈 请点击左侧列表查看详情")
# --- 5. 主程序入口 ---
def main():
    if __name__ == "__main__":
        main()