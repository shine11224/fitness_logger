import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime
import mysql.connector
import requests
import PyPDF2  # <--- 新引入的“显微镜”，用于读取 PDF
import tiktoken # 引入消耗的token计算

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="Dr. AI 个人助手", page_icon="👨‍⚕️", layout="wide")
# layout="wide" 让页面变宽，适合阅读文献

# --- 2. 核心工具类 (所有科室公用的设备) ---

# A. 连接 API
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("未找到 API Key，请配置 secrets.toml")
    st.stop()

client = openai.Client(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)


# B. 数据库连接工具
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["tidb"]["host"],
        port=st.secrets["tidb"]["port"],
        user=st.secrets["tidb"]["user"],
        password=st.secrets["tidb"]["password"],
        database=st.secrets["tidb"]["database"]
    )


# C. 飞书工具
def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"  # 注意：通常是 tenant_access_token
    req = {
        "app_id": st.secrets["feishu"]["app_id"],
         "app_secret": st.secrets["feishu"]["app_secret"]
        }
    resp = requests.post(url, json=req).json()
    return resp.get("tenant_access_token")
# ---3. 调用ai工具函数---
def get_food_info(user_input):
    system_prompt = """
    You are a nutritionist. Analyze user input and return JSON.
    Format requirements:
    {
        "food_name": "Food name in Chinese",
        "calories": integer (kcal),
        "protein": integer (g),
        "carbohydrate": integer (g),
        "fat": integer (g),
        "tips": "One short health advice in English"
    }
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        st.error(f"AI 连接超时或出错: {e}")
        return None


def get_exercise_info(user_input):
    system_prompt = """
    You are a fitness coach. Estimate calories burned based on user input.
    Return JSON format:
    {
        "exercise_name": "Exercise name in Chinese",
        "duration": "Duration string (e.g. '30 mins')",
        "calories_burned": integer (kcal, positive number),
        "tips": "Short recovery advice in English"
    }
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def count_tokens(text):
    """【新增】计算文本的 Token 数量"""
    # 使用 cl100k_base 编码器 (目前大多数先进模型通用的编码标准)
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

# ---4. 数据保存函数---
def save_to_db(table_name, data_dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if table_name == "diet_log":
            sql = "INSERT INTO diet_log (food_name, calories, protein, carbohydrate, fat, tips, log_time) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (data_dict['food_name'], data_dict['calories'], data_dict['protein'],
                   data_dict.get('carbohydrate', 0), data_dict.get('fat', 0),  # 使用 .get 防止 AI 没返回这些字段报错
                   data_dict['tips'], current_time)

        elif table_name == "exercise_log":
            sql = "INSERT INTO exercise_log (exercise_name, duration, calories_burned, tips, log_time) VALUES (%s, %s, %s, %s, %s)"
            val = (data_dict['exercise_name'], data_dict['duration'], data_dict['calories_burned'],
                   data_dict['tips'], current_time)

        cursor.execute(sql, val)
        conn.commit()
        cursor.close()
        conn.close()
        return True  # <--- 关键修复：必须返回 True
    except Exception as e:
        st.error(f"❌ TiDB 写入失败: {e}")
        return False
def save_to_feishu(type_key, data):
    try:
        token = get_feishu_token()
        if not token:
            st.error("飞书 Token 获取失败")
            return False

        app_token = st.secrets["feishu"]["app_token"]

        # 关键修复：统一转为小写比较，防止 Diet != diet
        if type_key.lower() == "diet":
            table_id = st.secrets["feishu"]["diet_table_id"]
            fields = {
                "food_name": data['food_name'],
                "calories": data['calories'],
                "protein": data['protein'],
                "carbohydrate": data.get('carbohydrate', 0),
                "fat": data.get('fat', 0),
                "tips": data['tips'],
                "log_time": int(datetime.now().timestamp() * 1000)  # 飞书日期通常接受毫秒时间戳
            }
        else:
            table_id = st.secrets["feishu"]["ex_table_id"]
            fields = {
                "exercise_name": data['exercise_name'],
                "duration": data['duration'],
                "calories_burned": data['calories_burned'],
                "tips": data['tips'],
                "log_time": int(datetime.now().timestamp() * 1000)
            }

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"fields": fields}

        resp = requests.post(url, headers=headers, json=payload).json()
        if resp.get("code") == 0:
            return True
        else:
            st.error(f"❌ 飞书报错: {resp}")
            return False
    except Exception as e:
        st.error(f"❌ 飞书连接失败: {e}")
        return False

# ---5. 数据读取函数---
def load_from_db(table_name):
    # 增加容错，防止读取失败导致页面崩溃
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return pd.DataFrame()

# --- 6. 功能模块 A：健康管理 (原来的代码打包) ---
def render_health_hub():
    st.header("🧬 AI 健康中枢")
    daily_goal = st.slider("每日热量目标 (kcal)", 1000, 3000, 1800)
    tab1, tab2, tab3 = st.tabs(["🍽️ 饮食记录", "🏃 运动打卡", "📊 数据看板"])
    with tab1:
        st.subheader("今天吃了什么？")
        food_input = st.text_input("输入食物...", key="food_input")
        if st.button("计算热量 (摄入)", key="btn_eat"):
            if not food_input:
                st.warning("请输入内容")
            else:
                with st.spinner('AI 正在计算卡路里...'):
                    result = get_food_info(food_input)
                    # 确保 result 不是 None 再继续
                    if result:
                        st.info(f"🇺🇸 Advice: {result['tips']}")

                        col1, col2 = st.columns(2)

                        # 1. 写 TiDB (使用正确的表名)
                        if save_to_db("diet_log", result):
                            col1.success(f"SQL 写入成功: {result['food_name']}")

                        # 2. 写 飞书 (使用小写 key)
                        if save_to_feishu("diet", result):
                            col2.success("飞书同步成功!")

    with tab2:
        st.subheader("今天练了什么？")
        ex_input = st.text_input("输入运动...", placeholder="例如：慢跑30分钟", key="ex_input")
        if st.button("计算消耗 (运动)", key="btn_move"):
            if not ex_input:
                st.warning("请输入内容")
            else:
                with st.spinner('AI 正在评估运动消耗...'):
                    result = get_exercise_info(ex_input)
                    if result:
                        st.info(f"💪 Coach: {result['tips']}")

                        col1, col2 = st.columns(2)

                        # 修正了参数传反的问题，删除了错误的 csv 调用
                        if save_to_db("exercise_log", result):
                            col1.success(f"SQL 写入成功! (-{result['calories_burned']} kcal)")

                        if save_to_feishu("exercise", result):
                            col2.success("飞书同步成功!")

    with tab3:
        st.subheader("📊 实时云端数据")
        # 加载数据
        df_diet = load_from_db("diet_log")
        df_ex = load_from_db("exercise_log")

        if not df_diet.empty:
            df_diet['log_time'] = pd.to_datetime(df_diet['log_time'])
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_cals = df_diet[df_diet['log_time'].dt.strftime("%Y-%m-%d") == today_str]['calories'].sum()
        else:
            today_cals = 0

        if not df_ex.empty:
            df_ex['log_time'] = pd.to_datetime(df_ex['log_time'])
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_burn = df_ex[df_ex['log_time'].dt.strftime("%Y-%m-%d") == today_str]['calories_burned'].sum()
        else:
            today_burn = 0

        col1, col2, col3 = st.columns(3)
        net_calories = today_cals - today_burn
        remaining = daily_goal - net_calories

        col1.metric("摄入 (In)", f"{today_cals}", delta="吃进去的")
        col2.metric("消耗 (Out)", f"{today_burn}", delta="-练掉的", delta_color="inverse")
        col3.metric("今日剩余额度", f"{remaining}", delta="还能吃多少",
                    delta_color="normal" if remaining > 0 else "inverse")

        st.divider()
        # 进度条防止报错 (分母不能为0，虽然 daily_goal 最小1000)
        progress = max(0.0, min(net_calories / daily_goal, 1.0))
        st.progress(progress, text=f"今日热量额度使用率: {int(progress * 100)}%")

        if remaining < 0:
            st.error("⚠️ 热量超标警告！")
        else:
            st.success("🟢 状态良好，继续保持！")

# --- 4. 功能模块 B：文献阅读 (新开发的科室) ---
# 【优化1】加上缓存装饰器：只要文件没变，就不需要重新解析 PDF
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """助手函数：把 PDF 文件变成字符串"""
    uploaded_file.seek(0)
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    # 遍历每一页读取文字
    for i, page in enumerate(pdf_reader.pages):
        page_content = page.extract_text()
        if page_content:
            # 【优化2】我们在每一页内容前加上 [第x页] 的标记
            # 这样 AI 就能知道这段话来自哪里
            text += f"\n\n--- [第 {i + 1} 页] ---\n\n"
            text += page_content

    return text

def render_med_reader():
    st.header("📄 AI 文献阅读助手")
    st.caption("上传医学论文(PDF)，让 AI 帮你快速提取核心观点")
    # 1. 添加上下文记忆
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # 2. 上传文件
    uploaded_file = st.file_uploader("请上传 PDF 文件", type="pdf")
    if uploaded_file:
        # 解析文件 (有缓存，第二次会很快)
        with st.spinner("正在读取论文内容..."):
            paper_text = extract_text_from_pdf(uploaded_file)
            # --- 【新增】计算并显示 Token ---
            tokens = count_tokens(paper_text)
            char_count = len(paper_text)
            # 显示字符数统计
            st.success("读取成功！")
            col1, col2 = st.columns(2)
            col1.metric("字符数 (Characters)", f"{char_count:,}")  # 加逗号，方便看千分位
            col2.metric("预估 Token (AI 消耗)", f"{tokens:,}", help="DeepSeek 最大支持 64k Context，请注意不要超标")
            # 警告：如果字数真的超级多（比如超过20万），才需要担心
            if len(paper_text) > 100000:
                st.warning("⚠️ 文献非常长，AI 处理可能会稍慢，请耐心等待。")
            if len(paper_text) > 2000:
                # 如果文章很长，显示头尾
                preview_content = paper_text[:1000] + "\n\n... (中间内容已省略) ...\n\n" + paper_text[-1000:]
            else:
                # 如果文章本身就不长，直接显示全部
                preview_content = paper_text
            with st.expander("点击展开查看文档预览"):
                st.markdown(preview_content)
        # 3. 如果换了新文件，清空以前的聊天记录
        # 我们用文件名来判断用户是否换了论文
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.chat_history = []  # 清空记忆
            st.session_state.last_file = uploaded_file.name  # 更新文件名记录
            st.toast("检测到新文件，聊天记录已重置")
        # 4. 显示历史聊天记录 (回放记忆)
        # 每次页面刷新，都要把之前的聊天气泡重新画一遍
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # 5. 问答环节
        query = st.chat_input("关于这篇论文，你想问什么？(例如：这篇研究的结论是什么？)")
        if query:
            # A. 立刻把用户的问题显示出来，并存入记忆
            with st.chat_message("user"):
                st.write(query)
            st.session_state.chat_history.append({"role": "user", "content": query})

            # B. 构造发给 AI 的完整消息列表
            # 关键点：System Prompt (含论文) + History (旧记录) + Query (新问题)

            # (1) 系统级指令：永远放在第一条，包含论文全文
            # 💡 DeepSeek 会自动缓存这一条，因为它是固定不变的“前缀”
            messages_payload = [
                {
                    "role": "system",
                    "content": f"""
                    你是一个严谨的医学科研助手。
                    1. 请基于我提供的【论文内容】回答问题。
                    2. **必须引用原文**：在回答的关键观点后，请标注出处，例如 (见第 3 页)。
                    3. 如果论文中没有相关信息，请直接回答“文中未提及”，不要编造。
                    4. 保持回答的逻辑性，使用 Markdown 格式（如列表、粗体）。
                    【论文全文】：
                    {paper_text}"""
                }
            ]

            # (2) 追加历史记录 (让 AI 知道上下文)
            # 我们把 session_state 里的记录加进去
            # *注意：为了省钱，你可以只取最近的 4-6 轮对话，这里演示取全部
            messages_payload.extend(st.session_state.chat_history)

            # C. 调用 API
            with st.chat_message("assistant"):
                with st.spinner("AI 正在思考..."):
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages_payload, # 发送完整对话链
                            temperature=0.1
                        )

                        answer = response.choices[0].message.content
                        st.markdown(answer)

                        # D. 把 AI 的回答也存入记忆
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        # E. 费用统计 (看看缓存有没有生效)
                        if response.usage:
                            prompt_tokens = response.usage.prompt_tokens  # 提问消耗 (PDF + 问题)
                            completion_tokens = response.usage.completion_tokens  # 回答消耗 (AI 写的字)
                            # 缓存命中的 Token 数量 (Cache Hit)
                            cached_tokens = response.usage.prompt_cache_hit_tokens
                            # 实际扣费的 Token 数量 (Cache Miss)
                            miss_tokens = response.usage.prompt_cache_miss_tokens
                            total = response.usage.total_tokens

                            st.caption(f"""
                            💰 **DeepSeek 缓存统计**:
                            - 📥 阅读 (Input): `{prompt_tokens}` Tokens
                            - ✅ 命中缓存: `{cached_tokens}` Tokens 
                            - 🆕 新增读取: `{miss_tokens}` Tokens 
                            - 📤 思考 (Output): `{completion_tokens}` Tokens
                            - 💰 总计 (Total): `{total}` Tokens
                            """)

                    except Exception as e:
                        st.error(f"出错: {e}")


# --- 5. 主程序入口 (总控室) ---
def main():
    # 侧边栏导航
    with st.sidebar:
        st.title("🏥 个人医学中心")
        choice = st.radio(
            "选择功能科室",
            ["健康管理部", "文献阅读部"],
            captions=["记录热量，管理健康", "上传论文，辅助科研"]
        )
        st.divider()
        st.caption("Dr. AI v2.0")

    # 根据选择渲染不同页面
    if choice == "健康管理部":
        # 为了让你的旧代码能跑，这里需要把你原来的逻辑完整放进去
        # 由于篇幅限制，建议你把原来的代码封装在 render_health_hub() 里
        # 或者直接在这里写： if choice == ...: (粘贴你原来的大部分代码)
        render_health_hub()  # 调用函数

    elif choice == "文献阅读部":
        render_med_reader()


if __name__ == "__main__":
    main()