import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime
import mysql.connector
import requests

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="AI Health Hub", page_icon="🧬", layout="centered")
st.title("🧬 AI 健康中枢 (双核版)")
st.caption("数据流向：TiDB Cloud (SQL) + 飞书多维表格 (可视化)")

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 设置")
    daily_goal = st.slider("每日热量目标 (kcal)", 1000, 3000, 1800)
    st.write("Keep fighting! 💪")

# --- 3. 连接 API ---
# 增加容错：防止没配 Key 报错
if "DEEPSEEK_API_KEY" not in st.secrets:
    st.error("未找到 API Key，请在 secrets.toml 中配置")
    st.stop()

client = openai.Client(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1"
)


# --- 4. 数据库模块 ---
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["tidb"]["host"],
        port=st.secrets["tidb"]["port"],
        user=st.secrets["tidb"]["user"],
        password=st.secrets["tidb"]["password"],
        database=st.secrets["tidb"]["database"]
    )


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


# --- 5. 飞书模块 ---
def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"  # 注意：通常是 tenant_access_token
    req = {
        "app_id": st.secrets["feishu"]["app_id"],
        "app_secret": st.secrets["feishu"]["app_secret"]
    }
    resp = requests.post(url, json=req).json()
    return resp.get("tenant_access_token")


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


# --- 6. AI 函数 ---
def get_food_info(user_input):
    # 提示词保持不变
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


# --- 7. 页面交互 ---
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