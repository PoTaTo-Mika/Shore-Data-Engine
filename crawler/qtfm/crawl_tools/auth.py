import requests
import logging
import time

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 蜻蜓FM的认证API端点
AUTH_URL = "https://user.qtfm.cn/u2/api/v4/auth"

def refresh_token(refresh_token: str, qingting_id: str) -> dict:
    logging.info("正在尝试刷新 access_token...")
    # 构造请求体
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "qingting_id": qingting_id,
        "device_id": "web"  # 与你之前抓包的设备保持一致
    }
    try:
        response = requests.post(AUTH_URL, json=payload)
        # 如果请求失败 (例如 4xx or 5xx), 抛出异常
        response.raise_for_status()
        data = response.json()
        # 检查业务逻辑错误
        if data.get("errorno") != 0:
            logging.error(f"刷新token时返回业务错误: {data.get('errormsg', '未知错误')}")
            return {}

        token_data = data.get('data', {})
        if 'access_token' in token_data:
            logging.info(f"成功刷新 access_token！新的 token 将在 {token_data.get('expires_in')} 秒后过期。")
            return token_data
        else:
            logging.error("刷新请求成功，但响应中未找到'data'或'access_token'字段。")
            return {}

    except requests.exceptions.RequestException as e:
        logging.error(f"刷新 access_token 时发生网络错误: {e}")
        return {}
    except Exception as e:
        logging.error(f"处理刷新 token 响应时发生未知错误: {e}")
        return {}