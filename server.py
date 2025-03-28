import vk_api
import pandas as pd
import joblib
import requests
import re
import shap
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Загружаем модель и SHAP
model = joblib.load("itog.pkl")
explainer = shap.Explainer(model)

# Получение года регистрации через первый пост на стене
def get_registration_year(user_id, vk):
    try:
        response = vk.wall.get(owner_id=user_id, count=1)

        if response["count"] == 0:
            return 2018  # Если у пользователя нет постов, ставим дефолтный год

        total_posts = response["count"]
        first_post_response = vk.wall.get(owner_id=user_id, count=1, offset=total_posts - 1)

        if not first_post_response["items"]:
            return 0  # Если нет записей, возвращаем 0

        first_post = first_post_response["items"][0]
        first_post_date = datetime.fromtimestamp(first_post["date"])
        return first_post_date.year

    except Exception:
        return 0

# Получение данных пользователя из VK API
def get_user_info(vk, user_id):
    try:
        user_info = vk.users.get(user_ids=user_id,
                                 fields="photo_max, first_name, last_name, followers_count, relation, personal, relatives, occupation, bdate, city, last_seen")[0]

        user_data = {
            "user_id": user_id,
            "first_name": user_info.get("first_name", "Неизвестно"),
            "last_name": user_info.get("last_name", "Неизвестно"),
            "photo": user_info.get("photo_max", ""),
            "year": get_registration_year(user_id, vk),
            "followers_count": user_info.get("followers_count", 0),
            "has_personal": int("personal" in user_info),
            "has_relatives": int("relatives" in user_info),
            "has_occupation": int("occupation" in user_info),
            "join_year": int(user_info.get("bdate", "0").split(".")[-1]) if "bdate" in user_info else 0,
            "relation": user_info.get("relation", 0),
            "city": user_info.get("city", {}).get("id", "0"),
            "friends_count": vk.friends.get(user_id=user_id, count=1)["count"],
            "photo_count": vk.photos.getAll(owner_id=user_id, count=1)["count"],
            "groups_count": vk.groups.get(user_id=user_id, extended=1)["count"],
        }

        posts = vk.wall.get(owner_id=user_id, count=100)
        user_data["status_update_frequency"] = len(posts["items"])
        status_text = vk.status.get(user_id=user_id).get('text', '')
        user_data["has_numbers"] = 1 if re.search(r"\d", status_text) else 0
        user_data["status_length"] = len(status_text)
        user_data["hashtags_count"] = sum(post["text"].count("#") for post in posts["items"])

        return user_data
    except vk_api.exceptions.ApiError as e:
        print(f"Ошибка при получении данных: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

# Flask маршрут для анализа пользователя
@app.route('/analyze', methods=['POST'])
def analyze_user():
    try:
        data = request.json
        user_id = data.get("user_id")

        if not user_id:
            return jsonify({"error": "Не передан user_id"}), 400

        print(f"🔍 Получен запрос на анализ пользователя: {user_id}")

        # Авторизация в VK API
        token = "vk1.a.0vXYjEl3M19azV0gpw05FI6Wo0sAsWeADDtaAtqo6tUux5uxALpoqTeX6-3nBIQcGgiG0IIy4s3r3IuY1S_QSQb8YxD4fTgpooucBvqHPKhCZjF-gLPHsRBOLCEQLDQrKinZ60J9ZpViGOVrcpA334jDaR87rr0kYphh97UqLAfc1NSespdFrsa8JPoufdjZYgsV4HyvyrGCAUkeqrXdTA"
        vk_session = vk_api.VkApi(token=token)
        vk = vk_session.get_api()

        user_data = get_user_info(vk, user_id)
        if not user_data:
            return jsonify({"error": "Не удалось получить данные пользователя"}), 500

        # Подготовка данных для предсказания
        df_user = pd.DataFrame([user_data])
        df_user = df_user[model.feature_names_in_]

        prediction = model.predict(df_user)[0]
        probabilities = model.predict_proba(df_user)[0]

        shap_values = explainer(df_user)

        feature_importance = np.abs(shap_values.values[0])
        top_features = np.argsort(-feature_importance)[:3]

        suspicious_criteria = []
        for i in top_features:
            feature_name = df_user.columns[i]
            feature_value = df_user.iloc[0, i]
            suspicious_criteria.append(f"{feature_name}: {feature_value}")

        result = "Real" if prediction == 1 else "Fake"
        fake_prob = round(probabilities[0] * 100, 2)
        real_prob = round(probabilities[1] * 100, 2)

        response = {
            "user_id": user_id,
            "first_name": user_data["first_name"],
            "last_name": user_data["last_name"],
            "photo": user_data["photo"],
            "result": result,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "suspicious_criteria": suspicious_criteria
        }

        print(f"📊 Анализ завершён: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
