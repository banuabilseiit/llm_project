import re
import pickle
import streamlit as st
import pymorphy3
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# ═══════════════════════════════════════════════════════════════════════
# Настройка страницы
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Классификатор обращений", page_icon="🤖")
st.title("🤖 Классификатор обращений студентов")
st.write(
    "Модель TF-IDF + LogisticRegression определяет категорию сообщения: "
    "**Жалоба**, **Благодарность** или **Вопрос**."
)

# ═══════════════════════════════════════════════════════════════════════
# Препроцессинг
# ═══════════════════════════════════════════════════════════════════════
morph   = pymorphy3.MorphAnalyzer()
stop_ru = set(stopwords.words("russian"))

def preprocess(text: str) -> str:
    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    tokens = [t for t in tokens if t not in stop_ru]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)

# ═══════════════════════════════════════════════════════════════════════
# Загрузка модели (один раз, кэшируется)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

LABEL_MAP = {0: "Жалоба", 1: "Благодарность", 2: "Вопрос"}
EMOJI_MAP = {0: "😠", 1: "🙏", 2: "❓"}

# ═══════════════════════════════════════════════════════════════════════
# Интерфейс
# ═══════════════════════════════════════════════════════════════════════
message = st.text_area(
    "Введите сообщение пользователя:",
    placeholder="Например: Почему мой доступ к урокам заблокирован?",
    height=100,
)

if st.button("Определить категорию", type="primary"):
    if not message.strip():
        st.warning("Введите текст сообщения.")
    else:
        processed = preprocess(message)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        category = LABEL_MAP.get(prediction, str(prediction))
        emoji = EMOJI_MAP.get(prediction, "")

        st.success(f"**Категория:** {category} {emoji}")

        st.write("**Вероятности по классам:**")
        for idx, p in enumerate(proba):
            st.write(f"  {LABEL_MAP.get(idx, idx)}: {p:.1%}")
            st.progress(float(p))

st.divider()
with st.expander("Примеры сообщений для теста"):
    st.write("""
    - Почему мой доступ к урокам заблокирован?
    - Спасибо, курс просто отличный, очень помог!
    - Как мне сбросить пароль от личного кабинета?
    - Это безобразие — третий день не работает видео!
    """)

st.caption("Домашнее задание: От классического NLP к современным LLM приложениям")
