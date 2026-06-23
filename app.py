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
st.set_page_config(page_title="Классификатор обращений", page_icon="🤖", layout="centered")
st.title("🤖 Классификатор обращений студентов")
st.write(
    "Сравнение двух подходов к классификации текста: "
    "**классический NLP (TF-IDF)** и **современный LLM (Hugging Face)**.\n\n"
    "Категории: **Жалоба**, **Благодарность**, **Вопрос**."
)

LABEL_MAP = {0: "Жалоба", 1: "Благодарность", 2: "Вопрос"}
EMOJI_MAP = {0: "😠", 1: "🙏", 2: "❓"}

# ═══════════════════════════════════════════════════════════════════════
# Препроцессинг (для TF-IDF)
# ═══════════════════════════════════════════════════════════════════════
morph   = pymorphy3.MorphAnalyzer()
stop_ru = set(stopwords.words("russian"))

def preprocess(text: str) -> str:
    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    tokens = [t for t in tokens if t not in stop_ru]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)

# ═══════════════════════════════════════════════════════════════════════
# Загрузка TF-IDF модели (кэшируется)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_tfidf_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

def predict_tfidf(message: str):
    vectorizer, model = load_tfidf_model()
    processed = preprocess(message)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    return prediction, proba

# ═══════════════════════════════════════════════════════════════════════
# LLM-цепочка через Hugging Face (кэшируется)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_llm_chain():
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    template = """Ты — ассистент технической поддержки образовательной платформы.
Определи категорию следующего сообщения пользователя.

Категории (выбери ровно одну):
- Жалоба
- Благодарность
- Вопрос

Примеры:
Сообщение: "Спасибо за отличный курс!"
Категория: Благодарность

Сообщение: "Как мне записаться на следующий модуль?"
Категория: Вопрос

Сообщение: "Видео не загружается уже третий день!"
Категория: Жалоба

Теперь определи категорию этого сообщения:
Сообщение: "{message}"

Ответь только одним словом — названием категории, без пояснений."""

    prompt = PromptTemplate(input_variables=["message"], template=template)

    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        max_new_tokens=20,
        temperature=0.01,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return prompt | llm | StrOutputParser()

def llm_text_to_label(category_text: str):
    text = category_text.strip().lower().rstrip(".!,")
    if "жалоб" in text:
        return 0
    if "благодар" in text:
        return 1
    if "вопрос" in text:
        return 2
    return None

def predict_llm(message: str):
    chain = load_llm_chain()
    result = chain.invoke({"message": message})
    label = llm_text_to_label(result)
    return label, result.strip()

# ═══════════════════════════════════════════════════════════════════════
# Интерфейс
# ═══════════════════════════════════════════════════════════════════════
message = st.text_area(
    "Введите сообщение пользователя:",
    placeholder="Например: Почему мой доступ к урокам заблокирован?",
    height=100,
)

mode = st.radio(
    "Что показать:",
    ["Только TF-IDF", "Только LLM (Hugging Face)", "Сравнить обе модели"],
    horizontal=True,
)

run_tfidf = mode in ("Только TF-IDF", "Сравнить обе модели")
run_llm   = mode in ("Только LLM (Hugging Face)", "Сравнить обе модели")

if st.button("Определить категорию", type="primary"):
    if not message.strip():
        st.warning("Введите текст сообщения.")
    else:
        columns = st.columns(2) if mode == "Сравнить обе модели" else [st.container()]

        if run_tfidf:
            col = columns[0]
            with col:
                st.subheader("📊 TF-IDF + LogReg")
                with st.spinner("Анализирую..."):
                    prediction, proba = predict_tfidf(message)
                category = LABEL_MAP.get(prediction, str(prediction))
                emoji = EMOJI_MAP.get(prediction, "")
                st.success(f"**{category}**")
                st.markdown(
                    f"<div style='text-align: center; font-size: 90px;'>{emoji}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Уверенность: "
                    + ", ".join(f"{LABEL_MAP[i]} {p:.0%}" for i, p in enumerate(proba))
                )

        if run_llm:
            col = columns[1] if mode == "Сравнить обе модели" else columns[0]
            with col:
                st.subheader("🧠 LLM (Llama-3.3-70B)")
                with st.spinner("Анализирую..."):
                    try:
                        prediction, raw_text = predict_llm(message)
                    except Exception as e:
                        st.error(f"Ошибка подключения к Hugging Face: {e}")
                        prediction, raw_text = None, None

                if prediction is not None:
                    category = LABEL_MAP.get(prediction, str(prediction))
                    emoji = EMOJI_MAP.get(prediction, "")
                    st.success(f"**{category}**")
                    st.markdown(
                        f"<div style='text-align: center; font-size: 90px;'>{emoji}</div>",
                        unsafe_allow_html=True,
                    )
                elif raw_text is not None:
                    st.warning(f"Неожиданный ответ модели: '{raw_text}'")

st.divider()
with st.expander("Примеры сообщений для теста"):
    st.write("""
    - Почему мой доступ к урокам заблокирован?
    - Спасибо, курс просто отличный, очень помог!
    - Как мне сбросить пароль от личного кабинета?
    - Это безобразие — третий день не работает видео!
    """)

st.caption("Домашнее задание: От классического NLP к современным LLM приложениям")
