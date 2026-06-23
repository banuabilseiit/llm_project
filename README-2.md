# 🤖 Классификатор обращений студентов (TF-IDF + LogisticRegression)

Веб-приложение на Streamlit, определяющее категорию сообщения: **Жалоба**, **Благодарность** или **Вопрос**.

## Файлы в репозитории

| Файл | Назначение |
|---|---|
| `app.py` | Streamlit-приложение (точка входа) |
| `model.pkl` | Обученная модель LogisticRegression |
| `vectorizer.pkl` | Обученный TfidfVectorizer |
| `requirements.txt` | Зависимости для Streamlit Cloud |
| `train_and_save_model.py` | Скрипт переобучения модели (опционально) |
| `dataset.xlsx` | Датасет (100 фраз, 3 категории) |

---

## Как выложить на GitHub

### 1. Создай новый репозиторий
На github.com → **New repository** → дай имя, например `nlp-classifier-app` → Create.

### 2. Загрузи файлы

**Вариант А — через веб-интерфейс GitHub (без терминала):**
1. Открой свой репозиторий → **Add file** → **Upload files**
2. Перетащи все файлы: `app.py`, `model.pkl`, `vectorizer.pkl`, `requirements.txt`
3. Внизу нажми **Commit changes**

**Вариант Б — через терминал/git (если установлен git локально):**
```bash
git init
git add app.py model.pkl vectorizer.pkl requirements.txt
git commit -m "Add NLP classifier app"
git branch -M main
git remote add origin https://github.com/ТВОЙ_ЛОГИН/nlp-classifier-app.git
git push -u origin main
```

---

## Как задеплоить на Streamlit Community Cloud

1. Перейди на **https://share.streamlit.io**
2. Войди через свой GitHub-аккаунт (Sign in with GitHub)
3. Нажми **New app**
4. Выбери:
   - **Repository:** `ТВОЙ_ЛОГИН/nlp-classifier-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Нажми **Deploy**

Через 1-2 минуты приложение будет доступно по ссылке вида:
```
https://ТВОЙ_ЛОГИН-nlp-classifier-app.streamlit.app
```

---

## Важно: бинарные файлы .pkl на GitHub

`model.pkl` и `vectorizer.pkl` — это бинарные файлы небольшого размера (обычно десятки КБ для такого датасета), GitHub их прекрасно принимает без проблем. Если файлы вдруг окажутся больше 25 МБ — GitHub откажет через веб-интерфейс, тогда нужен Git LFS (но для этого проекта размер точно не будет проблемой).

## Если нужно переобучить модель на новых данных

```bash
pip install pymorphy3 nltk scikit-learn openpyxl pandas
python train_and_save_model.py
```
Это создаст новые `model.pkl` и `vectorizer.pkl` — их нужно закоммитить и запушить в репозиторий заново, после чего Streamlit Cloud автоматически передеплоит приложение.
