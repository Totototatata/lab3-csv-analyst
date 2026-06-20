import streamlit as st
import pandas as pd
import anthropic
import re
import sys
import io
import traceback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

matplotlib.use("Agg")

st.set_page_config(page_title="CSV Аналитик", page_icon="📊", layout="wide")
st.title("📊 CSV Аналитик")
st.markdown("Загрузи любой датасет — ИИ-агент сам напишет код для анализа и выдаст отчёт.")

with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("Anthropic API ключ", type="password")
    st.markdown("---")
    st.markdown("**Как пользоваться:**")
    st.markdown("1. Вставь API ключ")
    st.markdown("2. Загрузи CSV файл")
    st.markdown("3. Напиши на что обратить внимание")
    st.markdown("4. Нажми «Анализировать»")
    st.markdown("---")
    st.markdown("**Как работает агент:**")
    st.markdown("Claude пишет Python-код → код запускается → Claude видит результат → делает выводы")


uploaded_file = st.file_uploader("Загрузи CSV файл", type=["csv"])

user_instruction = st.text_area(
    "На что обратить внимание? (необязательно)",
    placeholder="Например: меня интересует динамика по годам и топ-категории",
    height=100
)


def is_safe(text):
    patterns = [
        r"ignore\s+(previous|all|above)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now", r"act\s+as",
        r"system\s*:", r"<\s*system\s*>",
        r"jailbreak", r"DAN",
        r"новые\s+инструкции",
        r"забудь\s+(всё|все|предыдущ)",
        r"игнорируй\s+(всё|предыдущ)",
        r"теперь\s+ты", r"притворись",
    ]
    return not any(re.search(p, text, re.IGNORECASE) for p in patterns)


def run_code(code: str, df: pd.DataFrame):
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    charts = []
    error = None
    try:
        env = {
            "df": df.copy(),
            "pd": pd, "plt": plt, "np": np,
            "Counter": Counter, "re": re,
            "__builtins__": __builtins__,
        }
        exec(code, env)
        for n in plt.get_fignums():
            b = io.BytesIO()
            plt.figure(n).savefig(b, format="png", dpi=100, bbox_inches="tight")
            b.seek(0)
            charts.append(b)
        plt.close("all")
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
    return buf.getvalue(), charts, error


def analyze(api_key, df, instruction):
    client = anthropic.Anthropic(api_key=api_key, base_url="https://apinet.cloud")

    tool = {
        "name": "python",
        "description": (
            "Запускает Python-код для анализа датасета. "
            "Датасет доступен в переменной df (pandas DataFrame). "
            "Используй print() для вывода. Для графиков используй plt. "
            "Доступны: pd, plt, np, Counter, re. Не пиши import."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"]
        }
    }

    hint = f"\nОбрати особое внимание: {instruction.strip()}" if instruction.strip() else ""

    system = (
        "Ты опытный аналитик данных. Датасет загружен в переменную df.\n"
        "Доступные библиотеки: pd, plt, np, Counter, re. НЕ пиши import.\n\n"
        "Выполни ровно 3 вызова инструмента python:\n"
        "1. Изучи структуру: df.shape, dtypes, isnull().sum(), value_counts для категориальных\n"
        "2. Посчитай статистику: describe(), топ значения, тренды\n"
        "3. Построй графики: один fig с 2-4 субплотами\n\n"
        "После трёх вызовов — напиши финальный отчёт текстом (без кода)."
    )

    messages = [{
        "role": "user",
        "content": (
            f"Проанализируй датасет: {df.shape[0]} строк, {df.shape[1]} столбцов. "
            f"Столбцы: {', '.join(df.columns)}.{hint}\n\n"
            "Выполни ровно 3 вызова инструмента python, затем напиши отчёт."
        )
    }]

    logs = []
    charts = []
    code_calls = 0

    for _ in range(5):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=[tool],
            messages=messages
        )

        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]

        if texts:
            logs.append(("text", "\n".join(texts)))

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            code = call.input.get("code", "")
            logs.append(("code", code))
            code_calls += 1

            out, new_charts, err = run_code(code, df)
            charts.extend(new_charts)

            if err:
                result_text = f"ОШИБКА: {err}"
                logs.append(("error", err))
            else:
                result_text = out or "Выполнено успешно"
                if new_charts:
                    result_text += f" [графиков: {len(new_charts)}]"
                logs.append(("output", result_text))

            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result_text
            })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": results})

        if code_calls >= 3:
            messages.append({
                "role": "user",
                "content": (
                    "Анализ завершён. Теперь напиши финальный отчёт на русском языке. "
                    "Только текст, без кода. Структура:\n"
                    "## 1. Описание датасета\n"
                    "## 2. Качество данных\n"
                    "## 3. Ключевые закономерности и тренды\n"
                    "## 4. Аномалии\n"
                    "## 5. Практические выводы и рекомендации"
                )
            })
            final = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system,
                messages=messages
            )
            for b in final.content:
                if b.type == "text" and b.text.strip():
                    logs.append(("text", b.text))
            break

    return logs, charts


if uploaded_file and st.button("🔍 Анализировать", type="primary"):

    if not api_key:
        st.error("Вставь API ключ в боковой панели.")
        st.stop()

    if user_instruction and not is_safe(user_instruction):
        st.error("В инструкции обнаружен недопустимый текст.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        st.stop()

    st.success(f"Файл загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")

    with st.expander("👀 Превью данных"):
        st.dataframe(df.head())

    st.subheader("🤖 Агент анализирует...")
    st.info("Claude сам решает какой код написать, запускает его и делает выводы")

    try:
        with st.spinner("Идёт анализ..."):
            logs, charts = analyze(api_key, df, user_instruction)

        with st.expander("🔍 Лог работы агента", expanded=False):
            for kind, content in logs:
                if kind == "code":
                    st.markdown("**Агент написал код:**")
                    st.code(content, language="python")
                elif kind == "output":
                    st.markdown("**Результат:**")
                    st.text(content)
                elif kind == "error":
                    st.error(content)
                elif kind == "text":
                    st.markdown(content)

        if charts:
            st.subheader("📈 Графики")
            cols = st.columns(2)
            for i, c in enumerate(charts):
                cols[i % 2].image(c, use_column_width=True)

        report = next(
            (c for k, c in reversed(logs) if k == "text" and len(c) > 300),
            None
        )
        if report:
            st.subheader("📋 Аналитический отчёт")
            st.markdown(report)
        else:
            st.warning("Отчёт не сформирован — посмотри лог выше.")

    except Exception as e:
        st.error(f"Ошибка: {e}")

elif not uploaded_file:
    st.info("👆 Загрузи CSV файл чтобы начать.")
