import io
import re
from typing import Dict, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="簡易BIツール", layout="wide")
st.title("📊 簡易BIツール")
st.caption(
    "CSVをアップロードし、列の型変換やグラフ描画（棒・折れ線・散布図）を行う簡易的なBIツールです。"
)

# -----------------------------
# Sidebar: CSV upload
# -----------------------------
st.sidebar.header("① CSVファイルのアップロード")
uploaded = st.sidebar.file_uploader(
    "CSVファイルをアップロードしてください（1行目が列名である必要があります）",
    type=["csv"],
    accept_multiple_files=False,
    help="UTF-8 または Shift-JIS（cp932）形式を推奨します。",
)

# サンプルデータ
sample_text = (
    "date,category,value1,value2\n"
    "2025-01-01,A,10,1.0\n"
    "2025-01-02,A,12,1.2\n"
    "2025-01-03,B,7,0.7\n"
    "2025-01-04,B,15,1.5\n"
    "2025-01-05,C,3,0.3\n"
)

use_sample = st.sidebar.toggle("サンプルデータを使用する", value=False)

encoding_choice = st.sidebar.selectbox(
    "文字コード（Shift-JISの場合はcp932を選択）",
    ["utf-8", "cp932", "utf-8-sig", "ISO-8859-1"],
    index=0,
)

st.sidebar.header("② 設定")
sep_choice = st.sidebar.selectbox("区切り文字", [",", ";", "\t", "|"], index=0)
na_values_text = st.sidebar.text_input(
    "欠損値とみなす文字列（カンマ区切り）",
    value="NA,NaN,",
)
na_values = [x for x in [s.strip() for s in na_values_text.split(",")] if x]


@st.cache_data(show_spinner=False)
def load_csv(file_or_str, encoding: str, sep: str, na_values: Optional[list]):
    if isinstance(file_or_str, io.BytesIO):
        return pd.read_csv(file_or_str, encoding=encoding, sep=sep, na_values=na_values)
    else:
        return pd.read_csv(
            io.StringIO(file_or_str), encoding=encoding, sep=sep, na_values=na_values
        )


# CSV読込
df = None
if uploaded and not use_sample:
    try:
        df = load_csv(uploaded, encoding_choice, sep_choice, na_values)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.stop()
elif use_sample:
    df = load_csv(sample_text, encoding_choice, sep_choice, na_values)
else:
    st.info(
        "👈 左のサイドバーからCSVをアップロード、またはサンプルデータを使用してください。"
    )
    st.stop()

if any(re.match(r"^Unnamed: \\d+$", str(c)) for c in df.columns):
    st.warning(
        "ヘッダー行（1行目）が列名として認識されていない可能性があります。CSVの構造を確認してください。"
    )

st.subheader("データプレビュー")
st.dataframe(df.head(200), use_container_width=True)

# -----------------------------
# 型変換
# -----------------------------
st.subheader("列の型変換")

TYPE_OPTIONS = [
    "変更しない",
    "文字列（string）",
    "整数（int）",
    "小数（float）",
    "真偽値（boolean）",
    "日付（datetime）",
    "カテゴリ（category）",
]


def _to_bool_series(s: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
    }
    result = s.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError("真偽値として解釈できない値が含まれています。")
    return result.astype("boolean")


def convert_column(df: pd.DataFrame, plan: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, target in plan.items():
        if target == "Auto":
            continue
        try:
            if target == "文字列（string）":
                out[col] = out[col].astype("string")
            elif target == "整数（int）":
                if not pd.to_numeric(out[col], errors="coerce").notna().all():
                    raise ValueError(f"列 '{col}' に数値以外の値が含まれています。")
                out[col] = out[col].astype(int)
            elif target == "小数（float）":
                if not pd.to_numeric(out[col], errors="coerce").notna().all():
                    raise ValueError(f"列 '{col}' に数値以外の値が含まれています。")
                out[col] = out[col].astype(float)
            elif target == "真偽値（boolean）":
                out[col] = _to_bool_series(out[col])
            elif target == "日付（datetime）":
                converted = pd.to_datetime(
                    out[col], errors="coerce", infer_datetime_format=True
                )
                if converted.isna().any():
                    raise ValueError(
                        f"列 '{col}' に日付として無効な値が含まれています。"
                    )
                out[col] = converted
            elif target == "カテゴリ（category）":
                out[col] = out[col].astype("category")
        except Exception as e:
            st.error(f"型変換エラー: 列 '{col}' → {target}: {e}")
            st.stop()
    return out


with st.form("type_form"):
    cols = st.columns([1, 1])
    plan: Dict[str, str] = {}
    for i, c in enumerate(df.columns):
        with cols[i % 2]:
            current = str(df[c].dtype)
            choice = st.selectbox(
                f"{c}（現在: {current}）", TYPE_OPTIONS, key=f"dtype_{c}"
            )
            plan[c] = choice
    apply_types = st.form_submit_button("型変換を実行")

if apply_types:
    df = convert_column(df, plan)
    st.success("型変換が完了しました。")

# -----------------------------
# フィルタ
# -----------------------------
with st.expander("オプション: 簡易フィルタ"):
    filter_cols = st.multiselect("フィルタする列を選択", df.columns, [])
    filtered = df.copy()
    for c in filter_cols:
        if pd.api.types.is_numeric_dtype(filtered[c]):
            rng = st.slider(
                f"{c} の範囲",
                float(filtered[c].min()),
                float(filtered[c].max()),
                (float(filtered[c].min()), float(filtered[c].max())),
            )
            filtered = filtered[(filtered[c] >= rng[0]) & (filtered[c] <= rng[1])]
        elif pd.api.types.is_datetime64_any_dtype(filtered[c]):
            min_dt, max_dt = filtered[c].min(), filtered[c].max()
            dr = st.date_input(f"{c} の期間", value=(min_dt, max_dt))
            if isinstance(dr, tuple) and len(dr) == 2:
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                filtered = filtered[(filtered[c] >= start) & (filtered[c] <= end)]
        else:
            vals = sorted([str(x) for x in filtered[c].dropna().unique()])[:2000]
            selected = st.multiselect(f"{c} の値", vals)
            if selected:
                filtered = filtered[filtered[c].astype(str).isin(selected)]
            else:
                filtered = df

st.subheader("変換・フィルタ後のデータ")
st.dataframe(filtered.head(1000), use_container_width=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ 変換後データをCSVとしてダウンロード",
    data=csv_bytes,
    file_name="data_transformed.csv",
    mime="text/csv",
)

# -----------------------------
# グラフ描画
# -----------------------------
st.subheader("グラフ作成")
chart_type = st.selectbox(
    "グラフの種類", ["棒グラフ", "折れ線グラフ", "散布図"], index=2
)

x_col = st.selectbox("X軸に使用する列", filtered.columns)
numeric_cols = [
    c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])
]
if chart_type in ("棒グラフ", "折れ線グラフ", "散布図"):
    if not numeric_cols:
        st.warning(
            "Y軸に使用できる数値列が存在しません。列を数値型に変換してください。"
        )
        st.stop()
    y_col = st.selectbox("Y軸に使用する列", numeric_cols)

agg = None
if chart_type in ("棒グラフ", "折れ線グラフ"):
    agg = st.selectbox(
        "集計方法（任意）", ["なし", "合計", "平均", "中央値", "件数"], index=0
    )

plot_df = filtered.copy()
if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
    plot_df = plot_df.sort_values(by=x_col)

if agg and agg != "なし" and chart_type in ("棒グラフ", "折れ線グラフ"):
    if agg == "件数":
        plot_df = (
            plot_df.groupby(x_col, dropna=False)[y_col].count().reset_index(name=y_col)
        )
    elif agg == "合計":
        plot_df = plot_df.groupby(x_col, dropna=False)[y_col].sum().reset_index()
    elif agg == "平均":
        plot_df = plot_df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
    elif agg == "中央値":
        plot_df = plot_df.groupby(x_col, dropna=False)[y_col].median().reset_index()

if chart_type == "棒グラフ":
    st.bar_chart(plot_df, x=x_col, y=y_col, use_container_width=True)
elif chart_type == "折れ線グラフ":
    st.line_chart(plot_df, x=x_col, y=y_col, use_container_width=True)
else:
    if not (pd.api.types.is_numeric_dtype(plot_df[y_col])):
        st.warning("散布図はY軸が数値型である必要があります。")
    else:
        st.scatter_chart(plot_df, x=x_col, y=y_col, use_container_width=True)

with st.expander("💡 ヒント"):
    st.markdown(
        """
        - **列の型変換** を使って、数値や日付、真偽値などに変換できます。
        - 無効な値が含まれる場合は、変換処理が停止してエラーが表示されます。
        - **簡易フィルタ** で特定の条件に絞り込み可能です。
        """
    )
