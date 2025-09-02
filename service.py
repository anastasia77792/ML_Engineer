# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
import psycopg2
from catboost import CatBoostClassifier, Pool

# ----------- CONST --------------------------------------------------
DSN = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
#DSN = "postgresql://postgres:postgres_password@127.0.0.1:5432/final_project"
ENGINE = create_engine(DSN, pool_pre_ping=True)

MODEL_PATH = "cat_model"

ADD_FEATS = [
    "post_ctr", "topic_ctr", "post_age_hours",
    "user_post_count", "user_ctr", "user_topic_ctr",
    "post_user_count", "post_gender_diversity",
    "user_action_count", "user_topic_count",
    "hour", "weekday", "is_weekend",
    "city_freq"
]

ADD_FEATS += [
    "user_source_ctr",
    "user_topic_ctr_norm"
]

ADD_FEATS += ["user_action_ctr", "topic_source_ctr", "post_age_topic_diff"]

# Базовые признаки
BASE_FEATS = [
    "user_id", "post_id", "source", "os", "action", "topic", "country",
    "gender", "age", "age_bin", "exp_group", "city"
]
# Категориальные признаки
CAT_COLS = ["user_id", "post_id", "city", "country", "os", "source", "action", "topic"]
CAT_COLS += ["user_topic", "source_action", "city_topic"]
FEATS = BASE_FEATS + ADD_FEATS
# Фиксируем случайность
RAND = 42

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path(MODEL_PATH)
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_features(batch_size: int = 50_000):
    """
    Читает таблицу чанками, сразу обрабатывает
    и не держит всё в памяти.
    """

    query = """
    SELECT
        user_id,
        (event->>'post_id')::int          AS post_id,
        (event->>'topic')                 AS topic,
        (event->>'gender')                AS gender,
        (event->>'age')::int              AS age,
        (event->>'exp_group')::int        AS exp_group,
        (event->>'action')                AS action,
        (event->>'country')               AS country,
        (event->>'city')                  AS city,
        (event->>'os')                    AS os,
        (event->>'source')                AS source,
        (event->>'target')::int           AS target,
        (event->>'timestamp')::timestamp  AS timestamp
    FROM pavel_kim_features_lesson_27,
         jsonb_array_elements(user_data) AS event;
    """

    for chunk in pd.read_sql(query, ENGINE, chunksize=batch_size):
        # сужаем типы
        chunk = chunk.astype({
            "user_id": "int32",  # пользователей <= 163k, не влезет в uint16, тогда оставь int32
            "post_id": "uint16",  # у тебя 7k постов → ок
            "age": "uint8",  # возраст < 128 → ок
            "exp_group": "uint8",
            "target": "uint8"
        })
        yield chunk


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 10000
    engine = ENGINE
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE, parse_dates=["timestamp"]):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# ----------- GLOBALS -----------------------------------------------
app = FastAPI()
model_g: Optional[CatBoostClassifier] = None
catalog: Optional[pd.DataFrame] = None
user_feats: Optional[pd.DataFrame] = None
seen_dict: Optional[Dict[int, Set[int]]] = None
post_text_cache: Dict[int, str] = {}
_initialized = False
# ----------- HELPERS -----------------------------------------------


def _age_to_bin(a: float) -> int:
    if pd.isna(a): return 5
    a = int(a)
    return np.digitize(a, [18, 26, 36, 46, 61])

def _build_seen(df: pd.DataFrame) -> Dict[int, Set[int]]:
    res: Dict[int, Set[int]] = {}
    for uid, pid in zip(df.user_id, df.post_id):
        res.setdefault(uid, set()).add(pid)
    return res

def _fetch_texts(pids: List[int]):
    missing = [pid for pid in pids if pid not in post_text_cache]
    if not missing:
        return
    q = f"SELECT post_id, text FROM post_text_df WHERE post_id IN ({','.join(map(str, missing))})"
    txt = pd.read_sql(q, ENGINE)
    post_text_cache.update(txt.set_index("post_id")["text"].to_dict())

# ----------- INIT ---------------------------------------------------
def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимизируем память:
    - все id → uint16 (если влезает)
    - счётчики → uint16
    - float-признаки → float16
    - категориальные → category
    """
    df = df.astype({
        "user_id": "int32",     # пользователей <= 163k, не влезет в uint16, тогда оставь int32
        "post_id": "uint16",     # у тебя 7k постов → ок
        "age": "uint8",          # возраст < 128 → ок
        "exp_group": "uint8",
        "target": "uint8"
    })

    # числовые float-признаки
    float_feats = [
        "post_ctr", "topic_ctr", "user_ctr", "user_topic_ctr",
        "city_freq", "post_age_hours"
    ]
    for col in float_feats:
        if col in df.columns:
            df[col] = df[col].astype("float16")

    # счётчики → uint16
    count_feats = [
        "user_post_count", "user_action_count",
        "user_topic_count", "post_user_count"
    ]
    for col in count_feats:
        if col in df.columns:
            df[col] = df[col].astype("uint16")

    # мелкие категориальные
    cat_feats = ["source", "os", "gender", "action", "country", "topic", "city"]
    for col in cat_feats:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


from collections import defaultdict

def build_features(feats: pd.DataFrame) -> pd.DataFrame:
    # ---------- post_ctr ----------
    post_sum = defaultdict(int)
    post_cnt = defaultdict(int)
    for pid, target in zip(feats["post_id"], feats["target"]):
        post_sum[pid] += target
        post_cnt[pid] += 1
    post_ctr = {pid: post_sum[pid] / post_cnt[pid] for pid in post_cnt}
    feats["post_ctr"] = feats["post_id"].map(post_ctr).astype("float16")

    # ---------- topic_ctr ----------
    topic_sum = defaultdict(int)
    topic_cnt = defaultdict(int)
    for topic, target in zip(feats["topic"], feats["target"]):
        topic_sum[topic] += target
        topic_cnt[topic] += 1
    topic_ctr = {t: topic_sum[t] / topic_cnt[t] for t in topic_cnt}
    feats["topic_ctr"] = feats["topic"].map(topic_ctr).astype("float16")

    # ---------- post_age_hours ----------
    max_ts = feats["timestamp"].max()
    feats["post_age_hours"] = (
        (max_ts - feats["timestamp"]).dt.total_seconds() / 3600
    ).astype("float16")

    # ---------- user_post_count ----------
    user_cnt = defaultdict(int)
    for uid in feats["user_id"]:
        user_cnt[uid] += 1
    feats["user_post_count"] = feats["user_id"].map(user_cnt).astype("uint16")

    # ---------- user_ctr ----------
    user_sum = defaultdict(int)
    for uid, target in zip(feats["user_id"], feats["target"]):
        user_sum[uid] += target
    user_ctr = {uid: user_sum[uid] / user_cnt[uid] for uid in user_cnt}
    feats["user_ctr"] = feats["user_id"].map(user_ctr).astype("float16")

    # ---------- user_topic_ctr ----------
    ut_sum = defaultdict(int)
    ut_cnt = defaultdict(int)
    for uid, topic, target in zip(feats["user_id"], feats["topic"], feats["target"]):
        key = (uid, topic)
        ut_sum[key] += target
        ut_cnt[key] += 1
    ut_ctr = {k: ut_sum[k] / ut_cnt[k] for k in ut_sum}
    feats["user_topic_ctr"] = list(map(lambda x: ut_ctr.get((x[0], x[1]), 0),
                                       zip(feats["user_id"], feats["topic"])))
    feats["user_topic_ctr"] = feats["user_topic_ctr"].astype("float16")

    # ---------- post_user_count ----------
    post_users = defaultdict(int)
    for pid in feats["post_id"]:
        post_users[pid] += 1
    feats["post_user_count"] = feats["post_id"].map(post_users).astype("uint16")

    # ---------- post_gender_diversity ----------
    post_genders = defaultdict(set)
    for pid, gender in zip(feats["post_id"], feats["gender"]):
        post_genders[pid].add(gender)
    post_gender_div = {pid: len(gset) for pid, gset in post_genders.items()}
    feats["post_gender_diversity"] = feats["post_id"].map(post_gender_div).astype("uint8")

    # ---------- user_action_count ----------
    ua_cnt = defaultdict(int)
    for uid, action in zip(feats["user_id"], feats["action"]):
        ua_cnt[(uid, action)] += 1
    feats["user_action_count"] = list(map(lambda x: ua_cnt[(x[0], x[1])],
                                          zip(feats["user_id"], feats["action"])))
    feats["user_action_count"] = feats["user_action_count"].astype("uint16")

    # ---------- user_topic_count ----------
    ut_count = defaultdict(int)
    for uid, topic in zip(feats["user_id"], feats["topic"]):
        ut_count[(uid, topic)] += 1
    feats["user_topic_count"] = list(map(lambda x: ut_count[(x[0], x[1])],
                                         zip(feats["user_id"], feats["topic"])))
    feats["user_topic_count"] = feats["user_topic_count"].astype("uint16")

    # ---------- city_freq ----------
    city_cnt = defaultdict(int)
    for city in feats["city"]:
        city_cnt[city] += 1
    total = len(feats)
    city_freq = {c: city_cnt[c] / total for c in city_cnt}
    feats["city_freq"] = feats["city"].map(city_freq).astype("float16")

    # ---------- user_source_ctr ----------
    user_source_sum = defaultdict(int)
    user_source_cnt = defaultdict(int)
    for uid, source, target in zip(feats["user_id"], feats["source"], feats["target"]):
        key = (uid, source)
        user_source_sum[key] += target
        user_source_cnt[key] += 1
    user_source_ctr = {k: user_source_sum[k] / user_source_cnt[k] for k in user_source_sum}
    feats["user_source_ctr"] = list(map(lambda x: user_source_ctr.get((x[0], x[1]), 0),
                                        zip(feats["user_id"], feats["source"]))).astype("float16")

    # ---------- user_topic_ctr_norm ----------
    feats["user_topic_ctr_norm"] = (feats["user_topic_ctr"] / feats["user_ctr"].replace(0, 1)).astype("float16")

    # ---------- user_action_ctr ----------
    user_action_sum = defaultdict(int)
    user_action_cnt = defaultdict(int)
    for uid, action, target in zip(feats["user_id"], feats["action"], feats["target"]):
        key = (uid, action)
        user_action_sum[key] += target
        user_action_cnt[key] += 1
    user_action_ctr = {k: user_action_sum[k] / user_action_cnt[k] for k in user_action_sum}
    feats["user_action_ctr"] = list(map(lambda x: user_action_ctr.get((x[0], x[1]), 0),
                                        zip(feats["user_id"], feats["action"]))).astype("float16")

    # ---------- topic_source_ctr ----------
    topic_source_sum = defaultdict(int)
    topic_source_cnt = defaultdict(int)
    for topic, source, target in zip(feats["topic"], feats["source"], feats["target"]):
        key = (topic, source)
        topic_source_sum[key] += target
        topic_source_cnt[key] += 1
    topic_source_ctr = {k: topic_source_sum[k] / topic_source_cnt[k] for k in topic_source_sum}
    feats["topic_source_ctr"] = list(map(lambda x: topic_source_ctr.get((x[0], x[1]), 0),
                                         zip(feats["topic"], feats["source"]))).astype("float16")

    # ---------- post_age_topic_diff ----------
    mean_topic_age = defaultdict(float)
    topic_counts = defaultdict(int)
    for topic, age in zip(feats["topic"], feats["post_age_hours"]):
        mean_topic_age[topic] += age
        topic_counts[topic] += 1
    for k in mean_topic_age:
        mean_topic_age[k] /= topic_counts[k]
    feats["post_age_topic_diff"] = list(map(lambda t, a: a - mean_topic_age[t],
                                            zip(feats["topic"], feats["post_age_hours"]))).astype("float16")

    # ---------- user_topic_comb ----------
    feats["user_topic_comb"] = (feats["user_id"].astype(str) + "_" + feats["topic"].astype(str)).astype("category")

    # ---------- source_action ----------
    feats["source_action"] = (feats["source"].astype(str) + "_" + feats["action"].astype(str)).astype("category")

    # ---------- city_topic ----------
    feats["city_topic"] = (feats["city"].astype(str) + "_" + feats["topic"].astype(str)).astype("category")

    return feats


def init_runtime():
    global _initialized, model_g, catalog, user_feats, seen_dict
    if _initialized:
        return

    model_g = load_models()

    # ----------- читаем чанками, сразу обрабатываем ----------------
    feats_iter = load_features(batch_size=50_000)

    feats_list = []
    for chunk in feats_iter:
        # минимизация типов уже на этапе загрузки
        chunk = chunk.astype({
            "user_id": "int32",
            "post_id": "int16",
            "age": "int8",
            "exp_group": "int8",
            "target": "int8"
        })

        # категориальные фичи
        for col in ["city", "topic", "action", "os", "source", "country", "gender"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")

        # бин по возрасту
        chunk["age_bin"] = chunk.age.apply(_age_to_bin).astype("int8")

        # преобразуем timestamp
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")

        feats_list.append(chunk)

    feats = pd.concat(feats_list, ignore_index=True)
    del feats_list  # освободим память

    # ---------------- вычисление признаков -------------------------
    max_ts = feats.timestamp.max()

    # CTR по посту
    post_ctr = feats.groupby("post_id")["target"].mean().astype("float32")
    feats = feats.merge(post_ctr.rename("post_ctr"), on="post_id", how="left")

    # CTR по топику
    topic_ctr = feats.groupby("topic")["target"].mean().astype("float32")
    feats = feats.merge(topic_ctr.rename("topic_ctr"), on="topic", how="left")

    # возраст поста
    feats["post_age_hours"] = (
        (max_ts - feats.timestamp).dt.total_seconds() / 3600
    ).astype("float32")

    # user_post_count
    user_post_count = feats.groupby("user_id")["post_id"].count().astype("int32")
    feats = feats.merge(user_post_count.rename("user_post_count"), on="user_id", how="left")

    # user_ctr
    user_ctr = feats.groupby("user_id")["target"].mean().astype("float32")
    feats = feats.merge(user_ctr.rename("user_ctr"), on="user_id", how="left")

    # user_topic_ctr
    user_topic_ctr = feats.groupby(["user_id", "topic"])["target"].mean().astype("float32")
    feats = feats.merge(user_topic_ctr.rename("user_topic_ctr"), on=["user_id", "topic"], how="left")

    # post_user_count
    post_user_count = feats.groupby("post_id")["user_id"].count().astype("int32")
    feats = feats.merge(post_user_count.rename("post_user_count"), on="post_id", how="left")

    # post_gender_diversity
    post_gender_div = feats.groupby("post_id")["gender"].nunique().astype("int8")
    feats = feats.merge(post_gender_div.rename("post_gender_diversity"), on="post_id", how="left")

    # время
    feats["hour"] = feats["timestamp"].dt.hour.astype("int8")
    feats["weekday"] = feats["timestamp"].dt.weekday.astype("int8")
    feats["is_weekend"] = (feats["weekday"] >= 5).astype("int8")

    # user_action_count
    user_action_count = feats.groupby(["user_id", "action"])["action"].count().astype("int32")
    feats = feats.merge(user_action_count.rename("user_action_count"), on=["user_id", "action"], how="left")

    # user_topic_count
    user_topic_count = feats.groupby(["user_id", "topic"])["target"].count().astype("int32")
    feats = feats.merge(user_topic_count.rename("user_topic_count"), on=["user_id", "topic"], how="left")

    # city_freq
    city_freq = feats["city"].value_counts(normalize=True).astype("float32")
    feats["city_freq"] = feats["city"].map(city_freq)

    # ---------------- новые фичи -------------------------
    # 🆕 CTR по комбинации пользователя и источника
    user_source_ctr = feats.groupby(["user_id", "source"])["target"].mean().astype("float32")
    feats = feats.merge(user_source_ctr.rename("user_source_ctr"), on=["user_id", "source"], how="left")

    # 🆕 Нормализованный CTR пользователя по теме
    feats["user_topic_ctr_norm"] = (feats["user_topic_ctr"] / feats["user_ctr"].replace(0, 1)).astype("float32")

    # 🆕 CTR пользователя по действию
    user_action_ctr = feats.groupby(["user_id", "action"])["target"].mean().astype("float32")
    feats = feats.merge(user_action_ctr.rename("user_action_ctr"), on=["user_id", "action"], how="left")

    # 🆕 CTR по комбинации темы и источника
    topic_source_ctr = feats.groupby(["topic", "source"])["target"].mean().astype("float32")
    feats = feats.merge(topic_source_ctr.rename("topic_source_ctr"), on=["topic", "source"], how="left")

    # 🆕 Разница возраста поста относительно среднего возраста темы
    mean_topic_age = feats.groupby("topic")["post_age_hours"].transform("mean")
    feats["post_age_topic_diff"] = (feats["post_age_hours"] - mean_topic_age).astype("float32")

    # 🆕 Комбинация пользователя и темы
    feats["user_topic_comb"] = feats["user_id"].astype(str) + "_" + feats["topic"].astype(str)
    feats["user_topic_comb"] = feats["user_topic_comb"].astype("category")

    # 🆕 Комбинация источника и действия
    feats["source_action"] = feats["source"].astype(str) + "_" + feats["action"].astype(str)
    feats["source_action"] = feats["source_action"].astype("category")

    # 🆕 Комбинация города и темы
    feats["city_topic"] = feats["city"].astype(str) + "_" + feats["topic"].astype(str)
    feats["city_topic"] = feats["city_topic"].astype("category")

    # ---------------- оптимизация типов -------------------------
    feats = optimize_types(feats)

    # ---------------- кэши -------------------------
    user_feats = (
        feats[["user_id", "source", "os", "gender", "age", "age_bin", "action",
               "country", "exp_group", "city"]]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    catalog = feats[["post_id", "topic"] + ADD_FEATS].drop_duplicates("post_id")
    seen_dict = _build_seen(feats[["user_id", "post_id"]])

    del feats  # очищаем память

    _initialized = True
    print("✅ Runtime initialised (optimized)")


# ----------- API MODELS --------------------------------------------
class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True

# ----------- RECOMMENDER -------------------------------------------
def _recommend(uid: int, now: datetime, limit: int) -> List[PostGet]:
    init_runtime()
    if uid not in user_feats.index:
        raise HTTPException(404, "user not found")

    # кандидаты = все посты - просмотренные
    cand = catalog[~catalog.post_id.isin(seen_dict.get(uid, set()))].copy()
    if cand.empty:
        raise HTTPException(404, "no candidates")

    # добавляем user-признаки
    u = user_feats.loc[uid]
    for col in ["source", "os", "gender", "age", "age_bin",
                "country", "exp_group", "city", "action"]:
        cand[col] = u[col]

    cand["user_id"] = uid
    cand["timestamp"] = now

    FEATS_ORDER = [
        # тут тот же список признаков, что и в train
        "user_id", "post_id", "source", "os", "action", "topic", "country",
        "gender", "age", "age_bin", "exp_group", "city"
    ]
    # предсказание
    cat_idxs = [i for i, col in enumerate(FEATS_ORDER) if col in CAT_COLS]
    cand_pool = Pool(cand[FEATS], cat_features=cat_idxs)
    cand["proba"] = model_g.predict_proba(cand_pool)[:, 1]

    threshold = 0.65
    filtered = cand[cand.proba >= threshold]

    if len(filtered) < limit:
        top = cand.nlargest(limit, "proba")[["post_id", "topic"]]
    else:
        top = filtered.nlargest(limit, "proba")[["post_id", "topic"]]

    _fetch_texts(top.post_id.tolist())
    return [PostGet(id=r.post_id,
                    text=post_text_cache.get(r.post_id, ""),
                    topic=r.topic)
            for r in top.itertuples()]


# ----------- ROUTE --------------------------------------------------
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10):
    return _recommend(id, time, limit)

# ----------- ENTRY --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
