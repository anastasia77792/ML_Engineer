from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
from table_user import User
from table_post import Post
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet
from typing import List, Optional

app = FastAPI()

SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int):
    with SessionLocal() as session:
        user = session.query(User).filter(User.id == id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user

@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int):
    with SessionLocal() as session:
        post = session.query(Post).filter(Post.id == id).first()
        if post is None:
            raise HTTPException(status_code=404, detail="Post not found")
        return post

@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = Query(10, gt=0)):
    with SessionLocal() as session:
        feed_actions = session.query(Feed)\
            .filter(Feed.user_id == id)\
            .order_by(desc(Feed.time))\
            .limit(limit)\
            .all()
        return feed_actions

@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = Query(10, gt=0)):
    with SessionLocal() as session:
        feed_actions = session.query(Feed)\
            .filter(Feed.post_id == id)\
            .order_by(desc(Feed.time))\
            .limit(limit)\
            .all()
        return feed_actions

@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(id: int = Query(None), limit: int = Query(10, gt=0)):
    with SessionLocal() as session:
        top_posts = session.query(
            Post.id,
            Post.text,
            Post.topic,
            func.count(Feed.post_id).label('likes_count')
        ).join(
            Feed, Feed.post_id == Post.id
        ).filter(
            Feed.action == 'like'
        ).group_by(
            Post.id
        ).order_by(
            desc('likes_count')
        ).limit(limit).all()
        
        result = []
        for post in top_posts:
            result.append(PostGet(
                id=post.id,
                text=post.text,
                topic=post.topic
            ))
        
        return result