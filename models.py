
# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class UserDetails(Base):
    __tablename__ = "user_details"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    mobile = Column(String, nullable=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # Store hashed passwords in production
    is_deleted = Column(Boolean, default=False)
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), onupdate=func.now())

    chats = relationship("UsersChat", back_populates="user")

class UsersChat(Base):
    __tablename__ = "users_chat"

    id = Column(Integer, primary_key=True, index=True)
    chat_name = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("user_details.id"), nullable=False)
    is_deleted = Column(Boolean, default=False)
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("UserDetails", back_populates="chats")
    chat_details = relationship("UserChatDetails", back_populates="chat")

class UserChatDetails(Base):
    __tablename__ = "user_chat_details"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("users_chat.id"), nullable=False)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    file_upload_url = Column(String, nullable=True)
    is_deleted = Column(Boolean, default=False)
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), onupdate=func.now())

    chat = relationship("UsersChat", back_populates="chat_details")