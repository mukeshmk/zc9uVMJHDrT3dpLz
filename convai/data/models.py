from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey

from convai.data.database import Base


class User(Base):
    """
    User model representing MovieLens Users
    """
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(1), nullable=False)
    occupation = Column(String(100), nullable=True)
    zip_code = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())
    
    # Relationships
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")


class Movie(Base):
    """
    Movie model representing Movies in the dataset
    """
    __tablename__ = "movies"
    
    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    release_date = Column(DateTime, nullable=True)
    video_release_date = Column(DateTime, nullable=True)
    imdb_url = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())
    
    # Relationships
    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    genres = relationship("Genre", secondary="movie_genre", back_populates="movies")


class Genre(Base):
    """
    Genre model for Movie Genres
    """
    __tablename__ = "genres"
    
    genre_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    
    # Relationships
    movies = relationship("Movie", secondary="movie_genre", back_populates="genres")


class MovieGenre(Base):
    """
    Association table for movie-genre relationship
    """
    __tablename__ = "movie_genre"
    
    movie_id = Column(Integer, ForeignKey("movies.movie_id", ondelete="CASCADE"), primary_key=True)
    genre_id = Column(Integer, ForeignKey("genres.genre_id", ondelete="CASCADE"), primary_key=True)


class Rating(Base):
    """
    Rating model representing User Ratings for movies
    """
    __tablename__ = "ratings"
    
    rating_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1-5 scale
    timestamp = Column(Integer, nullable=False)  # Unix timestamp
    rated_at = Column(DateTime, default=lambda: datetime.now())
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
    
