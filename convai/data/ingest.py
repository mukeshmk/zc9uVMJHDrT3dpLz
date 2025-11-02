import re
import csv
from pathlib import Path
from typing import Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from convai.data import database
from convai.utils import download
from convai.data.models import User, Movie, Genre, Rating


class MovieLensLoader:
    USERS_FILE = "u.user"
    ITEMS_FILE = "u.item"
    RATINGS_FILE = "u.data"
    GENRES_FILE = "u.genre"
    
    @staticmethod
    def load_genres(db: Session, data_path: Path) -> int:
        count = 0
        genres_file = data_path / MovieLensLoader.GENRES_FILE

        with open(genres_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    genre_name, genre_id = row[0], int(row[1])
                    genre = Genre(
                        genre_id = genre_id,
                        name = genre_name
                    )
                    db.add(genre)
                    count += 1
            db.commit()
        
        return count
    
    @staticmethod
    def load_users(db: Session, data_path: Path) -> int:
        users_file = data_path / MovieLensLoader.USERS_FILE
        count = 0
        
        with open(users_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 5:
                    user_id, age, gender, occupation, zip_code = row[:5]
                    
                    # Check if user already exists
                    existing = db.query(User).filter(User.user_id == int(user_id)).first()
                    if not existing:
                        user = User(
                            user_id=int(user_id),
                            age=int(age),
                            gender=gender,
                            occupation=occupation if occupation else None,
                            zip_code=zip_code if zip_code else None
                        )
                        db.add(user)
                        count += 1
            
            db.commit()
        return count
    
    @staticmethod
    def load_movies(db: Session, data_path: Path) -> int:
        items_file = data_path / MovieLensLoader.ITEMS_FILE
        count = 0
        
        with open(items_file, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 24:
                    movie_id = int(row[0])
                    title = re.sub(r'\s*\(\d{4}\)\s*$', '', row[1])
                    release_date_str = row[2]
                    
                    # Parse release date
                    release_date = None
                    if release_date_str:
                        try:
                            release_date = datetime.strptime(release_date_str, "%d-%b-%Y")
                        except ValueError:
                            pass
                    
                    # Check if movie already exists
                    existing = db.query(Movie).filter(Movie.movie_id == movie_id).first()
                    if not existing:
                        movie = Movie(
                            movie_id=movie_id,
                            title=title,
                            release_date=release_date,
                            imdb_url=row[4] if len(row) > 4 else None
                        )
                        db.add(movie)
                        db.flush()
                        
                        # Add genres (columns 5-23 are genre flags)
                        for genre_idx in range(5, 24):
                            if row[genre_idx] == '1':
                                genre_id = genre_idx - 5
                                genre = db.query(Genre).filter(
                                    Genre.genre_id == genre_id
                                ).first()
                                if genre:
                                    movie.genres.append(genre)
                        count += 1
            
            db.commit()
        return count
    
    @staticmethod
    def load_ratings(db: Session, data_path: Path, batch_size: int = 1000) -> int:
        ratings_file = data_path / MovieLensLoader.RATINGS_FILE
        count = 0
        batch = []
        
        with open(ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    user_id = int(row[0])
                    movie_id = int(row[1])
                    rating = int(row[2])
                    timestamp = int(row[3])
                    
                    # Check if rating already exists
                    existing = db.query(Rating).filter(
                        (Rating.user_id == user_id) & (Rating.movie_id == movie_id)
                    ).first()
                    
                    if not existing:
                        rating_obj = Rating(
                            user_id=user_id,
                            movie_id=movie_id,
                            rating=rating,
                            timestamp=timestamp
                        )
                        batch.append(rating_obj)
                        count += 1
                        
                        # Commit in batches
                        if len(batch) >= batch_size:
                            db.add_all(batch)
                            db.commit()
                            logger.debug(f"Loaded {count} ratings")
                            batch = []
            
            # Commit remaining ratings
            if batch:
                db.add_all(batch)
                db.commit()
        
        return count
    
    @staticmethod
    def load_all(db: Session, data_path: str = "./ml-100k") -> Tuple[int, int, int, int]:
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Load in order: genres -> users -> movies -> ratings
        genres_count = MovieLensLoader.load_genres(db, data_path)
        users_count = MovieLensLoader.load_users(db, data_path)
        movies_count = MovieLensLoader.load_movies(db, data_path)
        ratings_count = MovieLensLoader.load_ratings(db, data_path)
        
        return users_count, movies_count, ratings_count, genres_count


if __name__ == '__main__':

    extracted_path = download.download_and_extract_zip()
    dataset_path =  Path(f"{extracted_path}/ml-100k")
    
    l = MovieLensLoader()

    database.init_db()
    with database.SessionLocal() as db:
        l.load_all(db, dataset_path)
    
    download.remove_temp_dir(extracted_path)
