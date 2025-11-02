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
    def load_all(db: Session, data_path: str = "./ml-100k") -> Tuple[int, int, int, int]:
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Load in order: genres -> users -> movies -> ratings
        genres_count = MovieLensLoader.load_genres(db, data_path)
        
        
        return  genres_count


if __name__ == '__main__':

    extracted_path = download.download_and_extract_zip()
    dataset_path =  Path(f"{extracted_path}/ml-100k")
    
    l = MovieLensLoader()

    database.init_db()
    with database.SessionLocal() as db:
        l.load_all(db, dataset_path)
    
    download.remove_temp_dir(extracted_path)
