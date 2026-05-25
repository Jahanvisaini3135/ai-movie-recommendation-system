"""
Movie Recommendation System Views
Integrates with advanced TMDB model training system
"""

import logging
import os
import threading
import json
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches

import pandas as pd
import numpy as np
from scipy.sparse import load_npz

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

# Global cache for recommender system
_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


class MovieRecommender:
    """Integrated recommender system matching training/infer.py logic"""

    def __init__(self, model_dir='models', progress_callback=None):
        """Initialize with trained model directory"""
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None

        self._load_models(progress_callback)

    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        global _MODEL_LOAD_PROGRESS

        logger.info(f"Loading models from {self.model_dir}...")

        # Load metadata
        if progress_callback:
            progress_callback(10)

        self.metadata = pd.read_parquet(
            self.model_dir / 'movie_metadata.parquet'
        )

        if progress_callback:
            progress_callback(25)

        # Load similarity matrix
        if progress_callback:
            progress_callback(40)

        if (self.model_dir / 'similarity_matrix.npz').exists():
            self.similarity_matrix = load_npz(
                self.model_dir / 'similarity_matrix.npz'
            ).toarray()
        else:
            self.similarity_matrix = np.load(
                self.model_dir / 'similarity_matrix.npy'
            )

        if progress_callback:
            progress_callback(65)

        # Load title mapping
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)

        if progress_callback:
            progress_callback(80)

        # Load config
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)

        if progress_callback:
            progress_callback(100)

        logger.info(
            f"Loaded {self.config['n_movies']:,} movies successfully"
        )

    def find_movie(self, title: str) -> Optional[str]:
        """Find closest matching movie title"""
        matches = get_close_matches(
            title,
            self.title_to_idx.keys(),
            n=1,
            cutoff=0.6
        )

        return matches[0] if matches else None

    def search_movies(self, query: str, n: int = 20) -> List[str]:
        """Search movies by partial title"""
        query_lower = query.lower()

        return [
            title for title in self.title_to_idx.keys()
            if query_lower in title.lower()
        ][:n]

    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_rating: float = None
    ) -> Dict:
        """Get movie recommendations with optional filtering"""

        matched_title = self.find_movie(movie_title)

        if not matched_title:
            return {
                'error': f"Movie '{movie_title}' not found",
                'suggestions': self.search_movies(movie_title, 5)
            }

        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]

        # Similarity scores
        sim_scores = list(
            enumerate(self.similarity_matrix[movie_idx])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )[1:]  # Exclude itself

        recommendations = []

        for idx, score in sim_scores:

            if len(recommendations) >= n:
                break

            if idx >= len(self.metadata):
                continue

            movie = self.metadata.iloc[idx]

            # Rating filter
            if (
                min_rating is not None and
                pd.notna(movie.get('vote_average')) and
                movie.get('vote_average', 0) < min_rating
            ):
                continue

            recommendations.append({
                'title': movie.get('title', 'Unknown'),
                'vote_average': movie.get('vote_average', 0),
                'genres': movie.get('genres', ''),
                'overview': movie.get('overview', ''),
                'poster_path': movie.get('poster_path', ''),
                'release_date': movie.get('release_date', ''),
                'similarity': float(score)
            })

        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': (
                    source_movie['primary_company']
                    if pd.notna(source_movie['primary_company'])
                    else 'Unknown'
                ),
                'rating': (
                    f"{source_movie['vote_average']:.1f}/10"
                    if pd.notna(source_movie['vote_average'])
                    else 'N/A'
                ),
                'genres': (
                    ', '.join(source_movie['genres'][:3])
                    if isinstance(source_movie['genres'], list)
                    else 'N/A'
                )
            },
            'recommendations': recommendations
        }


def _load_model_in_background():
    """Load model in background thread"""

    global _RECOMMENDER
    global _MODEL_LOADING
    global _MODEL_LOAD_PROGRESS
    global _LOAD_ERROR

    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None

    model_dir = getattr(
        settings,
        'MODEL_DIR',
        os.environ.get('MODEL_DIR', 'models')
    )

    # Fallback to static directory
    if not Path(model_dir).exists():
        model_dir = 'static'
        logger.warning(
            "Model directory not found, using static directory"
        )

    try:

        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS

            _MODEL_LOAD_PROGRESS = progress
            logger.info(
                f"Model loading progress: {progress}%"
            )

        _RECOMMENDER = MovieRecommender(
            model_dir,
            progress_callback
        )

        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100

        logger.info("Model loaded successfully")

    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)

        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    """Start model loading in background if not already started"""

    global _LOADING_THREAD
    global _RECOMMENDER
    global _MODEL_LOADING

    if _RECOMMENDER is None and not _MODEL_LOADING:

        if (
            _LOADING_THREAD is None or
            not _LOADING_THREAD.is_alive()
        ):
            logger.info(
                "Starting model loading in background..."
            )

            _LOADING_THREAD = threading.Thread(
                target=_load_model_in_background,
                daemon=True
            )

            _LOADING_THREAD.start()


def _get_recommender():
    """Get or initialize recommender"""

    global _RECOMMENDER
    global _LOAD_ERROR

    if _RECOMMENDER is None:
        _start_model_loading()

        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)

        return None

    return _RECOMMENDER


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    """

    # Start loading
    _start_model_loading()

    recommender = _get_recommender()

    # Still loading
    if recommender is None:

        if request.method == 'GET':
            return render(
                request,
                'recommender/index.html',
                {
                    'all_movie_names': [],
                    'total_movies': 0,
                }
            )

        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': (
                    'Model is still loading. '
                    'Please wait a moment and try again.'
                ),
            }
        )

    titles_list = list(recommender.title_to_idx.keys())

    # GET request
    if request.method == 'GET':

        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
            }
        )

    # POST request
    movie_name = request.POST.get(
        'movie_name',
        ''
    ).strip()

    if not movie_name:

        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'error_message': 'Please enter a movie name.',
            }
        )

    # Get recommendations
    result = recommender.get_recommendations(
        movie_name,
        n=15
    )

    if 'error' in result:

        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'input_movie_name': movie_name,
                'error_message': result['error'],
                'suggestions': result.get('suggestions', [])
            }
        )

    return render(
        request,
        'recommender/result.html',
        {
            'all_movie_names': titles_list,
            'input_movie_name': result['query_movie'],
            'source_movie': result['source_movie'],
            'recommended_movies': result['recommendations'],
            'total_recommendations': len(
                result['recommendations']
            ),
        }
    )


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for autocomplete"""

    query = request.GET.get('q', '').strip()

    if len(query) < 2:
        return JsonResponse({
            'movies': [],
            'count': 0
        })

    try:
        recommender = _get_recommender()

        if recommender is None:
            return JsonResponse({
                'movies': [],
                'count': 0,
                'loading': True
            })

        matching_movies = recommender.search_movies(
            query,
            n=20
        )

        return JsonResponse({
            'movies': matching_movies,
            'count': len(matching_movies)
        })

    except Exception as e:
        logger.error(f"Error in search: {e}")

        return JsonResponse({
            'error': 'Search failed'
        }, status=500)


@require_http_methods(["GET"])
def model_status(request):
    """Check model loading status"""

    global _RECOMMENDER
    global _MODEL_LOADING
    global _MODEL_LOAD_PROGRESS
    global _LOAD_ERROR

    _start_model_loading()

    if _LOAD_ERROR:

        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'error': _LOAD_ERROR
        })

    elif _RECOMMENDER is not None:

        return JsonResponse({
            'loaded': True,
            'progress': 100,
            'status': 'ready'
        })

    elif _MODEL_LOADING:

        return JsonResponse({
            'loaded': False,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })

    else:

        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'initializing'
        })


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""

    try:
        recommender = _get_recommender()

        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")

        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)