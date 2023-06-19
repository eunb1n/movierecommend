# Library load
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv("/content/sample_data/tmdb_5000_movies.csv")
print(movies.shape)
movies.head(1)

# null 값 체크
movies.isnull().sum()

# 주요 컬럼 추출
movies_df=movies[['id', 'title', 'genres', 'vote_average', 'vote_count']]
movies_df.head(5)

# 컬럼 길이 100으로 세팅
pd.set_option('max_colwidth', 100)
movies_df[['genres']][:1]

# apply()에 literal_eval 함수를 적용해 문자열을 객체로 변경
movies_df['genres']=movies_df['genres'].apply(literal_eval)
movies_df.head(1)

# apply lambda를 이용하여 리스트 내 여러 개의 딕셔너리의 'name' 키 찾아 리스트 객체로 변환.
movies_df['genres']=movies_df['genres'].apply(lambda x : [ y['name'] for y in x])
movies_df[['genres']][:1]

movies_df[['genres']]

# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환.
movies_df['genres_literal']=movies_df['genres'].apply(lambda x : (' ').join(x))

# min_df는 너무 드물게로 나타나는 용어를 제거하는 데 사용. min_df = 0.01은 "문서의 1 % 미만"에 나타나는 용어를 무시한다. 
# ngram_range는 n-그램 범위.
count_vect=CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat=count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)

genre_sim=cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:1])

# [:, ::-1] axis = 1 기준으로 2차원 numpy 배열 뒤집기
genre_sim_sorted_ind=genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:3])

movies_df[['title', 'vote_average', 'vote_count']].sort_values('vote_average', ascending=False)[:10]

percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)  # 평점을 부여하기 위한 최소 평가 수
C = movies_df['vote_average'].mean()  # 전체 영화의 평균 평점

def weighted_vote_average(record):
  v = record['vote_count']  # 영화에 평가를 매긴 횟수
  R = record['vote_average']  # 영화의 평균 평점

  return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )  # 가중 평점 계산 식

movies_df['weighted_vote'] = movies.apply(weighted_vote_average, axis=1)

movies_df[['title', 'weighted_vote', 'vote_count']].sort_values('weighted_vote', ascending=False)[:10]


def find_sim_movie(df, sorted_ind, title_name, top_n=10):
  title_movie = df[df['title'] == title_name]
  title_index = title_movie.index.values

  # top_n의 2배에 해당하는 장르 유사성이 높은 인덱스 추출
  similar_indexes = sorted_ind[title_index, :(top_n*2)]
  # reshape(-1) 1차열 배열 반환
  similar_indexes = similar_indexes.reshape(-1)
  # 기준 영화 인덱스는 제외
  similar_indexes = similar_indexes[similar_indexes != title_index]

  # top_n의 2배에 해당하는 후보군에서 weighted_vote가 높은 순으로 top_n만큼 추출
  return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies=find_sim_movie(movies_df, genre_sim_sorted_ind, 'No Country for Old Men', 10)
similar_movies[['title', 'vote_count', 'weighted_vote']]

