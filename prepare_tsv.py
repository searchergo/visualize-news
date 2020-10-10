import numpy as np
import pandas as pd

# select news.title, news.published_at, news.polarity, news.url, reduced_embeddings.vector from reduced_embeddings left join news
# on news.id = reduced_embeddings.id
# where reduced_embeddings.day = CURRENT_DATE
embedding_size = 100

str_to_np = lambda x: np.fromstring(x.replace('(', '').replace(')', ''), sep=',', count=embedding_size, dtype=float)
fname = 'tsv_prepare.csv'
data = pd.read_csv(fname,
                   dtype={
                       'title': str,
                       'published_at': str,
                       'polarity': str,
                       'url': str},
                   converters={'vector':str_to_np})

full_embedding = np.array(list(data.vector.values))
print(full_embedding.shape)

# data[['title', 'published_at']].to_csv('data.tsv', sep='\t', index=False)
data[['title']].to_csv('data.tsv', sep='\t', index=False, header=None)
np.savetxt('embeddings.tsv', full_embedding, '%.6f', delimiter='\t')