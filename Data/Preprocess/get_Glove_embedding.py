# initialize the word embeddings using Glove
import numpy as np

def preTrained(vocabulary_src, vocabulary_tgt):
    file = '../original/glove.840B.300d.txt'
    all_words = dict()
    all_embeddings = list()
    src_embedding = list()
    tgt_embedding = list()
    n = 0
    with open(file, "r", ) as f:
        for line in f.readlines():
            if n % 200000 == 0:
                print(n)
            try:
                content = line.strip().lower().split()
                all_words[content[0]] = n
                float_content = [x for x in map(float, content[1:])]
                all_embeddings.append(float_content)
            except ValueError:
                pass
            n += 1
    a = 0
    for i,src_word in enumerate(vocabulary_src):
        if all_words.get(src_word) is not None:
            idx = all_words.get(src_word)
            src_embedding.append(all_embeddings[idx])
        else:
            a += 1
            print(i, a, src_word)
            src_embedding.append(np.random.uniform(-0.1, 0.1, 300).tolist())
    print("#"*100)
    b = 0
    for j,tgt_word in enumerate(vocabulary_tgt):
        if all_words.get(tgt_word) is not None:
            idx = all_words.get(tgt_word)
            tgt_embedding.append(all_embeddings[idx])
        else:
            b += 1
            print(j, b, tgt_word)
            tgt_embedding.append(np.random.uniform(-0.1, 0.1, 300).tolist())
    return src_embedding, tgt_embedding


graph_vocabulary_src = np.load('../processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_src.npy').tolist()
graph_vocabulary_tgt = np.load('../processed/SQuAD1.0/Graph_Analysis/SceneGraph/graph_vocabulary_tgt.npy').tolist()

graph_src_embedding, graph_tgt_embedding = preTrained(graph_vocabulary_src, graph_vocabulary_tgt)

np.save('../processed/SQuAD1.0/Graph_Analysis/SceneGraph/embeddings_src', graph_src_embedding)
np.save('../processed/SQuAD1.0/Graph_Analysis/SceneGraph/embeddings_tgt', graph_tgt_embedding)
print('End......')
