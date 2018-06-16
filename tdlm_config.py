#see table 1 in paper for variable names
from os.path import join
#preprocessing options
vocab_minfreq=10 #minimum vocab frequency to filter
vocab_maxfreq=0.1 #proportion of most-frequent vocab to filter
stopwords="NLTKstopwords.txt"
tm_sent_len=3 #m_1; topic model sequence length
lm_sent_len=30 #m_2; language model sequence length
doc_len=150 #m_3; document max length

#training options
seed=1
batch_size=64 #n_batch
rnn_layer_size=1 #n_layer
rnn_hidden_size=600 #n_hidden
epoch_size=10 #n_epoch
topic_number=100 #k
word_embedding_size=300 #e; setting ignored if word_embeding_model is provided
word_embedding_model="word2vec/skipgram.bin" #pre-trained word embedding (gensim format); None if no pre-trained model
word_embedding_update=True #update word embedding for topic model
filter_sizes=[2] #h
filter_number=20 #a; topic input vector dimension
conv_activation="identity" #relu or identity (identity function is used in paper)
topic_embedding_size=50 #b; topic output vector dimension
learning_rate=0.001 #l
tm_keep_prob=0.4 #p_1
lm_keep_prob=0.6 #p_2
max_grad_norm=5 #gradient clipping
alpha=0.0 #additional loss to penalise similar topics; not used in paper (0.0)
num_samples=0 #sampled softmax to speed up training; not used in paper (0)
tag_embedding_size=0 #tag embedding dimension; 0 to disable tags

#misc
save_model=True #save model to output_dir/output_prefix
verbose=True #print progress

#input/output
name = 'imdb'
output_dir="output"
output_prefix=name+"-model"

train_corpus=join("data", name, "train.txt")
#train_target="data/toy-train-label.txt" #comment out if unsupervised
#train_tag="data/toy-train-tag.txt" #comment out if no document tags
valid_corpus=join("data", name, "valid.txt")
#valid_target="data/toy-valid-label.txt" #comment out if unsupervised
#valid_tag="data/toy-valid-tag.txt" #comment out if no document tags

