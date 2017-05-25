# -*- coding: utf-8 -*-
import os
import nltk
# noinspection PyPep8Naming
import six.moves.cPickle as pickle
import urllib2
import numpy as np
from multiprocessing import Process, Lock
from vocab_mapping import VocabMapping

dirs = ["data/aclImdb/test/pos", "data/aclImdb/test/neg",
        "data/aclImdb/train/pos", "data/aclImdb/train/neg"]
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def build_data(max_seq_length, max_vocab_size):
    if os.path.exists("data/processed"):
        return
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not os.path.exists("data/checkpoints/"):
        os.makedirs("data/checkpoints")
    if not os.path.isdir("data/aclImdb"):
        print "Data not found, downloading dataset..."
        file_name = download_file(url)
        import tarfile
        tar_file = tarfile.open(file_name, 'r:gz')
        print "Extracting dataset..."
        tar_file.extractall('data/')
        tar_file.close()
    if os.path.exists("data/vocab.txt"):
        print "vocab mapping found..."
    else:
        print "no vocab mapping found, running preprocessor..."
        create_vocab(dirs, max_vocab_size)
    os.makedirs("data/processed/")
    print "No processed data file found, running preprocessor..."
    vocab = VocabMapping()
    dir_count = 0
    processes = []
    lock = Lock()
    for d in dirs:
        print "Processing data with process: " + str(dir_count)
        p = Process(
            target=create_processed_data_file,
            args=(vocab, d, dir_count, max_seq_length, lock)
        )
        p.start()
        processes.append(p)
        dir_count += 1
    for p in processes:
        if p.is_alive():
            p.join()


def create_processed_data_file(vocab_mapping, directory, pid, max_seq_length, lock):
    """
    To speed up the data processing (I probably did it way too inefficiently),
    I decided to split the task in n processes, where n is the number of directories
    A lock was used to ensure while writing to std.out bad things don't happen.
    :type vocab_mapping: VocabMapping
    :type directory: str
    :param pid
    :param max_seq_length
    :type lock: Lock
    """
    data = []
    for i, f in enumerate(os.listdir(directory)):
        # Print the file name each 100 files.
        if i % 100 == 0:
            lock.acquire()
            print "Processing: " + f + " the " + str(
                i) + "th file... on process: " + str(pid)
            lock.release()
        with open(os.path.join(directory, f), 'r') as review:
            # Change each doc into a list of words in lower-case letters.
            tokens = tokenize(review.read().lower())
            num_tokens = len(tokens)
            # Change each doc into a list of word indices
            indices = [vocab_mapping.get_index(j) for j in tokens]
            # Pad sequence to max length or cut the redundance.
            if len(indices) < max_seq_length:
                indices += [vocab_mapping.get_index("<PAD>")
                            for _ in range(max_seq_length - len(indices))]
            else:
                indices = indices[0: max_seq_length]
        if "pos" in directory:
            indices.append(1)
        else:
            indices.append(0)
        # indices: word index list, label, length
        indices.append(min(num_tokens, max_seq_length))
        # assert len(indices) == max_seq_length + 2, str(len(indices))
        data.append(indices)
    data = np.vstack(data)
    # `data` is now a matrix with shape
    # (os.listdir(directory)) * (max_seq_length + 2)
    lock.acquire()
    print "Saving data file{0} to disk...".format(str(pid))
    lock.release()
    save_data(data, pid)


def download_file(file_url):
    """
    method from
      http://stackoverflow.com/questions/22676/
      how-do-i-download-a-file-over-http-using-python
    """
    file_name = os.path.join("data/", file_url.split('/')[-1])
    u = urllib2.urlopen(file_url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    file_size_dl = 0
    block_sz = 8192
    while True:
        buff = u.read(block_sz)
        if not buff:
            break
        file_size_dl += len(buff)
        f.write(buff)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status += chr(8) * (len(status) + 1)
        print status,
    f.close()
    return file_name


def tokenize(text):
    """
    This function tokenizes sentences
    """
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)


def save_data(array, index):
    """
    Saves processed data numpy array
    """
    name = "data{0}.npy".format(str(index))
    outfile = os.path.join("data/processed/", name)
    print "numpy array is: {0}x{1}".format(len(array), len(array[0]))
    np.save(outfile, array)


def create_vocab(dir_list, max_vocab_size):
    """
    create vocab mapping file
    """
    print "Creating vocab mapping..."
    dic = {}
    for d in dir_list:
        for f in os.listdir(d):
            with open(os.path.join(d, f), 'r') as review:
                tokens = tokenize(review.read().lower())
                for t in tokens:
                    if t not in dic:
                        dic[t] = 1
                    else:
                        dic[t] += 1
    d = {}
    counter = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        d[w] = counter
        counter += 1
        # take most frequent 50k tokens
        if counter >= max_vocab_size:
            break
    # add out of vocab token and pad token
    d["<UNK>"] = counter
    counter += 1
    d["<PAD>"] = counter
    with open('data/vocab.txt', 'wb') as handle:
        pickle.dump(d, handle)
