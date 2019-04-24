"""
Original script: Out-of-core classification of text documents

The original script is modified and
collected fetching Reuters-21578 data function by @nghia nh.

This separate the downloading and fetching data functions.
TODO: Then we need to independently fetch_data first before get_reuters.

"""
# Authors: Eustache Diemert <eustache@diemert.fr>
#          @FedericoV <https://github.com/FedericoV/>
# License: BSD 3 clause

from glob import glob
import os.path
import re
import tarfile
import sys

import numpy as np

from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves.urllib.request import urlretrieve
from sklearn.datasets import get_data_home


###############################################################################
# Download Reuters

def fetch_data(data_path=None):
    """
    Download and uncompress Reuters 21578.

    Note that this script does not check whether
    the data is downloaded in data_path or not.
    :param data_path:
    :return:
    """
    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    """Download the dataset."""
    print("downloading dataset (once and for all) into %s" %
          data_path)
    os.mkdir(data_path)

    def progress(blocknum, bs, size):
        total_sz_mb = '%.2f MB' % (size / 1e6)
        current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
        sys.stdout.write(
            '\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb))

    archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
    urlretrieve(DOWNLOAD_URL, filename=archive_path,
                reporthook=progress)
    sys.stdout.write('\r')
    print("untarring Reuters dataset...")
    tarfile.open(archive_path, 'r:gz').extractall(data_path)
    print("done.")


###############################################################################
# Parsing Reuters

class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path):
    """Iterate over documents of the Reuters dataset.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.
    :param data_path: data file location
    """
    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        with open(filename, 'rb') as f:
            for doc in parser.parse(f):
                yield doc


def get_batch(doc_iter, pos_class='acq'):
    """Extract a batch of examples, return a tuple X_text, y.
    We learn a binary classification between the "acq" class and all the others.
    "acq" was chosen as it is more or less evenly distributed in the Reuters
    files. For other datasets, one should take care of creating a test set with
    a realistic portion of positive instances.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(u'{title}\n\n{body}'.format(**doc), pos_class in doc['topics'])
            for doc in doc_iter if doc['topics']]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    x_text, y = zip(*data)
    return x_text, y


def get_reuters(folder_dir, positive_labels):
    # Iterator over parsed Reuters SGML files.
    data_stream = stream_reuters_documents(data_path=folder_dir)
    x_text, y = get_batch(data_stream, positive_labels)

    return x_text, y
