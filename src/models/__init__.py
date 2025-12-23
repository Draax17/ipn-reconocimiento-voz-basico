try:
    from .hmm_model import HMMClassifier
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    HMMClassifier = None

from .classifier import train_svm, train_random_forest, train_mlp
