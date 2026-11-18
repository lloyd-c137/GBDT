from .features import model_matrix
from .schema import FEATURES
from .train import build_model, validate_training_matrix

def fit_final(b):
    x=model_matrix(b,FEATURES); validate_training_matrix(x); y=b.label_binary.astype(int); _, model=build_model("E2"); model.fit(x,y); return model
