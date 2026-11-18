from .features import model_matrix
from .schema import FEATURES
from .train import build_model

def fit_final(b):
    x=model_matrix(b,FEATURES); y=b.label_binary.astype(int); _, model=build_model("E2"); model.fit(x,y); return model
