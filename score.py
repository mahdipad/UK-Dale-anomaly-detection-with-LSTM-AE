
import numpy as np

FEATURE_WEIGHTS = np.array([0.55,0.20,0.10,0.15],dtype=np.float32)

def reconstruction_error_matrix(model,X,batch_size=1024):
    rec=model.predict(X,batch_size=batch_size,verbose=1)
    return np.square(X-rec).astype(np.float32)

def aggregate_scores(squared_error,top_fraction=0.10):
    e=np.asarray(squared_error,dtype=np.float32)
    if e.ndim!=3: raise ValueError("Expected (samples,timesteps,features)")
    e=e[:,:,:4]
    weighted=(e*FEATURE_WEIGHTS[None,None,:]).sum(axis=2)
    mean_score=weighted.mean(axis=1)
    k=max(1,int(weighted.shape[1]*top_fraction))
    topk=np.partition(weighted,-k,axis=1)[:,-k:].mean(axis=1)
    return (0.70*topk+0.30*mean_score).astype(np.float32)

def reconstruction_score(model,X,batch_size=1024,top_fraction=0.10):
    return aggregate_scores(reconstruction_error_matrix(model,X,batch_size),top_fraction)
