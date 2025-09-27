import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct

def load_metrics(model_directory):
    event_file = os.listdir(f"{model_directory}/logs/train")[0]
    event_file = f"{model_directory}/logs/train/{event_file}"

    g_loss=[]
    d_loss =[]
    d_gp=[]
    d_w_loss=[]
    steps=[]
    real_score = []
    gen_score = []
    for e in summary_iterator(event_file):    
        for v in e.summary.value:
            if v.tag == 'epoch_g_loss':            
                g_loss.append(struct.unpack('f', v.tensor.tensor_content)[0])
            
            elif v.tag == 'epoch_d_loss':            
                d_loss.append(struct.unpack('f', v.tensor.tensor_content)[0])
            
            elif v.tag == 'epoch_d_gp':            
                d_gp.append(struct.unpack('f', v.tensor.tensor_content)[0])

            elif v.tag == 'epoch_d_wass_loss':            
                d_w_loss.append(struct.unpack('f', v.tensor.tensor_content)[0])

            elif v.tag == 'epoch_real_socre':            
                real_score.append(struct.unpack('f', v.tensor.tensor_content)[0])
            
            elif v.tag == 'epoch_gen_score':            
                gen_score.append(struct.unpack('f', v.tensor.tensor_content)[0])
        return {"g_loss": g_loss, "d_loss": d_loss, "d_gp": d_gp, "d_w_loss": d_w_loss, "real_score": real_score, "gen_score": gen_score}

def get_latest_model_path(model_dir):
    files = os.listdir(model_dir)
    paths = [os.path.join(model_dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)

