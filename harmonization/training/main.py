# Mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow import concat as tf_concat
from json import dump as json_dump


#################INPUT DATASETS#########################
from input_pipeline.tf_dataset import datasets_from_tfrecords, datasets_from_tfrecords_biasSampling
dataset_pairs = # TO DEFINE


##########INPUT PARAMETERS################
DEST_DIR_PATH = # TO DEFINE
N_EPOCHS = 100
STEPS_PER_EPOCH = 200


# Instancitation of the generators and the discriminators
import sys
sys.path.append('..')
from model_architectures import Generator, Discriminator
gen_univ = Generator()
gens_bwd = [Generator() for _ in range(len(dataset_pairs))]
discs_ref = [Discriminator() for _ in range(len(dataset_pairs))]
discs_bwd = [Discriminator() for _ in range(len(dataset_pairs))]


##################VALIDATION############################
# Definition of an evaluation function for the current model. Takes no argument.
import numpy as np
def eval_model(): return np.random.uniform()
EVAL_FREQ = 5 # evaluates the model every X epochs
#########################################################


# Initialization of the optimizers
INIT_LR = 0.0002
END_LR = 0.00002

n_steps_gen_univ = N_EPOCHS*STEPS_PER_EPOCH*len(dataset_pairs)
n_steps_gen_bwd = N_EPOCHS*STEPS_PER_EPOCH
n_steps_discs = N_EPOCHS*STEPS_PER_EPOCH

def optimizer(n_steps):
    opt = Adam(learning_rate=PolynomialDecay(INIT_LR, n_steps, END_LR))
    return mixed_precision.LossScaleOptimizer(opt, dynamic=True)

genUnivOptimizer = optimizer(n_steps_gen_univ)
genOptimizers_bwd = [optimizer(n_steps_gen_bwd) for _ in range(len(dataset_pairs))]
discOptimizers_ref = [optimizer(n_steps_discs) for _ in range(len(dataset_pairs))]
discOptimizers_bwd = [optimizer(n_steps_discs) for _ in range(len(dataset_pairs))]


# Instantiation of the training objects
from trainers import Discriminator_trainer, Generator_trainer
genTrainers = [Generator_trainer(gen_univ, gens_bwd[i], discs_ref[i], discs_bwd[i], genUnivOptimizer, genOptimizers_bwd[i]) for i in range(len(dataset_pairs))]
discTrainers_ref = [Discriminator_trainer(discs_ref[i], gen_univ, discOptimizers_ref[i]) for i in range(len(dataset_pairs))]
discTrainers_src = [Discriminator_trainer(discs_bwd[i], gens_bwd[i], discOptimizers_bwd[i]) for i in range(len(dataset_pairs))]


# Main training function
DISC_N_BATCHS = 2 # number of batches to train the discriminators
indices_sites = np.arange(len(dataset_pairs))
def train_step():
    np.random.shuffle(indices_sites)
    results = {'genFwd_idLoss':0}
    for idSite in indices_sites:
        batchs = tf_concat([dataset_pairs[idSite].get_next() for _ in range(DISC_N_BATCHS*2)], axis=1)
        imagesRef = batchs[0]
        imagesSrc = batchs[1]
        results[f"discRef_{idSite+1}_loss"] = discTrainers_ref[idSite].train(imagesRef[DISC_N_BATCHS:], imagesSrc[DISC_N_BATCHS:])
        results[f"discSrc_{idSite+1}_loss"] = discTrainers_src[idSite].train(imagesSrc[:DISC_N_BATCHS], imagesRef[:DISC_N_BATCHS])
        
        batchRef, batchSrc = dataset_pairs[idSite].get_next()
        (results[f'genFwd_adv_loss_{idSite+1}'], results[f'genBwd_adv_loss_{idSite+1}'], results[f'cycle_loss_refSref_{idSite+1}'],
         results[f'cycle_loss_srcRsrc_{idSite+1}'], genFwd_idLoss, results[f'genBwd_idLoss_{idSite+1}']) = genTrainers[idSite].train(batchSrc, batchRef)
        results['genFwd_idLoss'] += (genFwd_idLoss/len(dataset_pairs))
    return results


# Training execution
BEST_GEN_PATH = DEST_DIR_PATH+'/best_genUniv.h5'
record_dict = {}
best_score = None

for epoch in range(1,N_EPOCHS+1):
    tmp_record = {}
    for step in range(1,STEPS_PER_EPOCH+1):
        res = train_step()
        
        if not tmp_record:
            for key in res.keys(): tmp_record[key] = res[key].numpy()
        else:
            for key in res.keys(): tmp_record[key] += res[key].numpy()
        
        log = f"End step {step}/{STEPS_PER_EPOCH} époque {epoch}/{N_EPOCHS} | "
        for k in sorted(res.keys()): log += f"{k} = {res[k]:.4f},  "
        print(log, end=f"{' '*20}\r")
        
    if not record_dict:
        for key in sorted(tmp_record.keys()):
            record_dict[key] = [tmp_record[key]/STEPS_PER_EPOCH]
    else:
        for key in tmp_record:
            record_dict[key].append(tmp_record[key]/STEPS_PER_EPOCH)
            
    log = f"Fin époque {epoch} -> "
    for key,value in record_dict.items():
        log += f"{key} : {value[-1]:.4f}, "
    print(log+' '*20)
    print()
    
    if epoch % EVAL_FREQ == 0:
        score = eval_model()
        print(f"Validation function, score = {score:.3f}")
        if not best_score or score>best_score:
            best_score=score
            gen_univ.save_weights(BEST_GEN_PATH)
            print('New best model saved')
            
            
# Saving of all the models
gen_univ.save_weights(DEST_DIR_PATH+'/generator_univ.h5')
for i in range(len(dataset_pairs)):
    gens_bwd[i].save_weights(f"{DEST_DIR_PATH}/genBwd_{i+1}.h5")
    discs_bwd[i].save_weights(f"{DEST_DIR_PATH}/discBwd_{i+1}.h5")
    discs_ref[i].save_weights(f"{DEST_DIR_PATH}/discRef_{i+1}.h5")
with open(DEST_DIR_PATH+'/stats.json','w') as f: json_dump(record_dict, f)
