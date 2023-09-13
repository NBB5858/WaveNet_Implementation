## WaveNet Implementation

This is a TensorFlow implementation of the WaveNet model first introduced in 
[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
The architecture for the model is based on appendix A of 
[Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1609.03499). 
The scripts within are tested on TensorFlow 2.13.0 and Python 3.9.6.

The weights for a demo model are saved at *Saved_Models* >> *Demo_Model*. This model was 
trained on ~60 minutes of Bach's *Well-Tempered Klavier*.  A demo result for this model is 
located at *Generated_Samples* >> *demo_sample.wav*.

## Training
To train the network, 

1) Place the training set into the 
*Training_Data* folder and the validation set in to the 
*Validation_Data* folder.  Training & validation data should be 
.wav files sampled at 8kHz.


2) In *params.json*, define the network parameters. The defaults are those
used in the demo model.


3) Navigate to *train.py*. Define the number of epochs, *n_epochs*. If this is the first 
run at training the network, set *load_from_weights = False*.  The training script will save 
the weights for the most recent model at *Saved_Models* >> *Last_Model*. It will save the
weights from the best model, i.e., with the lowest validation loss, at *Saved_Models* >> *Best_Model*.  Training metrics
are saved in the *Metrics* folder. Upon subsequent training runs, set *load_from_weights = True*. This 
will load last set of model weights, so training can be resumed where it was left off.



## Audio Generation

To generate audio, navigate to the *generate.py* script. Choose the number of samples 
to generate *num_generate*. Choose which validation file, *val_file*, to use as a seed. Choose the start and end range for the seed,
*start* and *end*. 


