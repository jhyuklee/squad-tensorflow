SQuAD-tensorflow
====================================================================
Implementation of several models for SQuAD
- Basic
- Multi-Perspective Context Matching (IBM)
- TBD

## Requirements
* Python 3.4.3 or higher
* Tensorflow 1.0.1 (GPU enabled)
* Cuda settings
* Gensim

## Downloads
* ./data/ : SQuAD dataset
* ~/embed\_data : glove pretrained file

## Run the code
```bash
# model: b (Basic), m (MPCM)
# debug: shows prediction process
# test: only runs few initial iterations
$ python main --model m --debug True --test False
```
