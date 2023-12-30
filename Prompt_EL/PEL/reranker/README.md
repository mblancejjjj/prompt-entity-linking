# Biomedical-Entity-Linking

## Environment setup
Clone the repository and set up the environment via "requirements.txt". Here we use python3.6. 
```
pip install -r requirements.txt
```

the code structure:

```
.
├── check_eval_res.py
├── checkpoints
│   └── best_model_state.bin
├── data
│   ├── predictions_eval.json
│   ├── predictions.json
│   ├── predictions_train.json
│   └── predict_scores.txt
├── data.py
├── model.py
├── post_process.py
├── predict.py
├── README.md
├── requirements.txt
├── src
│   └── biosyn
│       └── biosyn.py
├── test.sh
└── train_ranker.py
```

## train

```
python train_ranker.py
```

## testing

```
python predict.py
```

## evaluating

```
python post_process.py
python check_eval_res.py
```

