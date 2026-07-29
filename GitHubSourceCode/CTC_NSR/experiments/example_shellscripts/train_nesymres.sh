# train a NeSymReS model by specifying training data and validation data

cd ../../
python scripts/train.py \
    architecture.conditioning=False \
    dataset.conditioning.mode="None" \
    train_path= \
    benchmark_path= \
    base_model=None \