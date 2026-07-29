# train a nsrgvs model by specifying training data and validation data

cd ../../

python scripts/train.py \
       architecture.conditioning=False \
       dataset.conditioning.mode="all" \
       train_path= \
       benchmark_path= \
       base_model=None \
       prepend_conditioning=True \
       architecture.length_eq=200 \
       architecture.number_possible_tokens=200 \
       dataset.conditioning.prob_symmetry=0 \
       dataset.conditioning.prob_complexity=0 \
       dataset.conditioning.positive.prob=0.5 \
       dataset.conditioning.positive.prob_pointers=0 \
       dataset.conditioning.negative.prob=0 \