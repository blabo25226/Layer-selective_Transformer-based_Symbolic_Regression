# finetune a NeSymReS model to obtain an NSRwH model
# it is also possible to train an NSRwH model from scratch

cd ../../

python scripts/train.py \
       architecture.conditioning=dec \
       dataset.conditioning.mode=True \
       train_path= \
       benchmark_path= \
       base_model=nopow \
       dataset.conditioning.prob_symmetry=0.4 \
       dataset.conditioning.prob_complexity=0.6 \
       dataset.conditioning.positive.prob=0.6 \
       dataset.conditioning.positive.prob_pointers=0.3 \
       dataset.conditioning.negative.prob=0.6 \