# experiment on the not_included dataset using NSRwH, providing the model with full prior knowledge

cd ../../

python scripts/exp.py \
    testing.model=nopow_finetuned \
    testing.experiment_mode=all \
    testing.test_set=nopow \
    testing.left=0 \
    testing.right=0 \
    testing.num_loops=1 \
    testing.beam_size=5 \
    result_options.save_results=True




