# experiment on the aif dataset using NSR-gvs

cd ../../

python scripts/exp.py \
    testing.model=prepend_positives \
    testing.experiment_mode=vanilla \
    testing.test_set=aif \
    testing.left=0 \
    testing.right=0 \
    testing.num_loops=30 \
    testing.beam_size=5 \
    result_options.save_results=True