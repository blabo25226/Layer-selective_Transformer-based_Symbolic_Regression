# experiment on the not_included dataset using NeSymReS

cd ../../

python scripts/exp.py \
    testing.model=nopow_original \
    testing.experiment_mode=vanilla \
    testing.test_set=nopow \
    testing.left=0 \
    testing.right=0 \
    testing.num_loops=1 \
    testing.beam_size=5 \
    result_options.save_results=True