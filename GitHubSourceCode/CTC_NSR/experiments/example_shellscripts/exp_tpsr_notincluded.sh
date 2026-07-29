# experiment on the not_included dataset using NeSymReS+TPSR

cd ../../

python scripts/exp.py \
    testing.model=nopow_original \
    testing.experiment_mode=vanilla \
    testing.test_set=nopow \
    testing.left=0 \
    testing.right=0 \
    testing.tpsr=True \
    testing.num_loops=1 \
    tpsr_params.num_beams=1 \
    result_options.save_results=True