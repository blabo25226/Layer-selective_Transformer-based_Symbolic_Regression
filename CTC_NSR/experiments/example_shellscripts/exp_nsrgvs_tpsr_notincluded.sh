# experiment on the not_included dataset using NSR-gvs+TPSR, this may take over 3 hours to complete for one expression

cd ../../

python scripts/exp.py \
    testing.model=nopow_prepend_positives \
    testing.experiment_mode=vanilla \
    testing.test_set=nopow \
    testing.left=0 \
    testing.right=0 \
    testing.num_loops=30 \
    testing.tpsr=True \
    tpsr_params.num_beams=1 \
    result_options.save_results=True