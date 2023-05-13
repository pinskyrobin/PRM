# PRM
Code for the paper "Personalized Re-ranking for Recommendation"

## Dataset
the toy data for training/validation/test

- `rec_train_set.sample.data`-->8000 lines
- `rec_validation_set.sample.data`-->1000 lines
- `rec_test_set.sample.data`-->1000 lines


## Training
### Without Personalized Vector

```bash
python exec.py \
        --train true \
        --train_set dataset/rec_train_set.sample.txt \
        --validation_set dataset/rec_validation_set.sample.txt \
        --batch_size 512 \
        --train_epochs 100 \
        --train_steps_per_epoch 1000 \
        --validation_steps 2000 \
        --early_stop_patience 10 \
        --lr_per_step 4000 \
        --d_feature 19 \
        --pos_embedding_mode 0 \
        --saved_model_name PRM_no_pv.h5
```

### With Personalized Vector

```bash
python exec.py \
        --train true \
        --train_set dataset/rec_train_set.sample.txt \
        --validation_set dataset/rec_validation_set.sample.txt \
        --batch_size 512 \
        --train_epochs 100 \
        --train_steps_per_epoch 1000 \
        --validation_steps 2000 \
        --early_stop_patience 10 \
        --lr_per_step 4000 \
        --d_feature 19 \
        --saved_model_name PRM_pv.h5
```

### Testing

```bash
python exec.py \
        --test_set dataset/rec_test_set.sample.txt \
        --validation_set dataset/rec_validation_set.sample.txt \
        --batch_size 512 \
        --saved_model_name PRM_pv.h5
```

### Metric Evaluation

```bash
python metric.py dataset/rec_test_set.sample.txt.predict.out
```

