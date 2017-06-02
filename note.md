```bash
export PYTHONPATH="/home/dairui/workspace/neural-sentiment/:$PYTHONPATH"
python -u train.py > /tmp/ns.log 2>&1 &
tensorboard --logdir=/tmp/tb_logs
rm -rf /tmp/tb_logs


###### 2lstm # 0.875
python train.py --embedding_dims=50 --num_layers=2 --keep_prob=0.5 --use_gru=False \
    --fact_size=0 --learning_rate=0.009 --batch_size=200 --lr_decay=0.7 \
    --max_epoch=20 > /tmp/2lstm.log 2>&1 &

###### 2gru # 0.511
python train.py --embedding_dims=50 --num_layers=2 --keep_prob=0.5 --use_gru=True \
    --fact_size=0 --learning_rate=0.007 --batch_size=200 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/2gru.log 2>&1 &

###### 1lstm # 0.825
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=0 --learning_rate=0.03 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1lstm.log 2>&1 &

# 0.637
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=0 --learning_rate=0.03 --batch_size=50 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1lstm.log 2>&1 &    # new run

###### 1gru # 0.502
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=True \
    --fact_size=0 --learning_rate=0.01 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1gru.log 2>&1 &

# 0.5
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=True \
    --fact_size=0 --learning_rate=0.002 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1gru.log 2>&1 &

#
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=True \
    --fact_size=0 --learning_rate=0.00003 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1gru.log 2>&1 &

# 2flstm # nan
python train.py --embedding_dims=50 --num_layers=2 --keep_prob=0.5 --use_gru=False \
    --fact_size=40 --learning_rate=0.002 --batch_size=50 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/2flstm.log 2>&1 &

###### 1flstm # 0.50
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=50 --learning_rate=0.002 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1flstm.log 2>&1 &

# nan after 2 epoch
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=50 --learning_rate=0.002 --batch_size=200 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1flstm.log 2>&1 &

# nan
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=50 --learning_rate=0.0005 --batch_size=150 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1flstm.log 2>&1 &

# 0.502
python train.py --embedding_dims=60 --num_layers=1 --keep_prob=0.5 --use_gru=False \
    --fact_size=50 --learning_rate=0.0005 --batch_size=100 --lr_decay=0.7 \
    --max_epoch=50 > /tmp/1flstm.log 2>&1 &
```
