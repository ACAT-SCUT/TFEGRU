model_name=TFEGRU

seq_len=5
pred_len=1
random_seed=41

timestamp=$(date +%s)  # getting current timestamp
formatted_datetime=$(date -d @"$timestamp" "+%Y-%m-%d_%H-%M-%S")

root_path='./dataset/'
dataset='google'
result_folder="./results/google/${formatted_datetime}_${random_seed}_TFEGRU"

train_size=0.6
val_size=0.2

batch_size=128
learning_rate=0.005
dropout=0.5
hidden_size=128
patience=5
decay_rate=0.9
num_epochs=100

save_data=true

python -u run.py \
--random_seed $random_seed \
--seq_len $seq_len \
--pred_len $pred_len \
--root_path $root_path \
--result_folder "$result_folder" \
--model $model_name \
--dataset $dataset \
--counts 100 \
--train_size $train_size \
--val_size $val_size \
--batch_size $batch_size \
--learning_rate $learning_rate \
--dropout $dropout \
--hidden_size $hidden_size \
--patience $patience \
--decay_rate $decay_rate \
--num_epochs $num_epochs \
--save_data $save_data \

