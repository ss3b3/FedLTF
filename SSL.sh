python main.py --dataset cifar10 \
    --partition iid \
    --num_clients 10 \
    --balance \
    --noise_type sym \
    --noise_ratio 0.7 \
    --seed 1 \
    --global_rounds 1 \
    --local_epochs 5 \
    --model resnet18 \
    --learning_rate 0.032 \
    --batch_size 128 \
    --SSL_method byol \
    --aggregate_target_encoder True \
    --algorithm FedSSL \
    --encoder_network resnet18 \
    --goal test_FedSSL_byol_resnet18