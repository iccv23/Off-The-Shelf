# DomainNet
# CUDA_VISIBLE_DEVICES=3 python main.py -hd sketch -sd quickdraw -gd real -nc 4 --T 0.45 --mode debias --feat_dim 64 --hp_lambda 1 --pos_prior 0.15 --queue_begin_epoch -1 --l2_reg 1e-5 -bs 64 -es 15
CUDA_VISIBLE_DEVICES=0 python main.py -hd sketch -sd quickdraw -gd real -nc 2 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd sketch -sd quickdraw -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd sketch -sd quickdraw -gd real -nc 8 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd sketch -sd quickdraw -gd real -nc 16 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32



# quickdraw
CUDA_VISIBLE_DEVICES=0 python main.py -hd quickdraw -sd sketch -gd real -nc 2 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd quickdraw -sd sketch -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd quickdraw -sd sketch -gd real -nc 8 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd quickdraw -sd sketch -gd real -nc 16 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32



#  painting
CUDA_VISIBLE_DEVICES=0 python main.py -hd painting -sd infograph -gd real -nc 2 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=0 python main.py -hd painting -sd infograph -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd painting -sd infograph -gd real -nc 8 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd painting -sd infograph -gd real -nc 16 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32



# infograph
CUDA_VISIBLE_DEVICES=1 python main.py -hd infograph -sd painting -gd real -nc 2 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd infograph -sd painting -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd infograph -sd painting -gd real -nc 8 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd infograph -sd painting -gd real -nc 16 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32


# clipart
CUDA_VISIBLE_DEVICES=1 python main.py -hd clipart -sd painting -gd real -nc 2 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd clipart -sd painting -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd clipart -sd painting -gd real -nc 8 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
CUDA_VISIBLE_DEVICES=1 python main.py -hd clipart -sd painting -gd real -nc 16 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 30 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 # --num_codebooks 32
# CUDA_VISIBLE_DEVICES=0 python test.py -hd sketch -sd quickdraw -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 24 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1 --start_lr 1e-3 --final_lr 1e-6 -lr 1e-3 -alpha 0.1 --num_codebooks 32
# CUDA_VISIBLE_DEVICES=2 python test.py -hd sketch -sd quickdraw -gd real -nc 4 --mode ce --feat_dim 64 --l2_reg 1e-5 -bs 128 -es 15 -opt adam --hp_gamma 0.1 --hp_lambda 0.1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -hd quickdraw -sd sketch -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 2 -bs 60 -mixl img -es 15
# CUDA_VISIBLE_DEVICES=0 python3 main.py -hd painting -sd infograph -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15
# CUDA_VISIBLE_DEVICES=0 python3 main.py -hd infograph -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15
# CUDA_VISIBLE_DEVICES=0 python3 main.py -hd clipart -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15



# # Sketchy
# CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 2 -wrat 0.5 -data Sketchy -bs 64 -mixl img -es 15 -eccv 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 1 -wrat 1 -data Sketchy -bs 64 -mixl img -es 15 -eccv 0

# # TUBerlin
# CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 1 -wrat 1 -data TUBerlin -bs 64 -mixl img -es 15
