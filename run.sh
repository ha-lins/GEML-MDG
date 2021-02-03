# fd means: fudan's CMDD dataset
# cy means: ours chunyu dataset
# the number following dis means: the id of target disease
# notice that there are three modes: train / fine-tune / evaluate

# Pretrain-Only setting (Multi-task learning): train + evaluate directly
# Fine-tuning setting: train + fine-tuning + evaluate
# Meta-learning setting: meta-train (w/o graph-evolving) + fine-tuning + evaluate
# GEML setting: meta-train + fine-tuning + evaluate

# geml: our method
# kgr: our ablated method (w/o graph-evolving)
# nkd: NKD
# poks: POKS
# Follwing commands are what we used mostly:

CUDA_VISIBLE_DEVICES=1 allennlp train   -s /data3/linshuai/gen/save_fd/geml/geml_hidden_dis17_test1 \
  our.json \
  --include-package our_geml_fd

CUDA_VISIBLE_DEVICES=1 allennlp train   -s /data3/linshuai/gen/save_fd/geml/geml_hidden_test1 \
  our.json \
  --include-package our_geml_cy

CUDA_VISIBLE_DEVICES=3 allennlp train   -s /data3/linshuai/gen/save_fd/pretrain/nkd_dis9_test1 \
  NKD.json \
  --include-package NKD

#meta
CUDA_VISIBLE_DEVICES=5 allennlp train   -s /data3/linshuai/gen/save_fd/meta/kgr_dis17_test1 \
  our.json \
  --include-package our_meta_fd

CUDA_VISIBLE_DEVICES=5 allennlp train   -s /data3/linshuai/gen/save_fd/meta/kgr_dis17_test1 \
  our.json \
  --include-package our_meta_fd

CUDA_VISIBLE_DEVICES=4 allennlp train   -s /data3/linshuai/gen/save_fd/meta/poks_dis9_test1 \
  poks.json \
  --include-package poks


CUDA_VISIBLE_DEVICES=5 allennlp train   -s /data3/linshuai/gen/save_fd/geml/geml_dis9_test1 \
  our.json \
  --include-package our_old_fd

CUDA_VISIBLE_DEVICES=4 allennlp evaluate /data3/linshuai/gen/save/ours_ft_fulldata1_dis73_0.1 \
  cy/dis_pk_dir/test_cy_73.pk \
  --include-package  our_old\
  --output-file /data3/linshuai/gen/save/ours_ft_fulldata1_dis73_0.1.txt

CUDA_VISIBLE_DEVICES=4 allennlp fine-tune   -m /data3/linshuai/gen/save/ours_fulldata_1 \
 -c our_ft.json   \
 --include-package our_old   \
 -s /data3/linshuai/gen/save/ours_ft_fulldata1_dis73_0.1

 CUDA_VISIBLE_DEVICES=3 allennlp evaluate  /data3/linshuai/gen/save/ft/geml_lr0.01_3 \
  cy/dis_pk_dir/meta_test/test_cy_52.pk \
  --include-package our_eval \
  --output-file /data3/linshuai/gen/save/ft/geml_ft_lr0.01_3

 CUDA_VISIBLE_DEVICES=3 allennlp evaluate  /data3/linshuai/gen/save/pretrain/poks_dis9_test1 \
  cy/dis_pk_dir/meta_test/test_cy_46.pk \
  --include-package poks \
  --output-file /data3/linshuai/gen/save/ztpoks_zt_pret_dis46_1

 CUDA_VISIBLE_DEVICES=3 allennlp evaluate  /data3/linshuai/gen/save/ft/kgr_meta_dis46_4 \
  cy/dis_pk_dir/meta_test/test_cy_46.pk \
  --include-package our_eval \
  --output-file /data3/linshuai/gen/save/ft/kgr_ft_meta_dis46_4

 CUDA_VISIBLE_DEVICES=3 allennlp fine-tune  -m /data3/linshuai/gen/save/meta/kgr_test1 \
 -c our_ft.json   \
 --include-package our_old \
 -s /data3/linshuai/gen/save/ft/kgr_meta_dis46_4

 CUDA_VISIBLE_DEVICES=0 allennlp fine-tune   -m /data3/linshuai/gen/save/geml/geml_hidden_1 \
 -c our_ft.json   \
 --include-package our_old \
 -s /data3/linshuai/gen/save/ft/geml_dis52_1

 CUDA_VISIBLE_DEVICES=1 allennlp evaluate  /data3/linshuai/gen/save/geml/geml5 \
  cy/dis_pk_dir/meta_test/test_cy_73.pk \
  --include-package our_eval \
  --output-file /data3/linshuai/gen/save/zt/geml5

 CUDA_VISIBLE_DEVICES=1 allennlp evaluate  /data3/linshuai/gen/save/discussion/geml_dis52_30_1 \
  cy/dis_pk_dir/meta_test/test_cy_52.pk \
  --include-package our_eval \
  --output-file /data3/linshuai/gen/save/zt/geml_dis52_30_1

  CUDA_VISIBLE_DEVICES=0 allennlp train   -s /data3/linshuai/gen/save/geml/geml_hidden_1 \
  our.json \
  --include-package our_geml_cy


  #zs
CUDA_VISIBLE_DEVICES=1 allennlp evaluate  /data3/linshuai/gen/save_fd/discussion/geml_hidden_dis9_100_test2 \
    fd/dis_pk_dir/meta_test/test_fd_9.pk \
    --include-package our_geml_eval_fd \
    --output-file /data3/linshuai/gen/save_fd/discussion/geml_hidden_dis9_100_test2_human

CUDA_VISIBLE_DEVICES=6 allennlp evaluate  /data3/linshuai/gen/save/discussion/geml_hidden_dis9_50_test2 \
    cy/dis_pk_dir/meta_test/test_cy_73.pk \
    --include-package poks_cy \
    --output-file /data3/linshuai/gen/save/pret/poks_pret_eval_dis73_1


 CUDA_VISIBLE_DEVICES=0 allennlp fine-tune   -m /data3/linshuai/gen/save_fd/geml/geml_hidden_dis9_test2 \
 -c our_ft.json   \
 --include-package our_geml_fd \
 -s /data3/linshuai/gen/save_fd/discussion/geml_hidden_dis9_100_test2


 CUDA_VISIBLE_DEVICES=1 allennlp fine-tune   -m /data3/linshuai/gen/save_fd/meta/kgr_dis9_test1 \
 -c our_ft.json   \
 --include-package our_old_fd \
 -s /data3/linshuai/gen/save_fd/ft/kgr_pret_dis9_test1

 CUDA_VISIBLE_DEVICES=0 allennlp fine-tune   -m /data3/linshuai/gen/save_fd/geml/geml_dis9_test1 \
 -c our_ft.json   \
 --include-package our_geml_fd \
 -s /data3/linshuai/gen/save_fd/zt/geml_hidden_dis9_test2

 CUDA_VISIBLE_DEVICES=6 allennlp fine-tune   -m /data3/linshuai/gen/save_fd/pretrain/kgr_dis9_test1 \
 -c our_ft.json   \
 --include-package poks \
 -s /data3/linshuai/gen/save_fd/ft/poks_meta_dis9_test1

 CUDA_VISIBLE_DEVICES=7 allennlp evaluate  /data3/linshuai/gen/save_fd/ft/poks_meta_dis17_test1 \
    fd/dis_pk_dir/meta_test_old/test_fd_17.pk \
    --include-package poks \
    --output-file /data3/linshuai/gen/save_fd/ft/poks_meta_eval_dis17_test1