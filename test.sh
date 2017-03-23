# MODEL="checkpoint_boundary_loss_coco_train_val.h5_gouache_house.jpg_384_1.0_21_5.0_5000_24_60000_60000_c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3.t7"
# MODEL="checkpoint_boundary_loss_coco_train_val.h5_6.jpg_384_16,23_1.0_21_5.0_5000_24_60000_60000_c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3.t7"
MODEL="checkpoint_boundary_loss_coco_train_with_style_idx_gb_20.h5_style_image_20_bold/_384_16,23_1.0_21_10.0_1000_24_240000_240000_c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3.t7"
DIR_INPUT='real_35'

th fast_neural_style.lua \
  -model "checkpoint/${MODEL}" \
  -input_dir "images/content/${DIR_INPUT}/" \
  -output_dir "out/${MODEL}_${DIR_INPUT}" \
  -gpu 1