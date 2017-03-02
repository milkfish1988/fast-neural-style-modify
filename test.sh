MODEL="checkpoint_boundary_loss_coco_train_val.h5_gouache_house.jpg_384_1.0_50.0_5.0_24.t7"

th fast_neural_style.lua \
  -model "checkpoint/${MODEL}" \
  -input_dir images/content/ \
  -output_dir "out/${MODEL}" \
  -gpu 1