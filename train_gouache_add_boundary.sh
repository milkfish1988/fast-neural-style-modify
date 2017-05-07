DATAFILE="coco_train_with_style_idx_coco_10.h5"
STYLE_DIR="/home/wzhang2/Colorsketch/fast-neural-style/images/styles/"

STYLE_IMAGE="style_image_10/"
STYLE_IMAGE_SIZE="384"
STYLE_LAYERS='16,23'

CONTENT_WEIGHTS="1.0"
STYLE_WEIGHTS="20.0"
BOUNDARY_WEIGHTS="5000.0"
ORI_WEIGHTS="0.0"

CONTENT_LAYERS='21'
BOUNDARY_LAYERS="24"
ORI_LAYERS='1'

NUM_ITERATIONS='240000'
LR_DECAY_EVERY='240000'

ARCH='c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3'

th train_with_boundary_loss.lua \
  -h5_file $DATAFILE \
  -style_image  $STYLE_DIR${STYLE_IMAGE} \
  -style_image_size $STYLE_IMAGE_SIZE \
  -style_layers $STYLE_LAYERS \
  -content_weights $CONTENT_WEIGHTS \
  -content_layers $CONTENT_LAYERS \
  -style_weights $STYLE_WEIGHTS \
  -boundary_weights $BOUNDARY_WEIGHTS \
  -boundary_layers $BOUNDARY_LAYERS \
  -ori_weights $ORI_WEIGHTS \
  -ori_layers $ORI_LAYERS \
  -num_iterations $NUM_ITERATIONS \
  -lr_decay_every $LR_DECAY_EVERY \
  -arch $ARCH \
  -checkpoint_name \
  "checkpoint/checkpoint_boundary_loss_${DATAFILE}_${STYLE_IMAGE}_${STYLE_IMAGE_SIZE}_${STYLE_LAYERS}_${STYLE_WEIGHTS}_${CONTENT_LAYERS}_${CONTENT_WEIGHTS}_${BOUNDARY_LAYERS}_${BOUNDARY_WEIGHTS}_${ORI_LAYERS}_${ORI_WEIGHTS}_${NUM_ITERATIONS}_${LR_DECAY_EVERY}_${ARCH}" \
  -gpu 0
