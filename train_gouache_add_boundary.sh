DATAFILE="coco_train_val.h5"

STYLE_IMAGE="gouache_house.jpg"
STYLE_IMAGE_SIZE="384"

CONTENT_WEIGHTS="1.0"
STYLE_WEIGHTS="50.0"
BOUNDARY_WEIGHTS="5.0"

BOUNDARY_LAYERS="24"

th train_with_boundary_loss.lua \
  -h5_file $DATAFILE \
  -style_image "images/styles/${STYLE_IMAGE}" \
  -style_image_size $STYLE_IMAGE_SIZE \
  -content_weights $CONTENT_WEIGHTS \
  -style_weights $STYLE_WEIGHTS \
  -boundary_weights $BOUNDARY_WEIGHTS \
  -boundary_layers $BOUNDARY_LAYERS \
  -checkpoint_name "checkpoint/checkpoint_boundary_loss_${DATAFILE}_${STYLE_IMAGE}_${STYLE_IMAGE_SIZE}_${CONTENT_WEIGHTS}_${STYLE_WEIGHTS}_${BOUNDARY_WEIGHTS}_${BOUNDARY_LAYERS}" \
  -gpu 0
