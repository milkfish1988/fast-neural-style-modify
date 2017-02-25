DATAFILE="coco_train_val.h5"

STYLE_IMAGE="gouache_house.jpg"
STYLE_IMAGE_SIZE="384"

CONTENT_WEIGHTS="1.0"
STYLE_WEIGHTS="100.0"

th train.lua \
  -h5_file $DATAFILE \
  -style_image "images/styles/${STYLE_IMAGE}" \
  -style_image_size $STYLE_IMAGE_SIZE \
  -content_weights $CONTENT_WEIGHTS \
  -style_weights $STYLE_WEIGHTS \
  -checkpoint_name "checkpoint/checkpoint_${DATAFILE}_${STYLE_IMAGE}_${STYLE_IMAGE_SIZE}_${CONTENT_WEIGHTS}_${STYLE_WEIGHTS}" \
  -gpu 0