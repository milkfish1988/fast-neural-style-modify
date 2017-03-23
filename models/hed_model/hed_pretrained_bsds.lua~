require 'nngraph'

modmap = {}

data = nn.Identity()()
modmap[#modmap+1] = {data}

conv1_1 = nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 35, 35)(data)
modmap[#modmap+1] = {conv1_1}

relu1_1 = nn.ReLU(true)(conv1_1)
modmap[#modmap+1] = {relu1_1}

conv1_2 = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(relu1_1)
modmap[#modmap+1] = {conv1_2}

relu1_2 = nn.ReLU(true)(conv1_2)
modmap[#modmap+1] = {relu1_2}

pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(relu1_2)
modmap[#modmap+1] = {pool1}

conv2_1 = nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(pool1)
modmap[#modmap+1] = {conv2_1}

relu2_1 = nn.ReLU(true)(conv2_1)
modmap[#modmap+1] = {relu2_1}

conv2_2 = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(relu2_1)
modmap[#modmap+1] = {conv2_2}

relu2_2 = nn.ReLU(true)(conv2_2)
modmap[#modmap+1] = {relu2_2}

pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(relu2_2)
modmap[#modmap+1] = {pool2}

conv3_1 = nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(pool2)
modmap[#modmap+1] = {conv3_1}

relu3_1 = nn.ReLU(true)(conv3_1)
modmap[#modmap+1] = {relu3_1}

conv3_2 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_1)
modmap[#modmap+1] = {conv3_2}

relu3_2 = nn.ReLU(true)(conv3_2)
modmap[#modmap+1] = {relu3_2}

conv3_3 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_2)
modmap[#modmap+1] = {conv3_3}

relu3_3 = nn.ReLU(true)(conv3_3)
modmap[#modmap+1] = {relu3_3}

pool3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(relu3_3)
modmap[#modmap+1] = {pool3}

conv4_1 = nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(pool3)
modmap[#modmap+1] = {conv4_1}

relu4_1 = nn.ReLU(true)(conv4_1)
modmap[#modmap+1] = {relu4_1}

conv4_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_1)
modmap[#modmap+1] = {conv4_2}

relu4_2 = nn.ReLU(true)(conv4_2)
modmap[#modmap+1] = {relu4_2}

conv4_3 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_2)
modmap[#modmap+1] = {conv4_3}

relu4_3 = nn.ReLU(true)(conv4_3)
modmap[#modmap+1] = {relu4_3}

pool4 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()(relu4_3)
modmap[#modmap+1] = {pool4}

conv5_1 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(pool4)
modmap[#modmap+1] = {conv5_1}

relu5_1 = nn.ReLU(true)(conv5_1)
modmap[#modmap+1] = {relu5_1}

conv5_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_1)
modmap[#modmap+1] = {conv5_2}

relu5_2 = nn.ReLU(true)(conv5_2)
modmap[#modmap+1] = {relu5_2}

conv5_3 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_2)
modmap[#modmap+1] = {conv5_3}

relu5_3 = nn.ReLU(true)(conv5_3)
modmap[#modmap+1] = {relu5_3}

score-dsn1 = nn.SpatialConvolution(64, 1, 1, 1, 1, 1, 0, 0)(relu1_2)
modmap[#modmap+1] = {score-dsn1}

crop = nn.Identity()() -- (Crop)
modmap[#modmap+1] = {crop}

 = nn.Sigmoid()(crop)
modmap[#modmap+1] = {}

score-dsn2 = nn.SpatialConvolution(128, 1, 1, 1, 1, 1, 0, 0)(relu2_2)
modmap[#modmap+1] = {score-dsn2}

upsample_2 = nn.Identity()() -- (Deconvolution)
modmap[#modmap+1] = {upsample_2}

crop = nn.Identity()() -- (Crop)
modmap[#modmap+1] = {crop}

 = nn.Sigmoid()(crop)
modmap[#modmap+1] = {}

score-dsn3 = nn.SpatialConvolution(256, 1, 1, 1, 1, 1, 0, 0)(relu3_3)
modmap[#modmap+1] = {score-dsn3}

upsample_4 = nn.Identity()() -- (Deconvolution)
modmap[#modmap+1] = {upsample_4}

crop = nn.Identity()() -- (Crop)
modmap[#modmap+1] = {crop}

 = nn.Sigmoid()(crop)
modmap[#modmap+1] = {}

score-dsn4 = nn.SpatialConvolution(512, 1, 1, 1, 1, 1, 0, 0)(relu4_3)
modmap[#modmap+1] = {score-dsn4}

upsample_8 = nn.Identity()() -- (Deconvolution)
modmap[#modmap+1] = {upsample_8}

crop = nn.Identity()() -- (Crop)
modmap[#modmap+1] = {crop}

 = nn.Sigmoid()(crop)
modmap[#modmap+1] = {}

score-dsn5 = nn.SpatialConvolution(512, 1, 1, 1, 1, 1, 0, 0)(relu5_3)
modmap[#modmap+1] = {score-dsn5}

upsample_16 = nn.Identity()() -- (Deconvolution)
modmap[#modmap+1] = {upsample_16}

crop = nn.Identity()() -- (Crop)
modmap[#modmap+1] = {crop}

 = nn.Sigmoid()(crop)
modmap[#modmap+1] = {}

concat = nn.JoinTable(1, 3)({crop, crop, crop, crop, crop})
modmap[#modmap+1] = {concat}

new-score-weighting = nn.SpatialConvolution(5, 1, 1, 1, 1, 1, 0, 0)(concat)
modmap[#modmap+1] = {new-score-weighting}

 = nn.Sigmoid()(new-score-weighting)
modmap[#modmap+1] = {}

model = nn.gModule({data}, {fuse_loss, dsn5_loss, dsn4_loss, dsn3_loss, dsn2_loss, dsn1_loss})

return model, modmap
