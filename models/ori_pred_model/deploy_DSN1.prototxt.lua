require 'cudnn'
local model = {}
-- warning: module 'input [type Input]' not found
table.insert(model, {'conv1_1', cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 35, 35, 1)})
table.insert(model, {'relu1_1', cudnn.ReLU(true)})
table.insert(model, {'conv1_2', cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)})
table.insert(model, {'relu1_2', cudnn.ReLU(true)})
table.insert(model, {'score-dsn1_', cudnn.SpatialConvolution(64, 8, 1, 1, 1, 1, 0, 0, 1)})
return model