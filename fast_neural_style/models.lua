require 'nn'
require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'


local M = {}

local function build_conv_block(dim_in, dim_out, padding_type, use_instance_norm, stride)
  local conv_block = nn.Sequential()
  conv_block:add(nn.SpatialConvolution(dim_in, dim_in, 3, 3, 1, 1, 1, 1))
  conv_block:add(nn.InstanceNormalization(dim_in))
  conv_block:add(nn.ReLU(true))
  if stride == 2 then
    conv_block:add(nn.SpatialConvolution(dim_in, dim_out, 3, 3, 2, 2, 1, 1))
    conv_block:add(nn.InstanceNormalization(dim_out))
    conv_block:add(nn.ReLU(true))
  end
  conv_block:add(nn.SpatialConvolution(dim_out, dim_out, 3, 3, 1, 1, 1, 1))
  conv_block:add(nn.InstanceNormalization(dim_out))
  return conv_block
end

local function build_res_block(dim_in, dim_out, padding_type, use_instance_norm, stride)
  local conv_block = build_conv_block(dim_in, dim_out, padding_type, use_instance_norm, stride)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)

  if stride == 2 then
    concat:add(nn.SpatialConvolution(dim_in, dim_out, 3, 3, 2, 2, 1, 1))
  else
    concat:add(nn.SpatialConvolution(dim_in, dim_out, 3, 3, 1, 1, 1, 1))
  end
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end

-- function M.build_model(opt)
--   local model = nn.Sequential()
--   model:add(nn.SpatialConvolution(3, 32, 9, 9, 1, 1, 4, 4))
--   model:add(nn.ReLU(true))
--   model:add(nn.InstanceNormalization(32))

--   layer = build_res_block(128, opt.padding_type, opt.use_instance_norm)

--   seq4 = nn.Sequential()
--   seq4:add(layer):add(nn.ReLU(true)):add(layer):add(nn.ReLU(true)):add(layer):add(nn.ReLU(true))

--   seq4_ = nn.Sequential()
--   seq4_:add(nn.SpatialConvolution(128, 128, 1, 1, 1, 1))
--   seq4_:add(nn.ReLU(true))

--   bch3 = nn.Concat(2)
--   bch3:add(seq4)
--   bch3:add(seq4_)

--   seq3 = nn.Sequential()
--   seq3:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, 1, 1))
--   seq3:add(nn.ReLU(true))
--   seq3:add(nn.InstanceNormalization(128))
--   seq3:add(bch3)
--   seq3:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, 1, 1))
--   seq3:add(nn.ReLU(true))  
--   seq3:add(nn.SpatialFullConvolution(64, 64, 3, 3, 2, 2, 1, 1, 1, 1))

--   seq3_ = nn.Sequential()
--   seq3_:add(nn.SpatialConvolution(64, 64, 1, 1, 1, 1))
--   seq3_:add(nn.ReLU(true))

--   bch2 = nn.Concat(2)
--   bch2:add(seq3)
--   bch2:add(seq3_)

--   seq2 = nn.Sequential()
--   seq2:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2, 1, 1))
--   seq2:add(nn.ReLU(true))
--   seq2:add(nn.InstanceNormalization(64))
--   seq2:add(bch2)
--   seq2:add(nn.SpatialConvolution(128, 32, 3, 3, 1, 1, 1, 1))
--   seq2:add(nn.ReLU(true))  
--   seq2:add(nn.SpatialFullConvolution(32, 32, 3, 3, 2, 2, 1, 1, 1, 1))

--   seq2_ = nn.Sequential()
--   seq2_:add(nn.SpatialConvolution(32, 32, 1, 1, 1, 1))
--   seq2_:add(nn.ReLU(true))

--   bch1 = nn.Concat(2)
--   bch1:add(seq2)
--   bch1:add(seq2_)

--   model:add(bch1)
--   model:add(nn.SpatialConvolution(64, 3, 9, 9, 1, 1, 4, 4))
--   model:add(nn.Tanh())
--   model:add(nn.MulConstant(opt.tanh_constant))
--   model:add(nn.TotalVariation(opt.tv_strength))

--   return model

-- end

function M.build_model(opt)
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(3, 32, 9, 9, 1, 1, 4, 4))
  model:add(nn.ReLU(true))
  model:add(nn.InstanceNormalization(32))

  seq4 = nn.Sequential()
  seq4:add(build_res_block(128, 128, opt.padding_type, opt.use_instance_norm, 1))
  seq4:add(nn.ReLU(true))
  seq4:add(build_res_block(128, 128, opt.padding_type, opt.use_instance_norm, 1))
  seq4:add(nn.ReLU(true))
  seq4:add(build_res_block(128, 128, opt.padding_type, opt.use_instance_norm, 1))
  seq4:add(nn.ReLU(true))    

  seq4_ = nn.Sequential()
  seq4_:add(nn.SpatialConvolution(128, 128, 1, 1, 1, 1))
  seq4_:add(nn.ReLU(true))

  bch3 = nn.Concat(2)
  bch3:add(seq4)
  bch3:add(seq4_)

  seq3 = nn.Sequential()
  seq3:add(build_res_block(64, 128, opt.padding_type, opt.use_instance_norm, 2))
  seq3:add(nn.ReLU(true))
  seq3:add(bch3)
  seq3:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, 1, 1))
  seq3:add(nn.ReLU(true))  
  seq3:add(nn.SpatialFullConvolution(64, 64, 3, 3, 2, 2, 1, 1, 1, 1))

  seq3_ = nn.Sequential()
  seq3_:add(nn.SpatialConvolution(64, 64, 1, 1, 1, 1))
  seq3_:add(nn.ReLU(true))

  bch2 = nn.Concat(2)
  bch2:add(seq3)
  bch2:add(seq3_)

  seq2 = nn.Sequential()
  seq2:add(build_res_block(32, 64, opt.padding_type, opt.use_instance_norm, 2))
  seq2:add(nn.ReLU(true))
  seq2:add(bch2)
  seq2:add(nn.SpatialConvolution(128, 32, 3, 3, 1, 1, 1, 1))
  seq2:add(nn.ReLU(true))  
  seq2:add(nn.SpatialFullConvolution(32, 32, 3, 3, 2, 2, 1, 1, 1, 1))

  seq2_ = nn.Sequential()
  seq2_:add(nn.SpatialConvolution(32, 32, 1, 1, 1, 1))
  seq2_:add(nn.ReLU(true))

  bch1 = nn.Concat(2)
  bch1:add(seq2)
  bch1:add(seq2_)

  model:add(bch1)
  model:add(nn.SpatialConvolution(64, 3, 9, 9, 1, 1, 4, 4))
  model:add(nn.Tanh())
  model:add(nn.MulConstant(opt.tanh_constant))
  model:add(nn.TotalVariation(opt.tv_strength))

  return model

end

return M