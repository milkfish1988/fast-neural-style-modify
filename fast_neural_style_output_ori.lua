require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
require 'PredOriModel'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


--[[
Use a trained feedforward model to stylize either a single image or an entire
directory of images.
--]]

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'models/instance_norm/candy.t7')
cmd:option('-image_size', 784)
cmd:option('-median_filter', 3)
cmd:option('-timing', 0)

-- Input / output options
cmd:option('-input_image', 'images/content/chicago.jpg')
cmd:option('-output_image', 'out.png')
cmd:option('-input_dir', '')
cmd:option('-output_dir', '')

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)


local function main()
  local opt = cmd:parse(arg)

  if (opt.input_image == '') and (opt.input_dir == '') then
    error('Must give exactly one of -input_image or -input_dir')
  end

  -- load ori loss net
  local loss_ori_net = define_Ori_net()
  loss_ori_net:evaluate()
  loss_ori_net:type(dtype)
  print('****print ori loss net: ')
  print(loss_ori_net)

  -- load ori pred net
  local pred_ori_net = loadcaffe.load('/home/wzhang2/Colorsketch/fast-neural-style/models/ori_pred_model/deploy_DSN1.prototxt', '/home/wzhang2/Colorsketch/fast-neural-style/models/ori_pred_model/hed_ft_ori.caffemodel', 'cudnn')
  pred_ori_net:evaluate()
  pred_ori_net:type(dtype)
  print('****print pred ori net: ')
  print(pred_ori_net)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local ok, checkpoint = pcall(function() return torch.load(opt.model) end)
  if not ok then
    print('ERROR: Could not load model from ' .. opt.model)
    print('You may need to download the pretrained models by running')
    print('bash models/download_style_transfer_models.sh')
    return
  end
  local model = checkpoint.model
  model:evaluate()
  model:type(dtype)
  if use_cudnn then
    cudnn.convert(model, cudnn)
    cudnn.convert(loss_ori_net, cudnn):cuda()
    -- cudnn.convert(pred_ori_net, cudnn):cuda()
    if opt.cudnn_benchmark == 0 then
      cudnn.benchmark = false
      cudnn.fastest = true
    end
  end

  local preprocess_method = checkpoint.opt.preprocessing or 'vgg'
  local preprocess = preprocess[preprocess_method]

  local function run_image(in_path, out_path)
    local img = image.load(in_path, 3)
    if opt.image_size > 0 then
      -- img = image.scale(img, opt.image_size)
      -- img = image.scale(img, opt.image_size, opt.image_size)
      img = image.scale(img, 256, 256)
    end
    local H, W = img:size(2), img:size(3)

    local img_pre = preprocess.preprocess(img:view(1, 3, H, W))
    -- local timer = nil
    -- if opt.timing == 1 then
    --   -- Do an extra forward pass to warm up memory and cuDNN
    --   model:forward(img_pre)
    --   timer = torch.Timer()
    --   if cutorch then cutorch.synchronize() end
    -- end
    -- local img_out = model:forward(img_pre)

    -- -- get ori 8ch predict
    -- if opt.timing == 1 then
    --   loss_ori_net:forward(img_out)
    -- end
    -- local ori_out_8 = loss_ori_net:forward(img_out)
    -- print('ori_out_8: ', ori_out_8:size())
    -- local maxval, ori_pred = torch.max(ori_out_8,2)
    -- ori_pred = torch.div(ori_pred:float(), 8.0)
    -- print('img_pre: ', img_pre:size(), 'ori_pred: ', ori_pred:size())
    -- print('ori_pred: ', torch.min(ori_pred), torch.max(ori_pred))
    -- print('type ', ori_pred:type())
    -- local H, W = ori_pred:size(2), ori_pred:size(3)
    -- local HH, WW = img_pre:size(2), img_pre:size(3)
    -- local xs = (H - HH) / 2
    -- local ys = (W - WW) / 2
    -- ori_pred = ori_pred[{{}, {xs + 1, H - xs}, {ys + 1, W - ys}}][1]

    -- get ori predict
    local timer = nil
    if opt.timing == 1 then
      pred_ori_net:forward(img_pre)
      timer = torch.Timer()
      if cutorch then cutorch.synchronize() end      
    end
    print(img_pre:size())
    local ori_out = pred_ori_net:forward(img_pre:cuda())
    print('ori_out: ', ori_out:size())
    local maxval_org, ori_pred_org = torch.max(ori_out,2)
    ori_pred_org = torch.div(ori_pred_org:float(), 8.0)


    if opt.timing == 1 then
      if cutorch then cutorch.synchronize() end
      local time = timer:time().real
      print(string.format('Image %s (%d x %d) took %f',
            in_path, H, W, time))
    end
    local img_out = preprocess.deprocess(img_out)[1]

    if opt.median_filter > 0 then
      img_out = utils.median_filter(img_out, opt.median_filter)
    end

    print('Writing output image to ' .. out_path)
    local out_dir = paths.dirname(out_path)
    if not path.isdir(out_dir) then
      paths.mkdir(out_dir)
    end
    image.save(out_path, img_out)

    ori_out_path = string.gsub(out_path, '.jpg', '_ori.png')
    print('ori_pred: ', torch.min(ori_pred), torch.max(ori_pred))
    image.save(ori_out_path, ori_pred[1])

    ori_out_org_path = string.gsub(out_path, '.jpg', '_ori_org.png')
    image.save(ori_out_org_path, ori_pred_org[1])
  end


  if opt.input_dir ~= '' then
    if opt.output_dir == '' then
      error('Must give -output_dir with -input_dir')
    end
    for fn in paths.files(opt.input_dir) do
      if utils.is_image_file(fn) then
        local in_path = paths.concat(opt.input_dir, fn)
        local out_path = paths.concat(opt.output_dir, fn)
        print(in_path, out_path)
        run_image(in_path, out_path)
      end
    end
  elseif opt.input_image ~= '' then
    if opt.output_image == '' then
      error('Must give -output_image with -input_image')
    end
    run_image(opt.input_image, opt.output_image)
  end
end


main()
