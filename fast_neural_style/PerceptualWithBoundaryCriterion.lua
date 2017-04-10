--[[
Modified from PerceptualCriterion.lua
Aim to add boundary loss
]]--

require 'torch'
require 'nn'

require 'fast_neural_style.ContentLoss'
require 'fast_neural_style.StyleLoss'
require 'fast_neural_style.OriLoss'
require 'fast_neural_style.DeepDreamLoss'

local layer_utils = require 'fast_neural_style.layer_utils'


local crit, parent = torch.class('nn.PerceptualWithBoundaryCriterion', 'nn.Criterion')


--[[
Input: args is a table with the following keys:
- cnn: A network giving the base CNN.
- content_layers: An array of layer strings
- content_weights: A list of the same length as content_layers
- style_layers: An array of layers strings
- style_weights: A list of the same length as style_layers
- agg_type: What type of spatial aggregaton to use for style loss;
  "mean" or "gram"
- deepdream_layers: Array of layer strings
- deepdream_weights: List of the same length as deepdream_layers
- loss_type: Either "L2", or "SmoothL1"
--]]
function crit:__init(args)
  args.content_layers = args.content_layers or {}
  args.boundary_layers = args.boundary_layers or {}
  args.style_layers = args.style_layers or {}
  args.ori_layers = args.ori_layers or {}
  args.deepdream_layers = args.deepdream_layers or {}
  
  self.net = args.cnn
  self.net:evaluate()
  self.net_boundary = args.cnn_boundary
  self.net_boundary:evaluate()
  self.net_ori = args.cnn_ori
  self.net_ori:evaluate()
  self.net_pred_ori = args.cnn_pred_ori
  self.net_pred_ori:evaluate()
  self.content_loss_layers = {}
  self.boundary_loss_layers = {}
  self.style_loss_layers = {}
  self.ori_loss_layers = {}
  self.deepdream_loss_layers = {}

  print('display loss layers')
  print(args.content_layers)
  print(args.style_layers)
  print(args.boundary_layers)
  print(args.ori_layers)

  -- Set up content loss layers
  for i, layer_string in ipairs(args.content_layers) do
    local weight = args.content_weights[i]
    local content_loss_layer = nn.ContentLoss(weight, args.loss_type)
    -- print('content_loss_layer: ', content_loss_layer)
    layer_utils.insert_after(self.net, layer_string, content_loss_layer)
    table.insert(self.content_loss_layers, content_loss_layer)
  end

  -- Set up boundary loss layers
  for i, layer_string in ipairs(args.boundary_layers) do
    local weight = args.boundary_weights[i]
    local boundary_loss_layer = nn.ContentLoss(weight, args.loss_type)
    layer_utils.insert_after(self.net_boundary, layer_string, boundary_loss_layer)
    table.insert(self.boundary_loss_layers, boundary_loss_layer)
  end
  -- print('self.boundary_loss_layers: ', self.boundary_loss_layers)

  -- Set up style loss layers
  for i, layer_string in ipairs(args.style_layers) do
    local weight = args.style_weights[i]
    local style_loss_layer = nn.StyleLoss(weight, args.loss_type, args.agg_type)
    layer_utils.insert_after(self.net, layer_string, style_loss_layer)
    table.insert(self.style_loss_layers, style_loss_layer)
  end

  -- Set up ori loss layers (only add to its final layer)
  for i, layer_string in ipairs(args.ori_layers) do
    local weight = args.ori_weights[i]
    local ori_loss_layer = nn.OriLoss(weight, 'Classification', self.net_pred_ori)
    -- print('ori_loss_layer: ', ori_loss_layer)
    layer_utils.insert_after(self.net_ori, layer_string, ori_loss_layer)
    table.insert(self.ori_loss_layers, ori_loss_layer)
  end
  -- print('self.ori_loss_layers: ', self.ori_loss_layers)

  -- Set up DeepDream layers
  for i, layer_string in ipairs(args.deepdream_layers) do
    local weight = args.deepdream_weights[i]
    local deepdream_loss_layer = nn.DeepDreamLoss(weight)
    layer_utils.insert_after(self.net, layer_string, deepdream_loss_layer)
    table.insert(self.deepdream_loss_layers, deepdream_loss_layer)
  end
  
  layer_utils.trim_network(self.net)
  layer_utils.trim_network(self.net_boundary)
  layer_utils.trim_network(self.net_ori)
  self.grad_net_output = torch.Tensor()
  self.grad_net_boundary_output = torch.Tensor()
  self.grad_net_ori_output = torch.Tensor()

end


--[[
target: Tensor of shape (1, 3, H, W) giving pixels for style target image
--]]
function crit:setStyleTarget(target)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('none')
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('capture')
  end
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    boundary_loss_layer:setMode('none')
  end  
  self.net:forward(target)
end


--[[
target: Tensor of shape (N, 3, H, W) giving pixels for content target images
--]]
function crit:setContentTarget(target)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('none')
  end
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('capture')
  end
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    boundary_loss_layer:setMode('none')
  end
  self.net:forward(target)
end

--[[
target: Tensor of shape (N, 3, H, W) giving pixels for content target images
--]]
function crit:setBoundaryTarget(target)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('none')
  end
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('none')
  end
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    boundary_loss_layer:setMode('capture')
  end
  self.net_boundary:forward(target)
end

--[[
target: Tensor of shape (N, 3, H, W) giving pixels for content target images
--]]
function crit:setOriTarget(target)
  for i, ori_loss_layer in ipairs(self.ori_loss_layers) do
    local ori_label = self.net_pred_ori:forward(target)
    ori_loss_layer:setOriLabel(ori_label)
  end
end

function crit:setStyleWeight(weight)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer.strength = weight
  end
end


function crit:setContentWeight(weight)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer.strength = weight
  end
end

function crit:setBoundaryWeight(weight)
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    boundary_loss_layer.strength = weight
  end
end

function crit:setOriWeight(weight)
  for i, ori_loss_layer in ipairs(self.ori_loss_layers) do
    ori_loss_layer.strength = weight
  end
end


--[[
Inputs:
- input: Tensor of shape (N, 3, H, W) giving pixels for generated images
- target: Table with the following keys:
  - content_target: Tensor of shape (N, 3, H, W)
  - style_target: Tensor of shape (1, 3, H, W)
--]]
function crit:updateOutput(input, target)
  if target.content_target then
    self:setContentTarget(target.content_target)
  end
  if target.boundary_target then
    self:setBoundaryTarget(target.boundary_target)
  end
  if target.style_target then
    self:setStyleTarget(target.style_target) -- not into if here but . or : ?? should be :!
  end
  if target.ori_net_input then  
    self:setOriTarget(target.ori_net_input)
  end

  -- Make sure to set all content and style loss layers to loss mode before
  -- running the image forward.
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('loss')
  end
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    boundary_loss_layer:setMode('loss')
  end  
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('loss')
  end
  for i, ori_loss_layer in ipairs(self.ori_loss_layers) do
    ori_loss_layer:setMode('loss')
  end


  local output = self.net:forward(input)
  local output_boundary = self.net_boundary:forward(input)
  local output_ori = self.net_ori:forward(input)

  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()
  self.grad_net_boundary_output:resizeAs(output_boundary):zero()
  self.grad_net_ori_output:resizeAs(output_ori):zero()

  -- Go through and add up losses
  self.total_content_loss = 0
  self.content_losses = {}
  self.total_boundary_loss = 0
  self.boundary_losses = {}
  self.total_style_loss = 0
  self.style_losses = {}
  self.total_ori_loss = 0
  self.ori_losses = {}
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    self.total_content_loss = self.total_content_loss + content_loss_layer.loss
    table.insert(self.content_losses, content_loss_layer.loss)
  end
  for i, boundary_loss_layer in ipairs(self.boundary_loss_layers) do
    self.total_boundary_loss = self.total_boundary_loss + boundary_loss_layer.loss
    table.insert(self.boundary_losses, boundary_loss_layer.loss)
  end  
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    self.total_style_loss = self.total_style_loss + style_loss_layer.loss
    table.insert(self.style_losses, style_loss_layer.loss)
  end
  for i, ori_loss_layer in ipairs(self.ori_loss_layers) do
    self.total_ori_loss = self.total_ori_loss + ori_loss_layer.loss
    table.insert(self.ori_losses, ori_loss_layer.loss)
  end
  
  self.output = self.total_style_loss + self.total_content_loss + self.total_boundary_loss + self.total_ori_loss
  print('total_style_loss: ', self.total_style_loss)
  print('total_content_loss: ', self.total_content_loss)
  print('total_boundary_loss: ', self.total_boundary_loss)
  print('total_ori_loss: ', self.total_ori_loss)
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  self.gradInput_boundary = self.net_boundary:updateGradInput(input, self.grad_net_boundary_output)
  self.gradInput_ori = self.net_ori:updateGradInput(input, self.grad_net_ori_output)
  self.gradInput = self.gradInput + self.gradInput_boundary + self.gradInput_ori
  return self.gradInput
end

