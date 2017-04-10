require 'torch'
require 'nn'

local OriLoss, parent = torch.class('nn.OriLoss', 'nn.Module')


--[[
Module to compute content loss in-place.

The module can be in one of three modes: "none", "capture", or "loss", which
behave as follows:
- "none": This module does nothing; it is basically nn.Identity().
- "capture": On the forward pass, inputs are captured as targets; otherwise it
  is the same as an nn.Identity().
- "loss": On the forward pass, compute the distance between input and
  self.target, store the result in self.loss, and return input. On the backward
  pass, add compute the gradient of self.loss with respect to the inputs, and
  add this value to the upstream gradOutput to produce gradInput.
--]]

function OriLoss:__init(strength, loss_type)
  parent.__init(self)
  self.strength = strength or 1.0
  self.loss = 0
  self.target = torch.Tensor()
  self.oriLabel = torch.Tensor()

  self.mode = 'none'
  loss_type = loss_type or 'Classification'
  if loss_type == 'Classification' then
    -- self.crit = nn.CrossEntropyCriterion()
    self.crit = cudnn.SpatialCrossEntropyCriterion()
  else
    error(string.format('Invalid loss_type "%s"', loss_type))
  end
end

--[[
 - first, for a layer in nn:
   'updateOutput(input)' will be called when doing 'forward(input)'
   'updateOutput(input)' is used for custom the forward function;

 - mode = 'capture', is to save target feature map:
   the input is the feature map corresponding to the content target image;

 - mode = 'loss', is to calculate loss between feature maps:
   the input is the feature map corresponding to the generated image.
--]]
function OriLoss:updateOutput(input)
  if self.mode == 'capture' then
    -- self.target:resizeAs(input):copy(input)
  elseif self.mode == 'loss' then
    maxval, self.oriLabel = torch.max(self.oriLabel,2)
    local H, W = self.oriLabel:size(3), self.oriLabel:size(4)
    local HH, WW = input:size(3), input:size(4)
    local xs = (H - HH) / 2
    local ys = (W - WW) / 2
    self.oriLabel = self.oriLabel[{{}, {}, {xs + 1, H - xs}, {ys + 1, W - ys}}][1]
    -- print('input:, ', input:size())
    -- print('self.oriLabel: ', self.oriLabel:size())
    -- print(self.oriLabel)
    self.loss = self.strength * self.crit:forward(input, self.oriLabel)
  end
  self.output = input
  return self.output
end


function OriLoss:updateGradInput(input, gradOutput)
  if self.mode == 'capture' or self.mode == 'none' then
    self.gradInput = gradOutput
  elseif self.mode == 'loss' then
    self.gradInput = self.crit:backward(input, self.oriLabel)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end


function OriLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end

function OriLoss:setOriLabel(label)
  self.oriLabel = label
end