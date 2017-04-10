require 'torch'
require 'nn'

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight = torch.repeatTensor( 
                 torch.repeatTensor( 
                       torch.Tensor({ {{-3,-5, 0, 5, 3, 1, 1}, 
                                       {-3,-5, 0, 5, 3, 1, 1},
                                       {-1,-3,-5, 0, 5, 3, 1},
                                       {-1,-3,-5, 0, 5, 3, 1},
                                       {-1,-3,-5, 0, 5, 3, 1},
                                       {-1,-1,-3,-5, 0, 5, 3},
                                       {-1,-1,-3,-5, 0, 5, 3}},
                                      {{ 0, 5, 3, 1, 1, 1, 1}, 
                                       {-5, 0, 5, 3, 1, 1, 1},
                                       {-3,-5, 0, 5, 3, 1, 1},
                                       {-1,-3,-5, 0, 5, 3, 1},
                                       {-1,-1,-3,-5, 0, 5, 3},
                                       {-1,-1,-1,-3,-5, 0, 5},
                                       {-1,-1,-1,-1,-3,-5, 0}},
                                      {{ 3, 3, 1, 1, 1, 1, 1}, 
                                       { 5, 5, 3, 3, 3, 1, 1},
                                       { 0, 0, 5, 5, 5, 3, 3},
                                       {-5,-5, 0, 0, 0, 5, 5},
                                       {-3,-3,-5,-5,-5, 0, 0},
                                       {-1,-1,-3,-3,-3,-5,-5},
                                       {-1,-1,-1,-1,-1,-3,-3}},
                                      {{ 1, 1, 1, 1, 1, 1, 1}, 
                                       { 3, 3, 3, 3, 3, 3, 3},
                                       { 5, 5, 5, 5, 5, 5, 5},
                                       { 0, 0, 0, 0, 0, 0, 0},
                                       {-5,-5,-5,-5,-5,-5,-5},
                                       {-3,-3,-3,-3,-3,-3,-3},
                                       {-1,-1,-1,-1,-1,-1,-1}},
                                      {{ 1, 1, 1, 1, 1, 3, 3}, 
                                       { 1, 1, 3, 3, 3, 5, 5},
                                       { 3, 3, 5, 5, 5, 0, 0},
                                       { 5, 5, 0, 0, 0,-5,-5},
                                       { 0, 0,-5,-5,-5,-3,-3},
                                       {-5,-5,-3,-3,-3,-1,-1},
                                       {-3,-3,-1,-1,-1,-1,-1}},
                                      {{ 1, 1, 1, 1, 3, 5, 0}, 
                                       { 1, 1, 1, 3, 5, 0,-5},
                                       { 1, 1, 3, 5, 0,-5,-3},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 3, 5, 0,-5,-3,-1,-1},
                                       { 5, 0,-5,-3,-1,-1,-1},
                                       { 0,-5,-3,-1,-1,-1,-1}},
                                      {{ 1, 1, 3, 5, 0,-5,-3}, 
                                       { 1, 1, 3, 5, 0,-5,-3},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 3, 5, 0,-5,-3,-1,-1},
                                       { 3, 5, 0,-5,-3,-1,-1}},
                                      {{ 1, 3, 5, 0,-5,-3,-1}, 
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1},
                                       { 1, 3, 5, 0,-5,-3,-1}}  }), 1, 1, 1, 1):permute(2,1,3,4), 1,3,1,1 )
      m.bias:fill(0)
   end
end

function define_Ori_net()
    local OreNet = nil

    local OreNet = nn.Sequential()

    OreNet:add( nn.SpatialConvolution(3, 8, 7, 7, 1, 1) )

    OreNet:apply(weights_init)

    return OreNet
end