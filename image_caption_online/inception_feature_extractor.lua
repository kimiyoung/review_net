require 'torch'
require 'image'
require 'nn'
require 'paths'
local pl = require('pl.import_into')()

-- Important
torch.setdefaulttensortype('torch.FloatTensor')

-- Some definitions copied from the TensorFlow model
-- input subtraction
local input_sub = 128
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 299

local load_image = function(path)
  local img   = image.load(path, 3)
  local w, h  = img:size(3), img:size(2)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, input_dim)
  -- normalize image
  img:mul(255):add(-input_sub):mul(input_scale)
  -- due to batch normalization we must use minibatches
  return img:float():view(1, img:size(1), img:size(2), img:size(3))
end

local args = pl.lapp [[
  -m (string) inception v3 model file
  -f (string) feature type: "conv" | "fc"
  -b (string) backend of the model: "nn"|"cunn"|"cudnn"
  -i (string) input image folder
  -o (string) output feature folder
]]

if args.b == "cunn" then
  require "cunn"
elseif args.b == "cudnn" then
  require "cunn"
  require "cudnn"
end

local net = torch.load(args.m)
net:evaluate()
-- inception (31): nn.Linear(2048 -> 1000)
-- global context vector
if args.f == "conv" then
    net.modules[27] = nil
    net.modules[28] = nil
    net.modules[29] = nil
    net.modules[30] = nil
    net.modules[31] = nil
    net.modules[32] = nil
elseif args.f == "fc" then
    net.modules[31] = nil
    net.modules[32] = nil
end


-- local synsets = pl.utils.readlines(args.s)

-- create output folder 
if not paths.dirp(args.o) then
    paths.mkdir(args.o)
end

local idx = 1
local files = {}
for file in paths.files(args.i) do
    if file:find('.jpg' .. '$') then
        table.insert(files, file)
        print('Extracting feature of ' .. idx .. 'th image ' .. file)
        
        local img = load_image(paths.concat(args.i, file))
        if args.b == "cudnn" or args.b == "cunn" then
            img = img:cuda()
        end
        local vecs = net:forward(img):squeeze()
        
        -- save features
        local name = string.sub(file, 1, file:len()-4)
        torch.save(paths.concat(args.o, name .. '.dat'), vecs)
                
        idx = idx + 1
    end
end







