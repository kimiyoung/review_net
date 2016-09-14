require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'

-- use float to store all data
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Etract VGG features')
cmd:text()
cmd:text('Options:')

cmd:option('-nGPU', 1, 'Choose GPU')
cmd:option('-imagePath', 
           '/usr1/public/zhiliny/ImageCaptioning/data/train2014/', 
           'Folder of input images')
cmd:option('-outPath', 
           '../data/train2014_features_vgg_vd19_fc7/',
           'path to save feature vectors') 
cmd:option('-model', 
           '../models/vgg_vd19_fc7.t7',
           'path to models') 


opt = cmd:parse(arg or {})

-- set device
cutorch.setDevice(opt.nGPU)

---------------------- util functions ----------------------------------------------------
loadSize   = {3, 256, 256}
sampleSize = {3, 224, 224}

function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- VGG preprocessing
bgr_means = {103.939,116.779,123.68}

function vggPreprocess(img)
  local im2 = img:clone()
  im2[{1,{},{}}] = img[{3,{},{}}]
  im2[{3,{},{}}] = img[{1,{},{}}]

  im2:mul(255)
  for i=1,3 do
    im2[i]:add(-bgr_means[i])
  end
  return im2
end

function centerCrop(input)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   return out
end

-- function to load the image
extractHook = function(path)
   collectgarbage()
   local input = loadImage(path)
   local vggPreprocessed = vggPreprocess(input)
   local out = centerCrop(vggPreprocessed)
   return out
end

---------------------------------------------------------------------------------

-- load models --
local model = torch.load(opt.model)
print('=> Model')
print(model)

local idx = 1
local files = {}
for file in paths.files(opt.imagePath) do
    if file:find('.jpg' .. '$') then
        table.insert(files, file)
        print('Extracting feature of ' .. idx .. 'th image ' .. file)
        
        local img = extractHook(paths.concat(opt.imagePath, file))
        local vecs = model:forward(img:cuda()):squeeze(1)
        
        -- save features
        local name = string.sub(file, 1, file:len()-4)
        torch.save(opt.outPath .. name .. '.dat', vecs)
                
        idx = idx + 1
    end
end
