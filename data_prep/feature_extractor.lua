require 'torch'
require 'inn'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'

-- use float to store all data
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Etract VGG features')
cmd:text()
cmd:text('Options:')

cmd:option('-nGPU', 3, 'Choose GPU')
cmd:option('-imagePath', 
           '../data/MSCOCO/val2014/', 
           'path to images')
cmd:option('-outPath', 
           '../data/MSCOCO/val2014_features_vgg_vd19_conv5_2nd/',
           'path to save feature vectors') 

opt = cmd:parse(arg or {})

-- set device
cutorch.setDevice(opt.nGPU)

-- load model
-- case 1: google net
-- local model = torch.load('models/googlenet.t7')
-- model.modules[24] = nil
-- model.modules[25] = nil

-- case 2: vgg
local model = paths.dofile('models/vgg_vd19_conv5.lua')

print('=> Model')
print(model)
-- os.exit()

-- transform utilities
local t = paths.dofile('utils/transforms.lua')
meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}

transform = t.Compose{
    t.Scale(256),
    t.ColorNormalize(meanstd),
    t.CenterCrop(224),
}

files = {}
idx = 1
for file in paths.files(opt.imagePath) do
    if file:find('.jpg' .. '$') then
        table.insert(files, file)
        print('Extracting feature of ' .. idx .. 'th image ' .. file)
        local img = image.load(paths.concat(opt.imagePath, file), 3, 'float')
        -- pre-processing
        img = transform(img)
        img = img:view(1, table.unpack(img:size():totable()))
        local vecs = model:forward(img:cuda()):squeeze(1)

        local name = string.sub(file, 1, file:len()-4)
        torch.save(opt.outPath .. name .. '.dat', vecs)
        
        idx = idx + 1
    end
end


