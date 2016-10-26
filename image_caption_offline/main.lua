require 'nn'
require 'nngraph'
require 'torch'
require 'cutorch'
local DataLoader = require 'dataloader'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = require 'opts'
local opt = opts.parse(arg)
print(opt)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

-- local model_pack = opt.use_reasoning and 'reasoning' or 'soft_att_lstm'
local model_pack = opt.model_pack
local M = require(model_pack)

-- Initialize dataloader
local dataloader = DataLoader(opt)

-- Create model
local model
if not opt.load_file then
    model = M.create_model(opt)
else
    model = torch.load('models/' .. opt.load_file_name)
end

-- Train
local optim_state = {learningRate = opt.LR}
local batches = dataloader:gen_batch(dataloader.train_len2captions, opt.batch_size)
local val_batches = dataloader:gen_batch(dataloader.val_len2captions, opt.batch_size)
M.train(model, opt, batches, val_batches, optim_state, dataloader)

