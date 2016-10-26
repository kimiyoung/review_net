
local MODEL_NAME = 'rev_reason' -- please assure the correct model name for loading!

print(MODEL_NAME)

local data = require 'data'
local model = require(MODEL_NAME)

local TRAIN_FILE = 'train.dat'
local DEV_FILE = 'dev.dat'
local TEST_FILE = 'test.dat'
local PREFIX_FILE = 'prefix.dat'

cutorch.setDevice(1)
torch.manualSeed(13)

local token2index, index2token, token_cnt, word_cnt = data.indexing({TRAIN_FILE, DEV_FILE, TEST_FILE})
local batches = data.prepare_data(TRAIN_FILE, token2index, word_cnt)
local test_batches = data.prepare_data(TEST_FILE, token2index, word_cnt)

local index2word = {}
for i = 1, word_cnt do
    index2word[i] = index2token[i]
end

print('#train batches', #batches, '#test batches', #test_batches)
print('word_cnt', word_cnt, 'token_cnt', token_cnt)

prefix = torch.load(PREFIX_FILE)
model.eval(test_batches, prefix, index2word)
model.train(batches, test_batches, word_cnt, token_cnt, true) -- for eval llh only

