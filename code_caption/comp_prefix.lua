
function max_common(s1, s2)
    for i = 1, math.min(s1:len(), s2:len()) do
        if s1[i] ~= s2[i] then return i - 1 end
    end
    return math.min(s1:len(), s2:len())
end

function comp_prefix(index2word, filename)
    local prefix = torch.Tensor(#index2word, #index2word)
    for i = 1, #index2word do
        if i % 100 == 0 then print('comp_prefix', i) end
        for j = i, #index2word do
            prefix[i][j] = max_common(index2word[i], index2word[j])
            prefix[j][i] = prefix[i][j]
        end
    end
    torch.save(filename, prefix)
    return prefix
end

require 'nn'
require 'cunn'

local data = require 'data'

local PREFIX_FILE = 'prefix.dat'

local TRAIN_FILE = 'train.dat'
local DEV_FILE = 'dev.dat'
local TEST_FILE = 'test.dat'

cutorch.setDevice(1)
torch.manualSeed(13)

local token2index, index2token, token_cnt, word_cnt = data.indexing({TRAIN_FILE, DEV_FILE, TEST_FILE})
local index2word = {}

for i = 1, word_cnt do
    index2word[i] = index2token[i]
end

comp_prefix(index2word, PREFIX_FILE)
