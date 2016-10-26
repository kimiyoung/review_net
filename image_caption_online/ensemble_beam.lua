require 'nn'
require 'cudnn'
require 'cunn'
require 'nngraph'
require 'torch'
require 'cutorch'
require 'optim'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'
local tablex = require 'pl.tablex'
local anno_utils = require 'utils.anno_utils_filter'

torch.setdefaulttensortype('torch.FloatTensor')

local DataLoader = require 'dataloader'

local opts = require 'opts'
local opt = opts.parse(arg)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

local dataloader = DataLoader(opt)

local model_set = {}
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed13.model'))
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed23.model'))
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed33.model'))

function idx2coord(k, n)
    local i = math.floor((k-1)/n) + 1
    local j = (k-1) % n + 1
    assert((i-1)*n + j == k)
    return i,j
end

function language_eval(predictions, id)
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

local model = torch.load('models/reason_att_copy_simp_ensemble13_23_33.model')

function beam_search(model_set, model, dataloader, opt)
    local max_t = opt.val_max_len
    print('Max sequence length is: ' .. max_t)
    print('actual clone times ' .. max_t)

    -- table in table
    local anno_utils = dataloader.anno_utils
    local beam_size = opt.beam_size
    local word_cnt = opt.word_cnt
    local index2word = dataloader.index2word

    for ii = 1, #model_set do model_set[ii].softmax:remove() end

    local function evaluate()
        for ii = 1, #model_set do
            for t = 1, opt.reason_step do model_set[ii].soft_att_lstm[t]:evaluate() end
            model_set[ii].lstm:evaluate()
        end
    end

    local START, END = 1, #dataloader.val_set
    local BATCH_NUM, MY_BATCH_NUM = 4, 1
    local fout
    local captions = {}
    evaluate()

    if opt.server_test_mode then
        fout = io.open('server_test/ensemble.' .. MY_BATCH_NUM .. '.txt', 'w')
        local batch_size = math.floor(#dataloader.val_set / BATCH_NUM)
        START = 1 + batch_size * (MY_BATCH_NUM - 1)
        END = batch_size * MY_BATCH_NUM
        if BATCH_NUM == MY_BATCH_NUM then END = #dataloader.val_set end
    end

    print('ranging ' .. START .. ' ' .. END)
    
    for i = START, END do   -- for each image
        collectgarbage()
        local att_seq, fc7_images, _, _ = dataloader:gen_test_data(i, i)
        local image_map = fc7_images

        local cs, hs = {}, {}

        for ii = 1, #model_set do
            local c, h = image_map, image_map
            for t = 1, opt.reason_step do
                c, h = unpack(model_set[ii].soft_att_lstm[t]:forward{att_seq, c, h})
            end
            cs[ii], hs[ii] = c, h
        end
        
        print('Beam search predicting ' .. i .. 'th image...')
            
        -- Vars of beam search
        local loss_beam                                                      -- loss of each beam
        local sentence_beam = torch.CudaTensor(beam_size, max_t):zero()      -- beam_size * max_t
        local stop_beam = {}     -- if one beam encounters stop word, set it to false
        for k = 1, beam_size do 
            table.insert(stop_beam, false)   -- encounter stop word? no!
        end

        local text_input = {[1] = torch.CudaTensor(1):fill(anno_utils.START_NUM)}  -- input of text in every time step
                       
        for t = 1, max_t do
            if t == 1 then
                local predictions = torch.CudaTensor(1, word_cnt, #model_set)

                for ii = 1, #model_set do
                    local embeddings = model_set[ii].emb:forward(text_input[t])
                    cs[ii], hs[ii] = unpack(model_set[ii].lstm:forward{embeddings, cs[ii], hs[ii]})
                    predictions[{{}, {}, ii}] = model_set[ii].softmax:forward(hs[ii])
                end
                local combine = model:forward(predictions)

                loss_beam, sentence_beam[{{}, 1}] = combine:topk(beam_size, true)
                loss_beam = loss_beam:t()     -- beam_size * 1

                for ii = 1, #model_set do
                    local tmp_c, tmp_h = cs[ii], hs[ii]
                    cs[ii] = torch.CudaTensor(beam_size, cs[ii]:size(2))
                    hs[ii] = torch.CudaTensor(beam_size, hs[ii]:size(2))
                    for k = 1, beam_size do
                        cs[ii][k]:copy(tmp_c)
                        hs[ii][k]:copy(tmp_h)
                    end
                end
                                               
            else   -- when k > 1
                -- choose input text
                -- same for each model
                text_input[t] = sentence_beam[{{}, t-1}]
                local predictions = torch.CudaTensor(beam_size, word_cnt, #model_set)

                for ii = 1, #model_set do
                    local embeddings = model_set[ii].emb:forward(text_input[t])
                    cs[ii], hs[ii] = unpack(model_set[ii].lstm:forward{embeddings, cs[ii], hs[ii]})
                    predictions[{{}, {}, ii}] = model_set[ii].softmax:forward(hs[ii])
                end
                local combine = model:forward(predictions)

                for k = 1, beam_size do
                    if stop_beam[k] then
                        combine[k]:fill(- 10000)
                        combine[k][2] = 0
                    end
                end
                combine:add(torch.repeatTensor(loss_beam, 1, word_cnt))

                -- get top-k from predictions
                -- local log_prob = torch.reshape(predictions[t], beam_size * word_cnt)
                local log_prob = torch.reshape(combine, beam_size * word_cnt)

                local res, idx = log_prob:topk(beam_size, true)      -- res:  beam_size , idx: beam_size
                
                -- update loss_beam and sentence_beam
                local tmp_loss = torch.CudaTensor(beam_size, 1)                        -- beam_size * 1
                local tmp_sentence = torch.CudaTensor(beam_size, t)
                local tmp_lstm_c = torch.CudaTensor(#model_set, beam_size, cs[1]:size(2))
                local tmp_lstm_h = torch.CudaTensor(#model_set, beam_size, hs[1]:size(2))
                
                for k = 1,beam_size do
                    local a,b = idx2coord(idx[k], word_cnt)    -- a: beam idx,   b: predicted word
                    tmp_loss[k][1] = res[k]         -- update loss
                    tmp_sentence[{k, {1,t-1}}] = sentence_beam[{a, {1,t-1}}]         
                    tmp_sentence[k][t] = b
                    
                    if b <= 3 then
                        stop_beam[k] = true                        
                    end

                    for ii = 1, #model_set do
                        tmp_lstm_c[ii][k]:copy(cs[ii][a])
                        tmp_lstm_h[ii][k]:copy(hs[ii][a])
                    end
                end
                
                -- update
                loss_beam = tmp_loss
                sentence_beam[{{},{1,t}}] = tmp_sentence
               	
                for ii = 1, #model_set do
                    cs[ii] = tmp_lstm_c[ii]
                    hs[ii] = tmp_lstm_h[ii]
                end
            end
        end
            
        collectgarbage()

        -- Get words and insert into captions
        local idx
        _, idx = torch.max(loss_beam,2)        
        local sentence = sentence_beam[idx[1][1]]   
         
        local caption = ''
        for t = 1, max_t do 
            if sentence[t] <= 3 then break end
            if caption ~= '' then
                caption = caption .. ' ' .. index2word[sentence[t]]
            else
                caption = index2word[sentence[t]]
            end
        end
                  
        if i < START + 10 then
            print(dataloader.val_set[i], caption)
        end

        if opt.server_test_mode then
            fout:write(dataloader.val_set[i] .. '\t' .. caption .. '\n')
        else
            table.insert(captions, {image_id = dataloader.val_set[i], caption = caption})
        end
    end
    if opt.server_test_mode then
        fout:close()
    else
        local eval_struct = language_eval(captions, 'beam_' .. beam_size .. '_' .. opt.model)
    end
end


-- Load models

beam_search(model_set, model, dataloader, opt)
