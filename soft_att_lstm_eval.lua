require 'nn'
require 'nngraph'
require 'torch'
require 'cutorch'
local DataLoader = require 'dataloader'
local M = require 'soft_att_lstm'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = require 'opts'
local opt = opts.parse(arg)
print(opt)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

-- Initialize dataloader
local dataloader = DataLoader(opt)

-- Load model
local model = torch.load('models/' .. opt.model)

function idx2coord(k, n)
    local i = math.floor((k-1)/n) + 1
    local j = (k-1) % n + 1
    assert((i-1)*n + j == k)
    return i,j
end

---------------- Beam Search ---------------------
function beam_search(model, dataloader, opt)
    -- local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    local max_t = opt.val_max_len
    print('Max sequence length is: ' .. max_t)
    
    print('actual clone times ' .. max_t)
    local clones = {}
    local anno_utils = dataloader.anno_utils
    local beam_size = opt.beam_size
    local word_cnt = opt.word_cnt
    local index2word = dataloader.index2word
    
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    
    local captions = {}
    local i = 1
    while i <= #dataloader.val_set do
        collectgarbage()
        local att_seq, fc7_images = dataloader:gen_test_data(i, i)
        local image_map
        if opt.lstm_size ~= opt.fc7_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end
        
        local att_seq_beam = torch.CudaTensor(beam_size, att_seq:size()[2], att_seq:size()[3])
        for k = 1,beam_size do
            att_seq_beam[k] = att_seq:clone()
        end
        
        print('Beam search predicting ' .. i .. 'th image...')
            
        -- Vars of beam search
        local loss_beam                                                      -- loss of each beam
        local sentence_beam = torch.CudaTensor(beam_size, max_t):zero()      -- beam_size * max_t
        local stop_beam = {}     -- if one beam encounters stop word, set it to false
        for k = 1, beam_size do 
            table.insert(stop_beam, false)   -- encounter stop word? no!
        end
                        
        -- LSTM states
        local embeddings = {}
        local initstate_h = image_map                -- hid_size, for time step 1, only 1-d
        local initstate_c = image_map:clone()
        local init_input = torch.CudaTensor(1):fill(anno_utils.START_NUM)
        local lstm_c = {[0]=initstate_c}       
        local lstm_h = {[0]=initstate_h}       
        local text_input = {[1] = init_input}        -- input of text in every time step
        local predictions = {}                       -- softmax outputs        
                       
        for t = 1, max_t do
            if t == 1 then
                embeddings[t] = clones.emb[t]:forward(text_input[t])
                if opt.use_attention then
                    lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                        forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})
                else
                    lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:
                        forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
                end
                    
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- log softmax forward
                -- print('Prediction size')
                -- print(#predictions[t])
                -- os.exit()
                    
                -- get top-beam_size
                loss_beam, sentence_beam[{{}, 1}] = predictions[t]:topk(beam_size, true)
                
                loss_beam = loss_beam:t()     -- beam_size * 1
                
                -- copy hidden state and cell
                local lstm_c_tmp = lstm_c[t]:clone()
                local lstm_h_tmp = lstm_h[t]:clone()                
                lstm_c[t] = torch.CudaTensor(beam_size, lstm_c_tmp:size()[2])
                lstm_h[t] = torch.CudaTensor(beam_size, lstm_h_tmp:size()[2])
                    
                for k = 1,beam_size do
                    lstm_c[t][k] = lstm_c_tmp:clone()
                    lstm_h[t][k] = lstm_h_tmp:clone()
                end
                                
            else   -- when k > 1
                -- choose input text
                text_input[t] = sentence_beam[{{}, t-1}]
                --print('Text input size...')
                --print(#text_input[t])
                -- os.exit()
                
                -- forward
                embeddings[t] = clones.emb[t]:forward(text_input[t])
                if opt.use_attention then
                    lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                        forward{embeddings[t], att_seq_beam, lstm_c[t-1], lstm_h[t-1]})
                else
                    lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:
                        forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
                end
                
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])  -- beam_size * word_cnt, log probability
                --print('Prediction size...')  
                --print(#predictions[t])
                --os.exit()
                
                -- deal with stop_word
                for k = 1,beam_size do
                    if stop_beam[k] == true then
                        predictions[t][k] = torch.CudaTensor(1, word_cnt):fill(-100000)
                        predictions[t][k][2] = 0      -- make STOP WORD position to be 0
                    end
                end
                
                -- Add previous accumulated loss
                predictions[t]:add(torch.repeatTensor(loss_beam, 1, word_cnt))
               
                -- get top-k from predictions
                local log_prob = torch.reshape(predictions[t], beam_size * word_cnt)
                --print('Size of log_prob')
                --print(#log_prob)
                
                local res, idx = log_prob:topk(beam_size, true)      -- res:  beam_size , idx: beam_size
                
                --print(#res)
                --print(#idx)
                --os.exit()
                
                -- update loss_beam and sentence_beam
                local tmp_loss = torch.CudaTensor(beam_size, 1)                        -- beam_size * 1
                local tmp_sentence = torch.CudaTensor(beam_size, t)
                local tmp_lstm_c = torch.CudaTensor(beam_size, lstm_c[t]:size()[2])    -- beam_size * hid_size
                local tmp_lstm_h = torch.CudaTensor(beam_size, lstm_h[t]:size()[2])
                
                for k = 1,beam_size do
                    local a,b = idx2coord(idx[k], word_cnt)    -- a: beam idx,   b: predicted word
                    tmp_loss[k][1] = res[k]         -- update loss
                    tmp_sentence[{k, {1,t-1}}] = sentence_beam[{a, {1,t-1}}]         
                    tmp_sentence[k][t] = b
                    
                    if b <= 3 then
                        stop_beam[k] = true                        
                    end
                    
                    -- make deep copy of lstm_c and lstm_h
                    tmp_lstm_c[k] = lstm_c[t][a]:clone()
                    tmp_lstm_h[k] = lstm_h[t][a]:clone()
                end
                
                -- update
                loss_beam = tmp_loss
                sentence_beam[{{},{1,t}}] = tmp_sentence
                
                lstm_c[t] = tmp_lstm_c:clone()
                lstm_h[t] = tmp_lstm_h:clone()
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
                  
        if i <= 10 then
            print(dataloader.val_set[i], caption)
        end
        table.insert(captions, {image_id = dataloader.val_set[i], caption = caption})
                   
        -- Next image
        i = i + 1

        if i == 10 then break end -- DEBUG!!! remember to clear
    end
    -- Evaluate it
    local eval_struct = M.language_eval(captions, 'beam_' .. beam_size .. ' ' .. opt.model)
end


beam_search(model, dataloader, opt)
