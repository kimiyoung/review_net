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
        if name ~= 'soft_att_lstm' and name ~= 'reason_softmax' and name ~= 'reason_criterion'
            and name ~= 'pooling' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    print('cloning reasoning lstm')
    clones.soft_att_lstm = model_utils.clone_many_times(model.soft_att_lstm, opt.reason_step)
    if opt.use_noun then
        clones.reason_softmax = model_utils.clone_many_times(model.reason_softmax, opt.reason_step)
        -- clones.reason_criterion = model_utils.clone_many_times(model.reason_criterion, opt.reason_step)
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

        local reason_c = {[0] = image_map}
        local reason_h = {[0] = image_map}
        local reason_h_att = torch.CudaTensor(att_seq:size()[1], opt.reason_step, opt.lstm_size)
        local reason_len = opt.reason_step

        for t = 1, reason_len do
            reason_c[t], reason_h[t] = unpack(clones.soft_att_lstm[t]:
                forward{att_seq, reason_c[t - 1], reason_h[t - 1]})
            reason_h_att:select(2, t):copy(reason_h[t])
        end
        
        att_seq = reason_h_att

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
        local lstm_c = {[0] = reason_c[reason_len]}
        local lstm_h = {[0] = reason_h[reason_len]}
        local text_input = {[1] = torch.CudaTensor(1):fill(anno_utils.START_NUM)}  -- input of text in every time step
        local predictions = {}                       -- softmax outputs        
                       
        for t = 1, max_t do
            if t == 1 then
                embeddings[t] = clones.emb[t]:forward(text_input[t])
                lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:            -- lstm forward
                    forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})
                    
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- log softmax forward
                    
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
                
                -- forward
                embeddings[t] = clones.emb[t]:forward(text_input[t])
                lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:            -- lstm forward
                    forward{embeddings[t], att_seq_beam, lstm_c[t-1], lstm_h[t-1]})
                
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])  -- beam_size * word_cnt, log probability
                
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
                
                local res, idx = log_prob:topk(beam_size, true)      -- res:  beam_size , idx: beam_size
                
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
    end
    -- Evaluate it
    -- local eval_struct = M.language_eval(captions, 'beam_' .. beam_size .. '_concat.1024.512.model')
    local eval_struct = M.language_eval(captions, 'beam_' .. beam_size .. '_' .. opt.model)
end

-- beam_search(model, dataloader, opt)
-- os.exit()

---------------- Greedy Search -------------------
function greedy_search(model, dataloader, opt) 
    --local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    local max_t = opt.val_max_len
    
    print('actual clone times ' .. max_t)
    local clones = {}
    local anno_utils = dataloader.anno_utils
    
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    
    local captions = {}
    local j1 = 1
    while j1 <= #dataloader.val_set do
        local j2 = math.min(#dataloader.val_set, j1 + opt.val_batch_size)
        att_seq, fc7_images = dataloader:gen_test_data(j1, j2)

        local image_map
        if opt.lstm_size ~= opt.fc7_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end

        local initstate_c = image_map:clone()
        local initstate_h = image_map
        local init_input = torch.CudaTensor(att_seq:size()[1]):fill(anno_utils.START_NUM)
                    
        ------------------- Forward pass -------------------
        local embeddings = {}              -- input text embeddings
        local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
        local lstm_h = {[0]=initstate_h}   -- output values of LSTM
        local predictions = {}             -- softmax outputs
        local max_pred = {[1] = init_input}          -- max outputs 
        local seq_len = max_t     -- sequence length 
                    
        for t = 1, seq_len do
            embeddings[t] = clones.emb[t]:forward(max_pred[t])    -- emb forward
            if opt.use_attention then
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                    forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})
            else
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:
                    forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
            end    
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- softmax forward
                _, max_pred[t + 1] = torch.max(predictions[t], 2)
                max_pred[t + 1] = max_pred[t + 1]:view(-1)
        end

        ------------------- Get captions -------------------
        index2word = dataloader.index2word
        for k = 1, att_seq:size()[1] do
            local caption = ''
            for t = 2, seq_len do
                local word_index = max_pred[t][k]
                if word_index == anno_utils.STOP_NUM then break end
                if caption ~= '' then
                    caption = caption .. ' ' .. index2word[word_index]
                else
                    caption = index2word[word_index]
                end
            end
            if j1 + k - 1 <= 20 then
                print(dataloader.val_set[j1 + k - 1], caption)
            end
            table.insert(captions, {image_id = dataloader.val_set[j1 + k - 1], caption = caption})
        end
        j1 = j2 + 1
    end
    
    -- Evaluate it
    local eval_struct = M.language_eval(captions, 'greedy_concat.1024.512.model')
end


if opt.eval_algo == 'beam' then
    beam_search(model, dataloader, opt)
else
    greedy_search(model, dataloader, opt)
end

