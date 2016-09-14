require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'
local tablex = require 'pl.tablex'

local M = {}
-- local ATT_NEXT_H = false

-- cmd:option('-emb_size', 100, 'Word embedding size')
-- cmd:option('-lstm_size', 4096, 'LSTM size')
-- cmd:option('-word_cnt', 9520, 'Vocabulary size')
-- cmd:option('-att_size', 196, 'Attention size')
-- cmd:option('-feat_size', 512, 'Feature size for each attention')
-- cmd:option('-batch_size', 32, 'Batch size in SGD')
function M.soft_att_lstm(opt)
    -- Model parameters
    local feat_size = opt.feat_size
    local att_size = opt.att_size
    local batch_size = opt.batch_size
    local rnn_size = opt.lstm_size
    local input_size = opt.emb_size

    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local att_seq = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------
    local att = nn.View(-1, feat_size)(att_seq)
    att = nn.Linear(feat_size, rnn_size)(att)
    att = nn.View(-1, att_size, rnn_size)(att)          -- batch * att_size * rnn_size <- batch * att_size * feat_size

    local dot = nn.MixtureTable(3){prev_h, att}         -- batch * att_size <- (batch * rnn_size, batch * att_size * rnn_size)
    local weight = nn.SoftMax()(dot)                    -- batch * att_size
    local att_seq_t = nn.Transpose({2, 3})(att_seq)             -- batch * rnn_size * att_size
    local att_res = nn.MixtureTable(3){weight, att_seq_t}       -- batch * rnn_size <- (batch * att_size, batch * rnn_size * att_size)
    
    --- Input to LSTM
    local att_add = nn.Linear(feat_size, 4 * rnn_size)(att_res)    -- batch * (4*rnn_size) <- batch * rnn_size

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({i2h, h2h, att_add})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    
    return nn.gModule({x, att_seq, prev_c, prev_h}, {next_c, next_h})
end

-- New attention model
function M.soft_att_lstm_2(opt)
    -- Model parameters
    local feat_size = opt.feat_size
    local att_size = opt.att_size
    local batch_size = opt.batch_size
    local rnn_size = opt.lstm_size
    local input_size = opt.emb_size

    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local att_seq = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------
    -- Transfer hidden state to 512-d
    local att_h = nn.Linear(rnn_size, feat_size)(prev_h)       -- batch * rnn_size => batch * feat_size    
    local dot = nn.MixtureTable(3){att_h, att_seq}             -- batch * att_size <- (batch * feat_size, batch * att_size * feat_size)
    local weight = nn.SoftMax()(dot)                           -- batch * att_size
    local att_seq_t = nn.Transpose({2, 3})(att_seq)            -- batch * feat_size * att_size
    local att = nn.MixtureTable(3){weight, att_seq_t}          -- batch * feat_size <- (batch * att_size, batch * rnn_size * att_size)
    
    --- Input to LSTM
    local att_add = nn.Linear(feat_size, 4 * rnn_size)(att)    -- batch * (4*rnn_size) <- batch * feat_size

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({i2h, h2h, att_add})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    
    return nn.gModule({x, att_seq, prev_c, prev_h}, {next_c, next_h})
end


-- Attention model, concat hidden and image feature
function M.soft_att_lstm_concat(opt)
    -- Model parameters
    local feat_size = opt.feat_size
    local att_size = opt.att_size
    local batch_size = opt.batch_size
    local rnn_size = opt.lstm_size
    local input_size = opt.emb_size
    local att_hid_size = opt.att_hid_size

    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local att_seq = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------
    local att = nn.View(-1, feat_size)(att_seq)         -- (batch * att_size) * feat_size
    local att_h, dot

    if att_hid_size > 0 then
        att = nn.Linear(feat_size, att_hid_size)(att)       -- (batch * att_size) * att_hid_size
        att = nn.View(-1, att_size, att_hid_size)(att)      -- batch * att_size * att_hid_size
        att_h = nn.Linear(rnn_size, att_hid_size)(prev_h)   -- batch * att_hid_size
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * att_hid_size
        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size * att_hid_size
        dot = nn.Tanh()(dot)                                -- batch * att_size * att_hid_size
        dot = nn.View(-1, att_hid_size)(dot)                -- (batch * att_size) * att_hid_size
        dot = nn.Linear(att_hid_size, 1)(dot)               -- (batch * att_size) * 1
        dot = nn.View(-1, att_size)(dot)                             -- batch * att_size
    else
        att = nn.Linear(feat_size, 1)(att)                  -- (batch * att_size) * 1
        att = nn.View(-1, att_size)(att)                    -- batch * att_size
        att_h = nn.Linear(rnn_size, 1)(prev_h)              -- batch * 1
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * 1
        att_h = nn.Squeeze()(att_h)                         -- batch * att_size
        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size
    end

    local weight = nn.SoftMax()(dot)
        
    local att_seq_t = nn.Transpose({2, 3})(att_seq)     -- batch * rnn_size * att_size
    local att_res = nn.MixtureTable(3){weight, att_seq_t}      -- batch * rnn_size <- (batch * att_size, batch * rnn_size * att_size)

    -------------- End of attention part -----------
    
    --- Input to LSTM
    local att_add = nn.Linear(feat_size, 4 * rnn_size)(att_res)   -- batch * (4*rnn_size) <- batch * feat_size

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({i2h, h2h, att_add})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    
    return nn.gModule({x, att_seq, prev_c, prev_h}, {next_c, next_h})
end

-------------------------------------
-- Train the model for an epoch
-- INPUT
-- batches: {{id, caption}, ..., ...}
-------------------------------------
function M.train(model, opt, batches, val_batches, optim_state, dataloader)
    local DEBUG_LEN = false
    local params, grad_params
    if opt.lstm_size ~= opt.fc7_size then
        if opt.use_noun then
            params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.softmax, model.linear, model.reason_softmax, model.pooling)
        else
            params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.softmax, model.linear)
        end
    else
        if opt.use_noun then
            params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.softmax, model.reason_softmax, model.pooling)
        else
            params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.softmax)
        end
    end
    local clones = {}
    anno_utils = dataloader.anno_utils
    
    -- Clone models
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'linear' and name ~= 'pooling' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end

    local att_seq, fc7_images, input_text, output_text, noun_list

    local function feval(params_, update)
        if update == nil then update = true end
        if params_ ~= params then
            params:copy(params_)
        end
        grad_params:zero()

        local image_map
        if opt.lstm_size ~= opt.fc7_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end

        local initstate_c = image_map:clone()
        local initstate_h = image_map
        local dfinalstate_c = torch.zeros(input_text:size()[1], opt.lstm_size):cuda()
        
        -- print('Start forward')
        ------------------- forward pass -------------------
        local embeddings = {}              -- input text embeddings
        local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
        local lstm_h = {[0]=initstate_h}   -- output values of LSTM
        local predictions = {}             -- softmax outputs
        local loss = 0
        local out_dim = opt.word_cnt
        local seq_len = input_text:size()[2]     -- sequence length 
        seq_len = math.min(seq_len, max_t) -- get truncated
        local reason_preds = {}
        local reason_pred_mat = torch.CudaTensor(input_text:size()[1], seq_len, out_dim)
        
        if DEBUG_LEN then print('seq_len', seq_len) end
        for t = 1, seq_len do
            if DEBUG_LEN then print('Forward time step ' .. t) end
            embeddings[t] = clones.emb[t]:forward(input_text:select(2, t))    -- emb forward
            if opt.use_attention then
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                    forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})    
            else
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:
                    forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
            end
            if opt.use_noun then
                reason_preds[t] = clones.reason_softmax[t]:forward(lstm_h[t])
                reason_pred_mat:select(2, t):copy(reason_preds[t])
            end
            
            predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- softmax forward
            loss = loss + clones.criterion[t]:forward(predictions[t], output_text:select(2, t))    -- criterion forward
        end

        local reason_pool
        local loss_2 = 0
        if opt.use_noun then
            reason_pool = model.pooling:forward(reason_pred_mat):float()
            local t_loss = model.reason_criterion:forward(reason_pool, noun_list) * opt.reason_weight
            if update then loss = loss + t_loss else loss_2 = loss_2 + t_loss end
        end
                    
        ------------------- backward pass -------------------
        if update then
            local dreason_pred
            if opt.use_noun then
                dreason_pred = model.reason_criterion:backward(reason_pool, noun_list):cuda() * opt.reason_weight
                dreason_pred = model.pooling:backward(reason_pred_mat, dreason_pred)
            end

            local dembeddings = {}                                    -- d loss / d input embeddings
            local dlstm_c = {[seq_len]=dfinalstate_c}                 -- internal cell states of LSTM
            local dlstm_h = {}                                        -- output values of LSTM
            
            for t = seq_len, 1, -1 do
                if DEBUG_LEN then print('Backward time step ' .. t) end
                local doutput_t = clones.criterion[t]:backward(predictions[t], output_text:select(2, t))  -- criterion backward
                if t == seq_len then
                    assert(dlstm_h[t] == nil)
                    dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
                else
                    dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))     -- softmax backward
                end

                if opt.use_noun then
                    local d_reason_output_t = clones.reason_softmax[t]:backward(lstm_h[t], dreason_pred:select(2, t))
                    dlstm_h[t]:add(d_reason_output_t)
                end
                
                -- backprop through LSTM timestep
                if opt.use_attention then
                    dembeddings[t], _, dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.soft_att_lstm[t]:
                        backward({embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]},
                        {dlstm_c[t], dlstm_h[t]}))                                           -- lstm backward
                else
                    dembeddings[t], dlstm_c[t - 1], dlstm_h[t - 1] = unpack(clones.soft_att_lstm[t]:
                        backward({embeddings[t], lstm_c[t - 1], lstm_h[t - 1]},
                        {dlstm_c[t], dlstm_h[t]}))
                end

                -- backprop through embeddings
                clones.emb[t]:backward(input_text:select(2, t), dembeddings[t])          -- emb backward
            end

            if opt.lstm_size ~= opt.fc7_size then
                dlstm_c[0]:add(dlstm_h[0])
                model.linear:backward(fc7_images, dlstm_c[0])
            end
        end
        
        grad_params:clamp(-5, 5)
        return loss, update and grad_params or loss_2
    end 
    --- end of feval

    local function comp_error(batches)
        local loss = 0
        local loss_2 = 0
        for j = 1, opt.max_eval_batch do
            if j > #batches then break end
            att_seq, fc7_images, input_text, output_text, noun_list = dataloader:gen_train_data(batches[j])
            local t_loss, t_loss_2 = feval(params, false)
            loss = loss + t_loss
            loss_2 = loss_2 + t_loss_2
        end
        return loss, loss_2
    end
    
    local max_bleu_4 = 0
    for epoch = 1, opt.nEpochs do
        local index = torch.randperm(#batches)
        for i = 1, #batches do
            if DEBUG_LEN and #batches[index[i]][1][2] < max_t then goto continue end
            att_seq, fc7_images, input_text, output_text, noun_list = dataloader:gen_train_data(batches[index[i]])
            optim.adagrad(feval, params, optim_state)
            if DEBUG_LEN then goto continue end
            
            ----------------- Evaluate the model in validation set ----------------
            if i == 1 or i % opt.loss_period == 0 then
                train_loss, train_loss_2 = comp_error(batches)
                val_loss, val_loss_2 = comp_error(val_batches)
                print(epoch, i, 'train', train_loss, train_loss_2, 'val', val_loss, val_loss_2)
                collectgarbage()
            end

            if i == 1 or i % opt.eval_period == 0 then
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
                    
                    ------------------- forward pass -------------------
                    local embeddings = {}              -- input text embeddings
                    local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
                    local lstm_h = {[0]=initstate_h}   -- output values of LSTM
                    local predictions = {}             -- softmax outputs
                    local max_pred = {[1] = init_input}                -- max outputs 
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
                        if j1 + k - 1 <= 10 then
                            print(dataloader.val_set[j1 + k - 1], caption)
                        end
                        table.insert(captions, {image_id = dataloader.val_set[j1 + k - 1], caption = caption})
                    end
                    j1 = j2 + 1
                end

                local eval_struct = M.language_eval(captions, 'attention')
                local bleu_4 = eval_struct.Bleu_4

                if bleu_4 > max_bleu_4 then
                    max_bleu_4 = bleu_4
                    if opt.save_file then
                        torch.save('models/' .. opt.save_file_name, model)
                    end
                end
                print(epoch, i, 'max_bleu', max_bleu_4, 'bleu', bleu_4)
            end
            
            ::continue::
        end
        -- end of for i
    end
    -- end of for epoch
end


-------------------------
-- create the final model
-------------------------
function M.create_model(opt)
    local model = {}
    model.emb = nn.LookupTable(opt.word_cnt, opt.emb_size)
    -- model.soft_att_lstm = opt.use_attention and M.soft_att_lstm(opt) or M.lstm(opt)
    -- model.soft_att_lstm = opt.use_attention and M.soft_att_lstm_2(opt) or M.lstm(opt)    -- new attention 
    model.soft_att_lstm = opt.use_attention and M.soft_att_lstm_concat(opt) or M.lstm(opt)
    
    
    model.softmax = nn.Sequential():add(nn.Linear(opt.lstm_size, opt.word_cnt)):add(nn.LogSoftMax())
    model.criterion = nn.ClassNLLCriterion()
    if opt.lstm_size ~= opt.fc7_size then
        model.linear = nn.Linear(opt.fc7_size, opt.lstm_size)
    end
    if opt.use_noun then
        local out_dim = opt.word_cnt
        model.reason_softmax = nn.Linear(opt.lstm_size, out_dim)
        model.pooling = nn.Max(2)
        model.reason_criterion = nn.MultiLabelMarginCriterion()
    end
    
    if opt.nGPU > 0 then
        model.emb:cuda()
        model.soft_att_lstm:cuda()
        model.softmax:cuda()
        model.criterion:cuda()
        if opt.lstm_size ~= opt.fc7_size then
            model.linear:cuda()
        end
        if opt.use_noun then
            model.reason_softmax:cuda()
            model.pooling:cuda()
        end
    end
    return model
end

-------------------------
-- Eval the model
-------------------------
function M.language_eval(predictions, id)
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

return M




