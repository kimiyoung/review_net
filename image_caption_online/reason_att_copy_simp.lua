require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'
local tablex = require 'pl.tablex'
local anno_utils = require 'utils.anno_utils_filter'

local M = {}

function M.lstm(opt)
    -- Model parameters
    local rnn_size = opt.lstm_size
    local input_size = opt.emb_size

    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)

    local all_input_sums = nn.CAddTable()({i2h, h2h})

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

    next_h = nn.Dropout(opt.gen_dropout)(next_h)
    
    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

-- Attention model, concat hidden and image feature
function M.soft_att_lstm_concat_nox(opt)
    -- Model parameters
    local feat_size = opt.feat_size
    local att_size = opt.att_size
    local rnn_size = opt.lstm_size
    local input_size = opt.fc7_size
    local att_hid_size = opt.att_hid_size

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

    local bn_wh, bn_att, bn_c = nn.Identity(), nn.Identity(), nn.Identity()
    
    --- Input to LSTM
    local att_add = bn_att(nn.Linear(feat_size, 4 * rnn_size)(att_res))   -- batch * (4*rnn_size) <- batch * feat_size

    ------------- LSTM main part --------------------
    local h2h = bn_wh(nn.Linear(rnn_size, 4 * rnn_size)(prev_h))
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({h2h, att_add})

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
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(bn_c(next_c))}) -- batch * rnn_size

    next_h = nn.Dropout(opt.reason_dropout)(next_h)
    
    return nn.gModule({att_seq, prev_c, prev_h}, {next_c, next_h})
end

-------------------------------------
-- Train the model for an epoch
-- INPUT
-- batches: {{id, caption}, ..., ...}
-------------------------------------
function M.train(model, opt, batches, val_batches, optim_state, dataloader)
    local params, grad_params
    local model_list = {model.emb, model.lstm, model.softmax}
    for t = 1, opt.reason_step do
        table.insert(model_list, model.soft_att_lstm[t])
    end
    params, grad_params = model_utils.combine_all_parameters(unpack(model_list))
    local clones = {}
    anno_utils = dataloader.anno_utils
    
    -- Clone models
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'soft_att_lstm' and name ~= 'reason_softmax' and name ~= 'reason_criterion'
            and name ~= 'pooling' and name ~= 'linear' and name ~= 'google_linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    print('cloning reasoning lstm')
    -- clones.soft_att_lstm = model_utils.clone_many_times(model.soft_att_lstm, opt.reason_step)
    clones.soft_att_lstm = model.soft_att_lstm

    local att_seq, fc7_images, input_text, output_text, noun_list, fc7_google_images

    local function evaluate()
        for t = 1, opt.reason_step do clones.soft_att_lstm[t]:evaluate() end
        for t = 1, max_t do clones.lstm[t]:evaluate() end
    end

    local function training()
        for t = 1, opt.reason_step do clones.soft_att_lstm[t]:training() end
        for t = 1, max_t do clones.lstm[t]:training() end
    end

    local function feval(params_, update)
        if update == nil then update = true end
        if update then training() else evaluate() end
        if params_ ~= params then
            params:copy(params_)
        end
        grad_params:zero()

        local image_map = fc7_images

        local zero_tensor = torch.zeros(input_text:size()[1], opt.lstm_size):cuda()
        local reason_c = {[0] = image_map}
        local reason_h = {[0] = image_map}
        local embeddings, lstm_c, lstm_h, predictions = {}, {}, {}, {}
        local loss = 0
        local seq_len = math.min(input_text:size()[2], max_t)
        local reason_len = opt.reason_step
        
        for t = 1, reason_len do
            reason_c[t], reason_h[t] = unpack(clones.soft_att_lstm[t]:
                forward{att_seq, reason_c[t - 1], reason_h[t - 1]})
        end

        lstm_c[0] = reason_c[reason_len]
        lstm_h[0] = reason_h[reason_len]

        for t = 1, seq_len do
            embeddings[t] = clones.emb[t]:forward(input_text:select(2, t))
            lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:
                forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
            predictions[t] = clones.softmax[t]:forward(lstm_h[t])
            loss = loss + clones.criterion[t]:forward(predictions[t], output_text:select(2, t))
        end

        if update then

            local dembeddings, dlstm_c, dlstm_h, dreason_c, dreason_h = {}, {}, {}, {}, {}
            dlstm_c[seq_len] = zero_tensor:clone()
            dlstm_h[seq_len] = zero_tensor:clone()

            for t = seq_len, 1, -1 do
                local doutput_t = clones.criterion[t]:backward(predictions[t], output_text:select(2, t))
                dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
                dembeddings[t], dlstm_c[t - 1], dlstm_h[t - 1] = unpack(clones.lstm[t]:
                    backward({embeddings[t], lstm_c[t - 1], lstm_h[t - 1]},
                    {dlstm_c[t], dlstm_h[t]}))
                clones.emb[t]:backward(input_text:select(2, t), dembeddings[t])
            end
            dreason_c[reason_len] = dlstm_c[0]
            dreason_h[reason_len] = dlstm_h[0]
            for t = reason_len, 1, -1 do
                _, dreason_c[t - 1], dreason_h[t - 1] = unpack(clones.soft_att_lstm[t]:
                    backward({att_seq, reason_c[t - 1], reason_h[t - 1]},
                    {dreason_c[t], dreason_h[t]}))
            end
        end
        
        grad_params:clamp(-5, 5)
        return loss, grad_params
    end 
    --- end of feval

    local function comp_error(batches)
        local loss = 0
        for j = 1, opt.max_eval_batch do
            if j > #batches then break end
            att_seq, fc7_images, input_text, output_text, noun_list, fc7_google_images = dataloader:gen_train_data(batches[j])
            local t_loss, _ = feval(params, false)
            loss = loss + t_loss
        end
        return loss
    end
    
    local max_bleu_4 = 0
    for epoch = 1, opt.nEpochs do
        local index = torch.randperm(#batches)
        for i = 1, #batches do
            att_seq, fc7_images, input_text, output_text, noun_list, fc7_google_images = dataloader:gen_train_data(batches[index[i]])
            optim.adagrad(feval, params, optim_state)
            
            ----------------- Evaluate the model in validation set ----------------
            if i == 1 or i % opt.loss_period == 0 then
                evaluate()
                train_loss = comp_error(batches)
                val_loss = comp_error(val_batches)
                print(epoch, i, 'train', train_loss, 'val', val_loss)
                collectgarbage()
            end

            if i == 1 or i % opt.eval_period == 0 then
                evaluate()
                local captions = {}
                local j1 = 1
                while j1 <= #dataloader.val_set do
                    local j2 = math.min(#dataloader.val_set, j1 + opt.val_batch_size)
                    att_seq, fc7_images, fc7_google_images = dataloader:gen_test_data(j1, j2)

                    local image_map = fc7_images

                    local reason_c = {[0] = image_map}
                    local reason_h = {[0] = image_map}
                    local embeddings, lstm_c, lstm_h, predictions, max_pred = {}, {}, {}, {}, {}
                    local reason_len = opt.reason_step
                    local seq_len = max_t
                    
                    for t = 1, reason_len do
                        reason_c[t], reason_h[t] = unpack(clones.soft_att_lstm[t]:
                            forward{att_seq, reason_c[t - 1], reason_h[t - 1]})
                    end

                    lstm_c[0] = reason_c[reason_len]
                    lstm_h[0] = reason_h[reason_len]
                    max_pred[1] = torch.CudaTensor(att_seq:size()[1]):fill(anno_utils.START_NUM)

                    for t = 1, seq_len do
                        embeddings[t] = clones.emb[t]:forward(max_pred[t])
                        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:
                            forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
                        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
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
                local bleu_4
                if opt.early_stop == 'cider' then
                    bleu_4 = eval_struct.CIDEr
                else
                    bleu_4 = eval_struct.Bleu_4
                end

                if bleu_4 > max_bleu_4 then
                    max_bleu_4 = bleu_4
                    if opt.save_file then
                        torch.save('models/' .. opt.save_file_name, model)
                    end
                end
                if opt.early_stop == 'cider' then
                    print(epoch, i, 'max_cider', max_bleu_4, 'cider', bleu_4)
                else
                    print(epoch, i, 'max_bleu', max_bleu_4, 'bleu', bleu_4)
                end
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
    -- model.soft_att_lstm = M.soft_att_lstm_concat_nox(opt)
    model.soft_att_lstm = {}
    for t = 1, opt.reason_step do
        model.soft_att_lstm[t] = M.soft_att_lstm_concat_nox(opt)
    end
    model.lstm = M.lstm(opt)
    model.softmax = nn.Sequential():add(nn.Linear(opt.lstm_size, opt.word_cnt)):add(nn.LogSoftMax())
    model.criterion = nn.ClassNLLCriterion()
    
    if opt.nGPU > 0 then
        model.emb:cuda()
        for _, m in ipairs(model.soft_att_lstm) do
            m:cuda()
        end
        model.lstm:cuda()
        model.softmax:cuda()
        model.criterion:cuda()
    end
    return model
end

-------------------------
-- Eval the model
-------------------------
function M.language_eval(predictions, id)
    print('using reasoning att')
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

return M




