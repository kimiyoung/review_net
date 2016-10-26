
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'masker'
require 'not_op'
require 'mask_table'
require 'mixture'
require 'linear3d'
local data = require 'data'
local model_utils = require 'model_utils'

local model = {}

local MAX_EPOCH = 100
local LSTM_INPUT_SIZE = 50
local LSTM_OUTPUT_SIZE = 256
local LEARNING_RATE = 1e-3
local CLAMP = 1.0
local EVAL_EVERY = 400
local MAX_EVAL_BATCH = 300
local ATT_NEXT_H = false
local REASON_STEP = 8
local LINEAR_TRANS = 0
local REASON_AFTER = 1
local MERGE_LINEAR = 0

-- local MODEL_FILE = 'reason.' .. REASON_STEP .. '.' .. LINEAR_TRANS .. '.model'

local MODEL_FILE = string.format("reason.step%d.litrans%d.after%d.merge%d.model", REASON_STEP, LINEAR_TRANS, REASON_AFTER, MERGE_LINEAR)
print('MODEL_FILE', MODEL_FILE)

function model.lstm(input_size, rnn_size)
    local x = nn.Identity()()
    local mask = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

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

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local n_mask = nn.NotOp()(mask)

    next_c = nn.CAddTable(){
        nn.MaskTable(){next_c, mask},
        nn.MaskTable(){prev_c, n_mask}
    }

    next_h = nn.CAddTable(){
        nn.MaskTable(){next_h, mask},
        nn.MaskTable(){prev_h, n_mask}
    }

    return nn.gModule({x, mask, prev_c, prev_h}, {next_c, next_h})
end

function model.att_lstm(input_size, rnn_size)
    local x = nn.Identity()()
    local mask = nn.Identity()()
    local h_src = nn.Identity()() -- batch * src_len * hid
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

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

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * hid

    local dot = nn.Mixture(3){prev_h, h_src} -- batch * src_len
    local weight = nn.SoftMax()(dot)
    local h_src_t = nn.Transpose({2, 3})(h_src) -- batch * hid * src_len
    local att_res = nn.Mixture(3){weight, h_src_t} -- batch * hid
    if LINEAR_TRANS > 0 then att_res = nn.Linear(rnn_size, rnn_size)(att_res) end
    local next_h_res = next_h
    if LINEAR_TRANS > 0 then next_h_res = nn.Linear(rnn_size, rnn_size)(next_h_res) end
    local next_att = nn.CAddTable(){att_res, next_h_res} -- batch * hid

    local n_mask = nn.NotOp()(mask)

    next_c = nn.CAddTable(){
        nn.MaskTable(){next_c, mask},
        nn.MaskTable(){prev_c, n_mask}
    }

    next_h = nn.CAddTable(){
        nn.MaskTable(){next_h, mask},
        nn.MaskTable(){prev_h, n_mask}
    }

    return nn.gModule({x, mask, h_src, prev_c, prev_h}, {next_c, next_h, next_att})
end

function model.reason(rnn_size)
    local h_src = nn.Identity()() -- batch * src_len * hid
    local src_mask = nn.Identity()() -- batch * src_len
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    local dot = nn.Mixture(3){prev_h, h_src} -- batch * src_len
    local weight = nn.SoftMax()(dot)
    weight = nn.CMulTable(){weight, src_mask} -- mask source sequence
    weight = nn.Normalize(1)(weight) -- batch * src_len, normalized distribution
    local h_src_t = nn.Transpose({2, 3})(h_src) -- batch * hid * src_len
    local att_res = nn.Mixture(3){weight, h_src_t} -- batch * hid
    -- local next_att = nn.CAddTable(){att_res, next_h} -- batch * hid

    local all_input_sums

    if REASON_AFTER == 0 then
        local att2h = nn.Linear(rnn_size, 4 * rnn_size)(att_res)
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
        all_input_sums = nn.CAddTable()({h2h, att2h})
    else
        all_input_sums = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    end

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * hid

    if REASON_AFTER > 0 then
        local att2h = nn.Linear(rnn_size, rnn_size)(att_res)
        next_h = nn.CAddTable(){att2h, next_h}
    end

    return nn.gModule({h_src, src_mask, prev_c, prev_h}, {next_c, next_h})
end

function model.merge_module(rnn_size)
    local forward_hid = nn.Identity()() -- {batch * hid} for src_len
    local backward_hid = nn.Identity()() -- {batch * hid} for src_len

    local forward_t = nn.JoinTable(2)(forward_hid) -- batch * (src_len * hid)
    forward_t = nn.View(-1, rnn_size):setNumInputDims(1)(forward_t) -- batch * src_len * hid
    local backward_t = nn.JoinTable(2)(backward_hid)
    backward_t = nn.View(-1, rnn_size):setNumInputDims(1)(backward_t) -- batch * src_len * hid

    if MERGE_LINEAR > 0 then
        forward_t = nn.Linear3d(rnn_size, rnn_size)(forward_t)
        backward_t = nn.Linear3d(rnn_size, rnn_size)(backward_t)
    end

    local hid = nn.CAddTable(){forward_t, backward_t} -- batch * src_len * hid

    return nn.gModule({forward_hid, backward_hid}, {hid})
end

function model.eval(test_batches, prefix, index2word)
    local M = torch.load(MODEL_FILE)

    local T1, T2 = data.CODE_TRUNCATE, data.COMMENT_TRUNCATE
    local T3 = REASON_STEP
    local lookup = model_utils.clone_many_times(M.lookup, T1 + T2)
    local enc_lookup, dec_lookup = {}, {}
    for t = 1, T1 do enc_lookup[t] = lookup[t] end
    for t = 1, T2 do dec_lookup[t] = lookup[T1 + t] end
    local enc_lstm = model_utils.clone_many_times(M.enc_lstm, T1)
    local rev_lstm = model_utils.clone_many_times(M.rev_lstm, T1)
    local dec_lstm = model_utils.clone_many_times(M.dec_lstm, T2)
    local linear = model_utils.clone_many_times(M.linear, T2)
    local mask = model_utils.clone_many_times(M.mask, T2)
    local criterion = model_utils.clone_many_times(M.criterion, T2)
    local comb = M.comb
    local merge = M.merge
    local reason = model_utils.clone_many_times(M.reason, T3)

    print('copy done')

    local enc_emb, enc_h, enc_c, dec_emb, dec_h, dec_c, dec_att, dec_linear, dec_mask = {}, {}, {}, {}, {}, {}, {}, {}
    local rev_h, rev_c = {}, {}
    local reason_h, reason_c = {}, {}

    local merge_h, reason_att

    local code_matrix, code_mask, comment_matrix, comment_mask, comment_next

    local avg_saved, com_sum = {0, 0, 0, 0, 0}, 0
    local input = torch.CudaTensor(1)

    for i = 1, #test_batches do
        local t_batch = test_batches[i]
        for j = 1, #t_batch do t_batch[j] = t_batch[j]:cuda() end
        code_matrix, code_mask, comment_matrix, comment_mask, comment_next = unpack(t_batch)

        local zero_tensor = torch.CudaTensor(code_matrix:size()[1], LSTM_OUTPUT_SIZE):zero()

        -- transpose to be contiguous
        local t_code_mask = nn.Transpose({1, 2}):cuda():forward(code_mask)
        local t_comment_mask = nn.Transpose({1, 2}):cuda():forward(comment_mask)

        rev_h, enc_h = {}, {} -- clear hidden states for the merge module
        enc_c[0] = zero_tensor:clone(); enc_h[0] = zero_tensor:clone()
        for t = 1, code_matrix:size()[2] do
            enc_emb[t] = enc_lookup[t]:forward(code_matrix:select(2, t))
            enc_c[t], enc_h[t] = unpack(enc_lstm[t]:forward{enc_emb[t], t_code_mask[t], enc_c[t - 1], enc_h[t - 1]})
        end
        rev_c[code_matrix:size()[2] + 1] = zero_tensor:clone()
        rev_h[code_matrix:size()[2] + 1] = zero_tensor:clone()
        for t = code_matrix:size()[2], 1, -1 do
            rev_c[t], rev_h[t] = unpack(rev_lstm[t]:forward{enc_emb[t], t_code_mask[t], rev_c[t + 1], rev_h[t + 1]})
        end

        enc_h[0], rev_h[code_matrix:size()[2] + 1] = nil, nil -- clear hidden states for the merge module
        merge_h = merge:forward{enc_h, rev_h}

        reason_h[0] = comb:forward{enc_h[code_matrix:size()[2]], rev_h[1]}
        reason_c[0] = reason_h[0]:clone()
        reason_att = torch.CudaTensor(code_matrix:size(1), T3, LSTM_OUTPUT_SIZE)
        for t = 1, T3 do
            reason_c[t], reason_h[t] = unpack(reason[t]:forward{merge_h, code_mask, reason_c[t - 1], reason_h[t - 1]})
            reason_att:select(2, t):copy(reason_h[t])
        end

        dec_h_0 = reason_h[T3]
        dec_c_0 = reason_c[T3]

        for j = 1, comment_matrix:size()[1] do
            local saved_sum, char_cnt = {0, 0, 0, 0, 0}, 0
            dec_c[0], dec_h[0] = dec_c_0[j]:reshape(1, LSTM_OUTPUT_SIZE), dec_h_0[j]:reshape(1, LSTM_OUTPUT_SIZE)
            for t = 1, comment_matrix:size()[2] do
                if comment_mask[j][t] > 0 then
                    input[1] = comment_matrix[j][t]
                    dec_emb[t] = dec_lookup[t]:forward(input)
                    dec_c[t], dec_h[t], dec_att[t] = unpack(dec_lstm[t]:forward{dec_emb[t], torch.CudaTensor({1}),
                        -- merge_h[j]:reshape(1, merge_h:size(2), merge_h:size(3)), code_mask[j]:reshape(1, code_mask:size(2)),
                        reason_att[j]:reshape(1, reason_att:size(2), reason_att:size(3)),
                        dec_c[t - 1], dec_h[t - 1]})
                    dec_linear[t] = linear[t]:forward(dec_att[t])
                    local output = dec_linear[t][1]:double()
                    local temp = prefix[comment_next[j][t]][(output - output[comment_next[j][t]]):gt(0)]
                    local max_2 = {-1, -1, -1, -1, -1}
                    if temp:nDimension() > 0 then
                        for k = 1, 5 do
                            max_2[k] = torch.max(temp)
                            local _, t_i = torch.max(temp, 1)
                            temp[t_i[1]] = -1
                        end
                    end
                    local cur_s = index2word[comment_next[j][t]]
                    for k = 1, 5 do
                        saved_sum[k] = saved_sum[k] + math.max(cur_s:len() - (max_2[k] + 1), 0)
                    end
                    char_cnt = char_cnt + cur_s:len()
                end
            end
            for k = 1, 5 do
                avg_saved[k] = avg_saved[k] + saved_sum[k] * 1.0 / char_cnt
            end
            com_sum = com_sum + 1
        end

        -- print('evaluating ' .. i .. ' out of ' .. #test_batches, com_sum > 0 and avg_saved / com_sum or 0.0)
    end

    for k = 1, 5 do
        print(k, avg_saved[k] / com_sum)
    end
end

function model.train(batches, test_batches, word_cnt, token_cnt, eval_llh)
    local M = {}

    if eval_llh then
        M = torch.load(MODEL_FILE)
    else
        M.lookup = nn.LookupTable(token_cnt, LSTM_INPUT_SIZE):cuda()
        M.enc_lstm = model.lstm(LSTM_INPUT_SIZE, LSTM_OUTPUT_SIZE):cuda()
        M.rev_lstm = model.lstm(LSTM_INPUT_SIZE, LSTM_OUTPUT_SIZE):cuda()
        M.merge = model.merge_module(LSTM_OUTPUT_SIZE):cuda()
        M.comb = nn.Sequential():add(nn.JoinTable(2, 2)):add(nn.Linear(LSTM_OUTPUT_SIZE * 2, LSTM_OUTPUT_SIZE)):cuda()
        M.reason = model.reason(LSTM_OUTPUT_SIZE):cuda()
        M.dec_lstm = model.att_lstm(LSTM_INPUT_SIZE, LSTM_OUTPUT_SIZE):cuda()
        M.linear = nn.Sequential():add(nn.Linear(LSTM_OUTPUT_SIZE, word_cnt)):add(nn.LogSoftMax()):cuda()
        M.mask = nn.Masker():cuda()
        M.criterion = nn.ClassNLLCriterion(nil, false):cuda() -- false: use sum over tokens
    end

    local x, dl_dx = model_utils.combine_all_parameters(M.lookup, M.enc_lstm, M.rev_lstm, M.merge, M.comb, M.dec_lstm, M.linear, M.mask)

    local T1, T2 = data.CODE_TRUNCATE, data.COMMENT_TRUNCATE
    local T3 = REASON_STEP
    local lookup = model_utils.clone_many_times(M.lookup, T1 + T2)
    local enc_lookup, dec_lookup = {}, {}
    for t = 1, T1 do enc_lookup[t] = lookup[t] end
    for t = 1, T2 do dec_lookup[t] = lookup[T1 + t] end
    local enc_lstm = model_utils.clone_many_times(M.enc_lstm, T1)
    local rev_lstm = model_utils.clone_many_times(M.rev_lstm, T1)
    local dec_lstm = model_utils.clone_many_times(M.dec_lstm, T2)
    local linear = model_utils.clone_many_times(M.linear, T2)
    local mask = model_utils.clone_many_times(M.mask, T2)
    local criterion = model_utils.clone_many_times(M.criterion, T2)
    local comb = M.comb
    local merge = M.merge
    local reason = model_utils.clone_many_times(M.reason, T3)

    print('copy done')

    local enc_emb, enc_h, enc_c, dec_emb, dec_h, dec_c, dec_att, dec_linear, dec_mask = {}, {}, {}, {}, {}, {}, {}, {}, {}
    local rev_h, rev_c = {}, {}
    local reason_h, reason_c = {}, {}
    local dec_grad_h, dec_grad_c, dec_grad_att, enc_grad_h, enc_grad_c = {}, {}, {}, {}, {}
    local rev_grad_h, rev_grad_c, grad_emb = {}, {}, {}
    local reason_grad_h, reason_grad_c = {}, {}

    -- new data structure for attention
    local merge_h; local merge_grad_h
    local reason_att; local reason_att_grad

    local code_matrix, code_mask, comment_matrix, comment_mask, comment_next

    local function feval(x_new, update)
        if update == nil then update = true end -- update = false: evaluation only
        if x_new ~= x then x:copy(x_new) end
        dl_dx:zero()

        local zero_tensor = torch.CudaTensor(code_matrix:size()[1], LSTM_OUTPUT_SIZE):zero()
        local loss = 0

        -- transpose to be contiguous
        local t_code_mask = nn.Transpose({1, 2}):cuda():forward(code_mask)
        local t_comment_mask = nn.Transpose({1, 2}):cuda():forward(comment_mask)

        rev_h, enc_h = {}, {} -- clear hidden states for the merge module
        enc_c[0] = zero_tensor:clone(); enc_h[0] = zero_tensor:clone()
        for t = 1, code_matrix:size()[2] do
            enc_emb[t] = enc_lookup[t]:forward(code_matrix:select(2, t))
            enc_c[t], enc_h[t] = unpack(enc_lstm[t]:forward{enc_emb[t], t_code_mask[t], enc_c[t - 1], enc_h[t - 1]})
        end
        rev_c[code_matrix:size()[2] + 1] = zero_tensor:clone()
        rev_h[code_matrix:size()[2] + 1] = zero_tensor:clone()
        for t = code_matrix:size()[2], 1, -1 do
            rev_c[t], rev_h[t] = unpack(rev_lstm[t]:forward{enc_emb[t], t_code_mask[t], rev_c[t + 1], rev_h[t + 1]})
        end

        enc_h[0], rev_h[code_matrix:size()[2] + 1] = nil, nil -- clear hidden states for the merge module
        merge_h = merge:forward{enc_h, rev_h}

        reason_h[0] = comb:forward{enc_h[code_matrix:size()[2]], rev_h[1]}
        reason_c[0] = reason_h[0]:clone()
        reason_att = torch.CudaTensor(code_matrix:size(1), T3, LSTM_OUTPUT_SIZE)
        for t = 1, T3 do
            reason_c[t], reason_h[t] = unpack(reason[t]:forward{merge_h, code_mask, reason_c[t - 1], reason_h[t - 1]})
            reason_att:select(2, t):copy(reason_h[t])
        end

        -- dec_h[0] = comb:forward{enc_h[code_matrix:size()[2]], rev_h[1]}
        -- dec_c[0] = dec_h[0]:clone()
        dec_h[0] = reason_h[T3]
        dec_c[0] = reason_c[T3]
        for t = 1, comment_matrix:size()[2] do
            dec_emb[t] = dec_lookup[t]:forward(comment_matrix:select(2, t))
            dec_c[t], dec_h[t], dec_att[t] = unpack(dec_lstm[t]:forward{dec_emb[t], t_comment_mask[t], reason_att, dec_c[t - 1], dec_h[t - 1]})
            dec_linear[t] = linear[t]:forward(dec_att[t])
            dec_mask[t] = mask[t]:forward{dec_linear[t], t_comment_mask[t]}
            loss = loss + criterion[t]:forward(dec_mask[t], nn.Masker():cuda():forward{comment_next:select(2, t), t_comment_mask[t]})
        end
        loss = loss / comment_mask:sum() -- normalize over tokens

        if update then
            dec_grad_h[comment_matrix:size()[2]], dec_grad_c[comment_matrix:size()[2]] = zero_tensor:clone(), zero_tensor:clone()
            reason_att_grad = reason_att:clone():zero()
            for t = comment_matrix:size()[2], 1, -1 do
                local grad = criterion[t]:backward(dec_mask[t], nn.Masker():cuda():forward{comment_next:select(2, t), t_comment_mask[t]})
                grad, _ = unpack(mask[t]:backward({dec_linear[t], t_comment_mask[t]}, grad))
                grad = linear[t]:backward(dec_att[t], grad)
                -- dec_grad_h[t]:add(grad)
                local t_reason_att_grad
                grad, _, t_reason_att_grad, dec_grad_c[t - 1], dec_grad_h[t - 1] = unpack(dec_lstm[t]:backward(
                    {dec_emb[t], t_comment_mask[t], reason_att, dec_c[t - 1], dec_h[t - 1]},
                    {dec_grad_c[t], dec_grad_h[t], grad}
                ))
                reason_att_grad:add(t_reason_att_grad)
                dec_lookup[t]:backward(comment_matrix:select(2, t), grad)
            end

            reason_grad_c[T3] = dec_grad_c[0]
            reason_grad_h[T3] = dec_grad_h[0]
            merge_grad_h = merge_h:clone():zero()
            for t = T3, 1, -1 do
                reason_grad_h[t]:add(reason_att_grad:select(2, t))
                local t_merge_grad_h
                t_merge_grad_h, _, reason_grad_c[t - 1], reason_grad_h[t - 1] = unpack(reason[t]:backward(
                    {merge_h, code_mask, reason_c[t - 1], reason_h[t - 1]},
                    {reason_grad_c[t], reason_grad_h[t]}))
                merge_grad_h:add(t_merge_grad_h)
            end
            enc_grad_h, rev_grad_h = unpack(merge:backward({enc_h, rev_h}, merge_grad_h))
            local grad_1, grad_2 = unpack(comb:backward(
                {enc_h[code_matrix:size()[2]], rev_h[1]},
                -- dec_grad_h[0] + dec_grad_c[0]
                reason_grad_h[0] + reason_grad_c[0]
            ))
            enc_grad_h[code_matrix:size()[2]]:add(grad_1)
            rev_grad_h[1]:add(grad_2)
            enc_grad_c[code_matrix:size()[2]], rev_grad_c[1] = zero_tensor:clone(), zero_tensor:clone()
            for t = code_matrix:size()[2], 1, -1 do
                local grad
                grad_emb[t], _, enc_grad_c[t - 1], grad_2 = unpack(enc_lstm[t]:backward(
                    {enc_emb[t], t_code_mask[t], enc_c[t - 1], dec_h[t - 1]},
                    {enc_grad_c[t], enc_grad_h[t]}
                ))
                if t > 1 then enc_grad_h[t - 1]:add(grad_2) end
            end
            for t = 1, code_matrix:size()[2] do
                local grad
                grad, _, rev_grad_c[t + 1], grad_2 = unpack(rev_lstm[t]:backward(
                    {enc_emb[t], t_code_mask[t], rev_c[t + 1], rev_h[t + 1]},
                    {rev_grad_c[t], rev_grad_h[t]}
                ))
                if t < code_matrix:size()[2] then rev_grad_h[t + 1]:add(grad_2) end
                grad_emb[t]:add(grad)
            end
            for t = 1, code_matrix:size()[2] do
                enc_lookup[t]:backward(code_matrix:select(2, t), grad_emb[t])
            end

            dl_dx:div(comment_mask:sum()) -- normalize over tokens
            dl_dx:clamp(- CLAMP, CLAMP)
        end

        return loss, dl_dx
    end

    local function comp_error(batches)
        local loss = 0; local inst_cnt = 0
        local n_batch = math.min(#batches, MAX_EVAL_BATCH)
        for j = 1, n_batch do
            local t_batch = batches[j]
            for k = 1, #t_batch do t_batch[k] = t_batch[k]:cuda() end
            code_matrix, code_mask, comment_matrix, comment_mask, comment_next = unpack(t_batch)
            local t_loss, _ = feval(x, false) -- false: don't update parameters
            loss = loss + t_loss * code_matrix:size()[1]
            inst_cnt = inst_cnt + code_matrix:size()[1]
        end
        return loss / inst_cnt
    end

    if eval_llh then
        print('test loss', comp_error(test_batches))
        return
    end

    local min_train_loss, min_test_loss = 1e6, 1e6

    for epoch = 1, MAX_EPOCH do
        local index = torch.randperm(#batches)
        for i = 1, #batches do
            local t_batch = batches[index[i]]
            for j = 1, #t_batch do t_batch[j] = t_batch[j]:cuda() end
            code_matrix, code_mask, comment_matrix, comment_mask, comment_next = unpack(t_batch)

            optim.adagrad(feval, x, {learningRate = LEARNING_RATE})

            if i == 1 or i % EVAL_EVERY == 0 then
                local train_loss = comp_error(batches)
                local test_loss = comp_error(test_batches)

                if test_loss < min_test_loss then
                    torch.save(MODEL_FILE, M)
                end

                min_train_loss = math.min(min_train_loss, train_loss)
                min_test_loss = math.min(min_test_loss, test_loss)
                print(epoch, i, 'train', train_loss, 'test', test_loss)
                print('min_train', min_train_loss, 'min_test', min_test_loss)
                collectgarbage()
            end
        end
    end
end

return model
