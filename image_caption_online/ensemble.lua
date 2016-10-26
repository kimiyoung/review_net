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
-- print(opt)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

local model_set = {}
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed13.model'))
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed23.model'))
table.insert(model_set, torch.load('models/reason_att_copy_simp_seed33.model'))

-- Initialize dataloader
local dataloader = DataLoader(opt)

local batches = dataloader:gen_batch(dataloader.train_len2captions, opt.batch_size)
local val_batches = dataloader:gen_batch(dataloader.val_len2captions, opt.batch_size)

function create_model()
    -- single instance only
    local model = nn.Sequential() -- (batch, word, model)
    model:add(nn.View(-1, #model_set):setNumInputDims(3)) -- (batch * word, model)
    model:add(nn.Linear(#model_set, 1)):add(nn.View(-1, opt.word_cnt):setNumInputDims(2)) -- (batch, word)
    model:add(nn.LogSoftMax()) -- (batch, word)
    model:cuda()

    local criterion = nn.ClassNLLCriterion()
    criterion:cuda()

    return model, criterion
end

function language_eval(predictions, id)
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

function train(model_set, model, criterion)
    local params, grad_params = model:getParameters()

    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len

    for ii = 1, #model_set do model_set[ii].softmax:remove() end

    print('cloning ensemble model')
    local model_clones = model_utils.clone_many_times(model, max_t)
    print('cloning ensemble criterion')
    local criterion_clones = model_utils.clone_many_times(criterion, max_t)

    local att_seq, fc7_images, input_text, output_text, noun_list, fc7_google_images

    local function evaluate()
        for ii = 1, #model_set do
            for t = 1, opt.reason_step do model_set[ii].soft_att_lstm[t]:evaluate() end
            model_set[ii].lstm:evaluate()
        end
    end

    local function feval(params_, update)
        if update == nil then update = true end
        if not update then evaluate() end
        if params_ ~= params then
            params:copy(params_)
        end
        grad_params:zero()

        local image_map = fc7_images

        local predictions = torch.CudaTensor(max_t, input_text:size(1), opt.word_cnt, #model_set)
        local seq_len = math.min(input_text:size()[2], max_t)
        for ii = 1, #model_set do
            local c, h = image_map, image_map
            local reason_len = opt.reason_step
            
            for t = 1, reason_len do
                c, h = unpack(model_set[ii].soft_att_lstm[t]:
                    forward{att_seq, c, h})
            end

            for t = 1, seq_len do
                local embeddings = model_set[ii].emb:forward(input_text:select(2, t))
                c, h = unpack(model_set[ii].lstm:
                    forward{embeddings, c, h})
                predictions[{t, {}, {}, ii}] = model_set[ii].softmax:forward(h)
            end
        end

        local loss = 0
        local combine = {}
        for t = 1, seq_len do
            combine[t] = model_clones[t]:forward(predictions[t])
            loss = loss + criterion_clones[t]:forward(combine[t], output_text:select(2, t))
        end

        if update then
            for t = 1, seq_len do
                local d_output = criterion_clones[t]:backward(combine[t], output_text:select(2, t))
                model_clones[t]:backward(predictions[t], d_output)
            end
        end
        
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
            optim.sgd(feval, params, {learningRate = opt.ensemble_LR})
            
            ----------------- Evaluate the model in validation set ----------------
            if i == 1 or i % opt.loss_period == 0 then
                train_loss = comp_error(batches)
                val_loss = comp_error(val_batches)
                print(epoch, i, 'train', train_loss, 'val', val_loss)
                -- print(model:get(2).weight)
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

                    local cs, hs = {}, {}
                    local seq_len = max_t

                    for ii = 1, #model_set do
                        local c, h = image_map, image_map
                        
                        for t = 1, opt.reason_step do
                            c, h = unpack(model_set[ii].soft_att_lstm[t]:
                                forward{att_seq, c, h})
                        end
                        cs[ii], hs[ii] = c, h
                    end

                    local max_pred = {}
                    max_pred[1] = torch.CudaTensor(att_seq:size()[1]):fill(anno_utils.START_NUM)
                    local predictions = torch.CudaTensor(att_seq:size(1), opt.word_cnt, #model_set)

                    for t = 1, seq_len do
                        for ii = 1, #model_set do
                            local embeddings = model_set[ii].emb:forward(max_pred[t])
                            cs[ii], hs[ii] = unpack(model_set[ii].lstm:
                                forward{embeddings, cs[ii], hs[ii]})
                            predictions[{{}, {}, ii}] = model_set[ii].softmax:forward(hs[ii])
                        end
                        local combine = model_clones[t]:forward(predictions)
                        _, max_pred[t + 1] = torch.max(combine, 2)
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

                local eval_struct = language_eval(captions, 'attention')
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

local model, criterion = create_model()
train(model_set, model, criterion)

