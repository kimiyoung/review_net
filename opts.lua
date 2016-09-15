local M = {}

function M.parse(arg)    
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Image Captioning')
    cmd:text()
    cmd:text('Options:')

    ------------ Model options ----------------------
    cmd:option('-emb_size', 100, 'Word embedding size') -- 100
    cmd:option('-lstm_size', 1024, 'LSTM size') -- 1024
    cmd:option('-att_size', 196, 'how many attention areas')
    cmd:option('-feat_size', 512, 'the dimension of each attention area')
    cmd:option('-fc7_size', 4096, 'the dimension of fc7')
    cmd:option('-att_hid_size', 512, 'the hidden size of the attention MLP; 0 if not using hidden layer')
    
    cmd:option('-val_size', 4000, 'Validation set size')
    cmd:option('-test_size', 4000, 'Test set size')

    cmd:option('-use_attention', true, 'Use attention or not')
    cmd:option('-use_noun', true, 'Use noun or not') -- true
    cmd:option('-reason_weight', 10.0, 'weight of reasoning loss')

    cmd:option('-model_pack', 'reason_att_copy', 'the model package to use, can be reason_att, reasoning, or soft_att_lstm')
    cmd:option('-reason_step', 8, 'Reasoning steps before the decoder')

    ------------ General options --------------------
    cmd:option('-data', 'data/', 'Path to dataset')
    cmd:option('-train_feat', 'train2014_features_vgg_vd19_conv5', 'Path to pre-extracted training image feature')
    cmd:option('-val_feat', 'val2014_features_vgg_vd19_conv5', 'Path to pre-extracted validation image feature')
    cmd:option('-test_feat', 'test2014_features_vgg_vd19_conv5', 'Path to pre-extracted test image feature')

    cmd:option('-train_fc7', 'train2014_features_vgg_vd19_fc7', 'Path to pre-extracted training fully connected 7')
    cmd:option('-val_fc7', 'val2014_features_vgg_vd19_fc7', 'Path to pre-extracted validation fully connected 7')
    cmd:option('-test_fc7', 'test2014_features_vgg_vd19_fc7', 'Path to pre-extracted test fully connected 7')

    cmd:option('-train_anno', 'annotations/captions_train2014.json', 'Path to training image annotaion file')
    cmd:option('-val_anno', 'annotations/captions_val2014.json', 'Path to validation image annotaion file')
    cmd:option('-nGPU', 1, 'Index of GPU to use, 0 means CPU')
    cmd:option('-seed', 13, 'Random number seed')

    cmd:option('-arctic_dir', 'arctic-captions/splits', 'Path to index file')

    ------------ Training options --------------------
    cmd:option('-nEpochs', 100, 'Number of epochs in training')
    cmd:option('-batch_size', 32, 'Batch size in SGD')
    cmd:option('-val_batch_size', 10, 'Batch size for testing')
    cmd:option('-LR', 1e-2, 'Initial learning rate') -- 1e-2
    cmd:option('-cnn_LR', 0, 'Learning rate for cnn')
    cmd:option('-truncate', 30, 'Text longer than this size gets truncated. -1 for no truncation.') -- 30
    cmd:option('-max_eval_batch', 50, 'max number of instances when calling comp error. 20000 = 4000 * 5')

    cmd:option('-save_file', false, 'whether save model file?')
    cmd:option('-save_file_name', 'review.vgg19.model', 'file name for saving model')

    cmd:option('-load_file', false, 'whether load model file?')
    cmd:option('-load_vgg_file', false)
    cmd:option('-load_file_name', 'review.vgg19.model')

    cmd:option('-train_only', false, 'if true then use 80k, else use 110k')
    cmd:option('-early_stop', 'cider', 'can be cider or bleu')
    cmd:option('-gen_dropout', 0.1) -- 0.1
    
    ------------ Evaluation options --------------------
    -- cmd:option('-model', 'copy.all.val.8.w10.noun.model', 'Model to evaluate')
    -- cmd:option('-model', 'copy.google.val.all.8.noun.w10.model', 'Model to evaluate')
    cmd:option('-eval_algo', 'beam', 'Evaluation algorithm, beam or greedy')
    cmd:option('-beam_size', 3, 'Beam size in beam search') -- 3
    cmd:option('-val_max_len', 20, 'Max length in validation state')

    cmd:option('-test_mode', false, 'eval on test set if true') --
    cmd:option('-server_train_mode', false, 'eval on test of val, and use the rest for training')
    cmd:option('-server_test_mode', false, 'eval on server test set if true; if true then test_mode will be false.')
    
    local opt = cmd:parse(arg or {})
    opt.eval_period = math.floor(3000 * 32 / opt.batch_size) * 2
    opt.loss_period = math.floor(600 * 32 / opt.batch_size)
    if opt.use_cat then opt.use_noun = false end
    if opt.server_test_mode then opt.test_mode = false end
    if opt.server_train_mode then opt.test_mode = false end
    opt.model = opt.load_file_name
    return opt
end

return M





