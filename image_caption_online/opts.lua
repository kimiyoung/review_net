local M = {}

function M.parse(arg)    
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Image Captioning')
    cmd:text()
    cmd:text('Options:')

    ------------ Model options ----------------------
    cmd:option('-emb_size', 100, 'Word embedding size') -- 100
    cmd:option('-lstm_size', 2048, 'LSTM size') -- 1024
    cmd:option('-att_size', 64, 'how many attention areas') -- 14 * 14
    cmd:option('-feat_size', 1280, 'the dimension of each attention area')
    cmd:option('-fc7_size', 2048, 'the dimension of fc7') -- 4096
    cmd:option('-google_fc7_size', 1024, 'the dimension of google last layer')
    -- cmd:option('-fc7_size', 1024, 'the dimension of fc7')
    cmd:option('-att_hid_size', 512, 'the hidden size of the attention MLP; 0 if not using hidden layer')
    
    cmd:option('-val_size', 4000, 'Validation set size')
    cmd:option('-test_size', 4000, 'Test set size')

    cmd:option('-use_attention', true, 'Use attention or not')
    cmd:option('-use_noun', false, 'Use noun or not') -- true
    cmd:option('-use_cat', false, 'Use category or not. If true then will disgard words.')
    cmd:option('-reason_weight', 10.0, 'weight of reasoning loss')
    cmd:option('-gen_weight', 6.0) -- 30

    -- cmd:option('-use_reasoning', true, 'Use reasoning. Will use attention in default.')
    cmd:option('-model_pack', 'reason_att_copy_simp', 'the model package to use, can be reason_att, reasoning, or soft_att_lstm') --
    cmd:option('-reason_step', 8, 'Reasoning steps before the decoder')

    ------------ General options --------------------
    cmd:option('-data', 'data/', 'Path to dataset')
    cmd:option('-train_feat', 'train2014_inceptionv3_conv', 'Path to pre-extracted training image feature')
    cmd:option('-val_feat', 'val2014_inceptionv3_conv', 'Path to pre-extracted validation image feature')
    cmd:option('-test_feat', 'test2014_inceptionv3_conv', 'Path to pre-extracted test image feature')

    -- cmd:option('-train_feat', 'train2014_features_vgg_vd19_conv5', 'Path to pre-extracted training image feature')
    -- cmd:option('-val_feat', 'val2014_features_vgg_vd19_conv5', 'Path to pre-extracted validation image feature')
    -- cmd:option('-test_feat', 'test2014_features_vgg_vd19_conv5_2nd', 'Path to pre-extracted test image feature')

    -- cmd:option('-train_feat', 'train2014_vgg_vd16_conv5', 'Path to pre-extracted training image feature')
    -- cmd:option('-val_feat', 'val2014_vgg_vd16_conv5', 'Path to pre-extracted validation image feature')
    -- cmd:option('-test_feat', 'test2014_vgg_vd16_conv5', 'Path to pre-extracted test image feature')

    -- cmd:option('-train_fc7', 'train2014_features_vgg_vd19_fc7', 'Path to pre-extracted training fully connected 7')
    -- cmd:option('-val_fc7', 'val2014_features_vgg_vd19_fc7', 'Path to pre-extracted validation fully connected 7')
    -- cmd:option('-test_fc7', 'test2014_features_vgg_vd19_fc7_2nd', 'Path to pre-extracted test fully connected 7')

    cmd:option('-train_fc7', 'train2014_inceptionv3_fc', 'Path to pre-extracted training fully connected 7')
    cmd:option('-val_fc7', 'val2014_inceptionv3_fc', 'Path to pre-extracted validation fully connected 7')
    cmd:option('-test_fc7', 'test2014_inceptionv3_fc', 'Path to pre-extracted test fully connected 7')

    -- cmd:option('-train_fc7', 'train2014_inceptionv3_fc', 'Path to pre-extracted training fully connected 7')
    -- cmd:option('-val_fc7', 'val2014_inceptionv3_fc', 'Path to pre-extracted validation fully connected 7')
    -- cmd:option('-test_fc7', 'test2014_inceptionv3_fc', 'Path to pre-extracted test fully connected 7')

    -- cmd:option('-train_fc7_google', 'train2014_features_googlenet', 'Path to pre-extracted training fully connected 7')
    -- cmd:option('-val_fc7_google', 'val2014_features_googlenet', 'Path to pre-extracted validation fully connected 7')
    -- cmd:option('-test_fc7_google', 'test2014_features_googlenet', 'Path to pre-extracted test fully connected 7')

    -- cmd:option('-train_jpg', 'train2014_jpg')
    -- cmd:option('-val_jpg', 'val2014_jpg')
    -- cmd:option('-test_jpg', 'test2014_jpg')

    -- cmd:option('-train_jpg', 'train2014')
    -- cmd:option('-val_jpg', 'val2014')
    -- cmd:option('-test_jpg', 'test2014')

    cmd:option('-train_anno', 'annotations/captions_train2014.json', 'Path to training image annotaion file')
    cmd:option('-val_anno', 'annotations/captions_val2014.json', 'Path to validation image annotaion file')
    cmd:option('-cat_file', 'annotations/cats.parsed.txt', 'Path to the category file')
    cmd:option('-nGPU', 1, 'Index of GPU to use, 0 means CPU')
    cmd:option('-seed', 13, 'Random number seed') -- 13

    cmd:option('-id2noun_file', 'data/annotations/id2nouns.txt', 'Path to the id 2 nouns file')
    cmd:option('-arctic_dir', 'arctic-captions/splits', 'Path to index file')

    ------------ Training options --------------------
    cmd:option('-nEpochs', 100, 'Number of epochs in training') -- 100
    -- cmd:option('-eval_period', 12000, 'Every certain period, evaluate current model')
    -- cmd:option('-loss_period', 2400, 'Every given number of iterations, compute the loss on train and test')
    cmd:option('-batch_size', 32, 'Batch size in SGD') -- 32
    cmd:option('-val_batch_size', 10, 'Batch size for testing')
    cmd:option('-LR', 1e-2, 'Initial learning rate') -- 1e-2
    cmd:option('-cnn_LR', 0, 'Learning rate for cnn') --
    cmd:option('-truncate', 30, 'Text longer than this size gets truncated. -1 for no truncation.')
    cmd:option('-max_eval_batch', 50, 'max number of instances when calling comp error. 20000 = 4000 * 5')

    cmd:option('-save_file', true, 'whether save model file?') --
    cmd:option('-save_file_name', 'reason_att_copy_simp.model', 'file name for saving model') --
    cmd:option('-save_conv5_name', '12000.1e-5.fine.conv5.model')
    cmd:option('-save_fc7_name', '12000.1e-5.fine.fc7.model')

    cmd:option('-load_file', false, 'whether load model file?') --
    cmd:option('-load_vgg_file', false) --
    cmd:option('-load_file_name', 'reason_att_copy_simp_seed33.model') --
    cmd:option('-load_conv5_name', 'vgg_input_conv5_cunn.t7')
    cmd:option('-load_fc7_name', 'vgg_conv5_fc7_cunn.t7')

    cmd:option('-train_only', false, 'if true then use 80k, else use 110k')
    cmd:option('-early_stop', 'cider', 'can be cider or bleu')
    cmd:option('-bn', false)
    cmd:option('-use_google', false) -- false
    cmd:option('-cnn_relu', false)
    cmd:option('-cnn_dropout', true) -- true
    cmd:option('-normalize', false) -- false
    cmd:option('-reason_dropout', 0.0) -- 0.0
    cmd:option('-gen_dropout', 0.1) -- 0.1
    cmd:option('-load_glove', false) -- false

    cmd:option('-ensemble_LR', 1e-3) --
    cmd:option('-ensemble_train_mode', false) --
    
    ------------ Evaluation options --------------------
    -- cmd:option('-model', 'copy.all.val.8.w10.noun.model', 'Model to evaluate')
    -- cmd:option('-model', 'copy.google.val.all.8.noun.w10.model', 'Model to evaluate')
    cmd:option('-eval_algo', 'beam', 'Evaluation algorithm, beam or greedy')
    cmd:option('-beam_size', 3, 'Beam size in beam search') -- 3
    cmd:option('-val_max_len', 20, 'Max length in validation state')

    cmd:option('-test_mode', false, 'eval on test set if true') --
    cmd:option('-server_train_mode', false, 'eval on test of val, and use the rest for training')
    cmd:option('-server_test_mode', false, 'eval on server test set if true; if true then test_mode will be false.')

    cmd:option('-batch_num', 4)
    cmd:option('-cur_batch_num', 1)
    
    local opt = cmd:parse(arg or {})
    opt.eval_period = math.floor(3000 * 32 / 32) * 2
    opt.loss_period = math.floor(600 * 32 / 32)
    if opt.use_cat then opt.use_noun = false end
    if opt.server_test_mode then opt.test_mode = false end
    if opt.server_train_mode then opt.test_mode = false end
    if opt.ensemble_train_mode then
        opt.test_mode, opt.server_train_mode, opt.server_test_mode = false, false, false
    end
    opt.jpg = false
    if opt.model_pack == 'reason_att_copy_finetune' or opt.model_pack == 'reason_att_copy_fineboth' or opt.model_pack == 'reason_att_copy_fineconv' or opt.model_pack == 'reason_att_copy_simp_fineboth' then opt.jpg = true end
    opt.model = opt.load_file_name
    return opt
end

return M





