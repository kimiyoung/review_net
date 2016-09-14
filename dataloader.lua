require 'paths'
require 'image'
local anno_utils = require 'utils.anno_utils_filter'
local tablex = require('pl.tablex')

local M = {}
local DataLoader = torch.class('lstm.DataLoader', M)

function DataLoader:__init(opt)
    print('Initialize dataloader...')
    self.feat_dirs = {}
    table.insert(self.feat_dirs, paths.concat(opt.data, opt.train_feat))
    table.insert(self.feat_dirs, paths.concat(opt.data, opt.val_feat))
    self.anno_dirs = {}
    table.insert(self.anno_dirs, paths.concat(opt.data, opt.train_anno))
    table.insert(self.anno_dirs, paths.concat(opt.data, opt.val_anno))
    self.fc7_dirs = {}
    table.insert(self.fc7_dirs, paths.concat(opt.data, opt.train_fc7))
    table.insert(self.fc7_dirs, paths.concat(opt.data, opt.val_fc7))

    self.att_size = opt.att_size
    self.feat_size = opt.feat_size
    self.fc7_size = opt.fc7_size
    self.reason_step = opt.reason_step

    self.anno_utils = anno_utils
    self.use_noun = opt.use_noun
    self.truncate = opt.truncate
    self.use_cat = opt.use_cat
    self.use_google = opt.use_google
    self.jpg = opt.jpg

    -- Prepare captions
    self.id2file, self.train_ids, self.val_ids = anno_utils.read_dataset(self.feat_dirs, '.dat')
    self.id2fc7_file, _, _ = anno_utils.read_dataset(self.fc7_dirs, '.dat')
    self.id2captions, self.word2index, self.index2word, self.word_cnt = anno_utils.read_captions(self.anno_dirs, nil)
    opt.word2index = self.word2index
    if opt.use_cat then
        self.id2cats, self.cat_cnt = anno_utils.read_cats(self.cat_dir)
        opt.cat_cnt = self.cat_cnt
    end

    if opt.jpg then
        self.jpg_dirs = {}
        table.insert(self.jpg_dirs, paths.concat(opt.data, opt.train_jpg))
        table.insert(self.jpg_dirs, paths.concat(opt.data, opt.val_jpg))
        self.id2jpg, _, _ = anno_utils.read_dataset(self.jpg_dirs, '.dat')
    end

    -- self.annid2nouns = anno_utils.read_nouns(opt.id2noun_file, self.word2index)

    print('Dataset summary:')
    print('id2file size: ' .. tablex.size(self.id2file))
    print('id2captions size: ' .. tablex.size(self.id2captions))
    print('word2index size: '.. tablex.size(self.word2index))
    print('word_cnt: ' .. self.word_cnt)
    if opt.use_cat then
        print('cat_cnt: ' .. self.cat_cnt)
    end
    
    -- This is for model
    opt.word_cnt = self.word_cnt
    
    -- Split dataset into train, val and test
    -- self.train_set, self.val_set, self.test_set = anno_utils.split_dataset(self.val_ids, opt.val_size, opt.test_size)
    self.train_set, self.val_set, self.test_set = anno_utils.read_index_split(opt)
    if opt.test_mode then
        self.val_set, self.test_set = self.test_set, self.val_set
    end

    if opt.server_train_mode then
        local cur_n = #self.train_set
        for i = 1, #self.val_set do
            self.train_set[cur_n + i] = self.val_set[i]
        end
        self.val_set = self.test_set
    end

    if opt.server_test_mode then
        server_id2file, server_test_ids, _ = anno_utils.read_dataset({paths.concat(opt.data, opt.test_feat)},'.dat')
        server_id2fc7_file, _, _ = anno_utils.read_dataset({paths.concat(opt.data, opt.test_fc7)}, '.dat')
        self.val_set = server_test_ids
        self.id2file = server_id2file
        self.id2fc7_file = server_id2fc7_file
        if opt.jpg then
            self.id2jpg, _, _ = anno_utils.read_dataset({paths.concat(opt.data, opt.test_jpg)}, '.jpg')
        end
    end

    print('validation set size: ' .. tablex.size(self.val_set))
    print('test set size: ' .. tablex.size(self.test_set))
    
    -- local cur_n = #self.train_set
    -- for i = 1, #self.train_ids do
    --     self.train_set[cur_n + i] = self.train_ids[i]
    -- end
    
    print('train set size: ' .. tablex.size(self.train_set))

    -- Generate len2captions
    self.train_len2captions = anno_utils.gen_len2captions(self.train_set, self.id2captions)
    print('size of train_len2captions: ' .. tablex.size(self.train_len2captions))
    if not opt.server_test_mode then
        self.val_len2captions = anno_utils.gen_len2captions(self.val_set, self.id2captions)
        print('size of val_len2captions: ' .. tablex.size(self.val_len2captions))
    end
    
    local max_seq_len = 1  
    for k,v in pairs(self.train_len2captions) do
        -- print(k)
        -- if k > opt.max_seq_len then opt.max_seq_len = k end
        max_seq_len = math.max(max_seq_len, k)
    end               
    opt.max_seq_len = max_seq_len + 2
    print('Max sequence length is: ' .. opt.max_seq_len)
    -- os.exit()
end

--------------------------------------------------------
-- genrate training data
-- INPUT
-- batch: a table of captions
-- OUTPUT
-- image: a tensor, 196*512
--------------------------------------------------------
function DataLoader:gen_train_data(batch)
    -- local images = torch.CudaTensor(#batch, 512, 14, 14)
    local images = torch.CudaTensor(#batch, self.att_size, self.feat_size)
    local input_text = torch.CudaTensor(#batch, #batch[1][3] + 1)
    local output_text = torch.CudaTensor(#batch, #batch[1][3] + 1)
    local fc7_images = torch.CudaTensor(#batch, self.fc7_size)
    local fc7_google_images = torch.CudaTensor(#batch, 1024)

    local jpg
    if self.jpg then jpg = torch.CudaTensor(#batch, 3, 224, 224) end

    local noun_list
    if self.use_cat then
        noun_list = torch.Tensor(#batch, self.cat_cnt):zero()
    elseif self.use_noun then
        noun_list = torch.Tensor(#batch, self.word_cnt):zero()
    else
        noun_list = nil
    end

    for i = 1, #batch do
        -- caption = {id, caption}
        local id, ann_id, caption = batch[i][1], batch[i][2], batch[i][3]
        -- local file = files[id2index[id]]
        local file = self.id2file[id]
        local fc7_file = self.id2fc7_file[id]
        local fc7_google_file = self.id2fc7_google[id]
        -- from 512*14*14
        images[i]:copy(torch.load(file):reshape(self.feat_size, self.att_size):transpose(1, 2))
        fc7_images[i]:copy(torch.load(fc7_file))
        fc7_google_images[i]:copy(torch.load(fc7_google_file))

        if self.jpg then
            local jpg_file = self.id2jpg[id]
            jpg[i]:copy(torch.load(jpg_file))
        end

        local word_dict = {}
        for j = 1, #caption do
            input_text[i][j + 1] = caption[j]
            output_text[i][j] = caption[j]
            word_dict[caption[j]] = true
        end
        input_text[i][1] = anno_utils.START_NUM
        output_text[i][#caption + 1] = anno_utils.STOP_NUM
        if self.use_cat then
            if self.id2cats[id] ~= nil then
                for k, cat in ipairs(self.id2cats[id]) do
                    noun_list[i][k] = cat
                end
            end
        elseif self.use_noun then
            local ind = 1
            for k, _ in pairs(word_dict) do
                if k > anno_utils.NUM then
                    noun_list[i][ind] = k
                    ind = ind + 1
                end
            end
        end
    end
    return images, fc7_images, input_text, output_text, noun_list, fc7_google_images, jpg
end

function DataLoader:gen_train_jpg(batch)
    local jpg = torch.CudaTensor(#batch, 3, 224, 224)

    local input_text = torch.CudaTensor(#batch, #batch[1][3] + 1)
    local output_text = torch.CudaTensor(#batch, #batch[1][3] + 1)
    local noun_list
    if self.use_cat then
        noun_list = torch.Tensor(#batch, self.cat_cnt):zero()
    elseif self.use_noun then
        noun_list = torch.Tensor(#batch, self.word_cnt):zero()
    else
        noun_list = nil
    end

    for i = 1, #batch do
        -- caption = {id, caption}
        local id, ann_id, caption = batch[i][1], batch[i][2], batch[i][3]
        -- local file = files[id2index[id]]
        local jpg_file = self.id2jpg[id]
        -- from 512*14*14
        jpg[i]:copy(torch.load(jpg_file))
        local word_dict = {}
        for j = 1, #caption do
            input_text[i][j + 1] = caption[j]
            output_text[i][j] = caption[j]
            word_dict[caption[j]] = true
        end
        input_text[i][1] = anno_utils.START_NUM
        output_text[i][#caption + 1] = anno_utils.STOP_NUM
        if self.use_cat then
            if self.id2cats[id] ~= nil then
                for k, cat in ipairs(self.id2cats[id]) do
                    noun_list[i][k] = cat
                end
            end
        elseif self.use_noun then
            local ind = 1
            for k, _ in pairs(word_dict) do
                if k > anno_utils.NUM then
                    noun_list[i][ind] = k
                    ind = ind + 1
                end
            end
        end
    end
    return jpg, input_text, output_text, noun_list 
end   

function DataLoader:gen_test_data(j1, j2)
    local images = torch.CudaTensor(j2 - j1 + 1, self.att_size, self.feat_size)
    local fc7_images = torch.CudaTensor(j2 - j1 + 1, self.fc7_size)
    local fc7_google_images

    local jpg
    if self.jpg then jpg = torch.CudaTensor(j2 - j1 + 1, 3, 224, 224) end

    if self.use_google then fc7_google_images = torch.CudaTensor(j2 - j1 + 1, 1024) end
    for i = j1, j2 do
        local id = self.val_set[i]
        local file = self.id2file[id]
        local fc7_file = self.id2fc7_file[id]
        images[i - j1 + 1]:copy(torch.load(file):reshape(self.feat_size, self.att_size):transpose(1, 2))
        fc7_images[i - j1 + 1]:copy(torch.load(fc7_file))
        if self.use_google then
            local fc7_google_file = self.id2fc7_google[id]
            fc7_google_images[i - j1 + 1]:copy(torch.load(fc7_google_file))
        end
        if self.jpg then
            local jpg_file = self.id2jpg[id]
            jpg[i - j1 + 1]:copy(torch.load(jpg_file))
        end
    end
    return images, fc7_images, fc7_google_images, jpg
end

function DataLoader:gen_test_jpg(j1, j2)
    local jpg = torch.CudaTensor(j2 - j1 + 1, 3, 224, 224)
    for i = j1, j2 do
        local id = self.val_set[i]
        local jpg_file = self.id2jpg[id]
        jpg[i - j1 + 1]:copy(torch.load(jpg_file))
    end
    return jpg
end

                
-----------------------------------------------------
-- generate batch with same length for training
-----------------------------------------------------
function DataLoader:gen_batch(len2captions, batch_size)
    local batches = {}
    for len, captions in pairs(len2captions) do
        local i = 1; local j = batch_size
        while i <= #captions do
            j = j <= #captions and j or #captions
            local batch = {}
            for k = i, j do table.insert(batch, captions[k]) end
            table.insert(batches, batch)
            i = j + 1; j = i + batch_size - 1
        end
    end
    return batches
end

return M.DataLoader
                
                
