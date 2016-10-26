---------------------------------------------------
-- Utility functioins for processing image captions
---------------------------------------------------

require 'paths'
local json = require "cjson"
local tablex = require('pl.tablex')

local utils = {}

-- constants
utils.START_NUM = 1
utils.STOP_NUM = 2
utils.UNK_NUM = 3
utils.NUM = 3

-----------------------------------------------------
-- conunt element number of a table
-----------------------------------------------------
function utils.len(t)
    local count = 0
    for _ in pairs(t) do count = count + 1 end
    return count
end


-----------------------------------------------------
-- shallow copy
-----------------------------------------------------
function utils.shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-----------------------------------------------------
-- concate two tables
-----------------------------------------------------
function utils.concat(t1, t2)
    local t = utils.shallowcopy(t1)
    for _, v in ipairs(t2) do
        table.insert(t, v)
    end
    return t
end

-----------------------------------------------------
-- a string split funtion
-----------------------------------------------------
function utils.mysplit(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} 
    local i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end


-----------------------------------------------------
-- parse image id from image file name
-----------------------------------------------------
function utils.get_image_id(filename)
    return tonumber(utils.mysplit(utils.mysplit(filename, '_')[3], '.')[1])
end

function utils.read_ids(filename)
    local ret = {}
    for line in io.open(filename):lines() do
        table.insert(ret, utils.get_image_id(line))
    end
    return ret
end

function utils.read_index_split(opt)
    -- local train_set = utils.read_ids(paths.concat(opt.arctic_dir, 'coco_train.txt'))
    -- local val_rest = utils.read_ids(paths.concat(opt.arctic_dir, 'coco_restval.txt'))
    -- if not opt.train_only then
    --     for _, id in ipairs(val_rest) do
    --         table.insert(train_set, id)
    --     end
    -- end
    -- local val_set = utils.read_ids(paths.concat(opt.arctic_dir, 'coco_val.txt'))
    -- local test_set = utils.read_ids(paths.concat(opt.arctic_dir, 'coco_test.txt'))
    local train_set, val_set, test_set = {}, {}, {}
    for line in io.open('eval/neuraltalk2/index.txt'):lines() do
        inputs = utils.mysplit(line)
        id = tonumber(inputs[1])
        if inputs[2] == 'train' then
            table.insert(train_set, id)
        elseif inputs[2] == 'val' then
            table.insert(val_set, id)
        else
            table.insert(test_set, id)
        end
    end

    return train_set, val_set, test_set
end

-----------------------------------------------------
-- generate (image id => filename) pairs
-- INPUT
-- dirs: a table of image paths
-- suffix: only read specified files
-- OUTPUT
-- id2file: a table, key is image id, value is filename
------------------------------------------------------
function utils.read_dataset(dirs, suffix)
    local id2file = {}; ret_ids = {}
    ret_ids[1] = {}; ret_ids[2] = {}   
    for k, dir in pairs(dirs) do
        for file in paths.files(dir) do
            if file:find(suffix) then
                -- print(file)
                -- print(utils)
                local id = utils.get_image_id(file)
                file = paths.concat(dir, file)
                id2file[id] = file
                table.insert(ret_ids[k], id)
            end
        end
    end
    return id2file, ret_ids[1], ret_ids[2]
end

------------------------------------------------------
-- generate (image id => caption) pairs
-- INPUT
-- filenames: a table of json file names
-- OUTPUT
-- id2captions: a table, key is image id, value is the 
--             caption of this image
------------------------------------------------------
function utils.read_captions(filenames, test)
    local MIN_WORD_CNT = 5

    local id2captions = {}
    local word2index = {}
    local word_cnt = 3

    local word2cnt = {}
    for _, filename in pairs(filenames) do
        local text = io.open(filename):read("*all")
        local annos = json.decode(text)['annotations']

        for _, anno in ipairs(annos) do
            local id, caption = anno['image_id'], anno['caption']:lower()
            caption = caption:gsub('[^%a%s]', ''):gsub('%s+', ' ')
            local seq = utils.mysplit(caption, ' ')
            for _, word in ipairs(seq) do
                if word2cnt[word] == nil then word2cnt[word] = 0 end
                word2cnt[word] = word2cnt[word] + 1
            end
        end
    end

    word2index['START'] = utils.START_NUM; word2index['STOP'] = utils.STOP_NUM; word2index['UNK'] = utils.UNK_NUM

    for k,filename in pairs(filenames) do
        local text = io.open(filename):read("*all")
        local annos = json.decode(text)['annotations']
        
        for _, anno in ipairs(annos) do 
            local id, caption, ann_id = anno['image_id'], anno['caption']:lower(), anno['id']
            caption = caption:gsub('[^%a%s]', ''):gsub('%s+', ' ')
            local seq = utils.mysplit(caption, ' ')
            
            local caption = {}
            for _, word in ipairs(seq) do
                if word2cnt[word] <= MIN_WORD_CNT then word = 'UNK' end

                if not test and word2index[word] == nil then
                    word_cnt = word_cnt + 1
                    word2index[word] = word_cnt
                end
                
                if word2index[word] ~= nil then
                    table.insert(caption, word2index[word])
                else
                    -- table.insert(caption, UNK_NUM)
                    table.insert(caption, utils.UNK_NUM)
                end        
            end
            
            if id2captions[id] == nil then
                id2captions[id] = {}
            end
            table.insert(id2captions[id], {ann_id, caption})
            
        end
    
    end
    
    -- get index to word
    index2word = {}
    for k,v in pairs(word2index) do 
        index2word[v] = k
    end
 
    return id2captions, word2index, index2word, word_cnt
end

function utils.read_cats(filename)
    local id2cats = {}
    local cat_cnt = 0

    for line in io.open(filename):lines() do
        local input = utils.mysplit(line)
        local image_id = tonumber(input[1])
        id2cats[image_id] = {}
        for i = 2, #input do
            local cat_id = tonumber(input[i])
            table.insert(id2cats[image_id], cat_id)
            cat_cnt = math.max(cat_cnt, cat_id)
        end
    end
    return id2cats, cat_cnt
end

function utils.read_nouns(filename, word2index)
    local id2noun = {}
    for line in io.open(filename):lines() do
        local input = utils.mysplit(line)
        local ann_id = tonumber(input[1])
        id2noun[ann_id] = {}
        for i = 3, #input do
            local noun = input[i]
            if word2index[noun] ~= nil then
                table.insert(id2noun[ann_id], word2index[noun])
            else
                table.insert(id2noun[ann_id], utils.UNK_NUM)
            end
        end
    end
    return id2noun
end

------------------------------------------------------
-- split images into train, val and test 
-- INPUT
-- ids: a table of json file names
-- OUTPUT
-- train_set, val_set, test_set
------------------------------------------------------
function utils.split_dataset(ids, VAL_SIZE, TEST_SIZE)
    torch.manualSeed(123)
    local n = tablex.size(ids)
    local TRAIN_SIZE = n - VAL_SIZE - TEST_SIZE
    assert(TRAIN_SIZE > 0)
    -- sort ids in a certain order
    local sorted_ids = {}
    for _, v in tablex.sortv(ids) do
        table.insert(sorted_ids, v)
    end

    -- generate a split of the data into TRAIN, TEST, VAL
    local perm = torch.randperm(n)

    local test_set = {}
    local val_set = {}
    local train_set = {}

    for k = 1,n do
        local index = perm[{k}]
        if k <= TEST_SIZE then
            table.insert(test_set, sorted_ids[index])
        elseif k <= TEST_SIZE+VAL_SIZE then
            table.insert(val_set, sorted_ids[index])
        else
            table.insert(train_set, sorted_ids[index])
        end
    end
    return train_set, val_set, test_set

end


-----------------------------------------------------------
-- genrate a table to gather captions with same length
-- INPUT
-- ids: a table with image ids
-- OUTPUT
-- train_set, val_set, test_set
-- format: key is length, value is a table of {id, captions}
------------------------------------------------------------
function utils.gen_len2captions(ids, id2captions)
    local len2captions = {}
    for id_k, id in pairs(ids) do
        for cap_k, caption_ in pairs(id2captions[id]) do
            -- here caption is a table
-- [[
--             local seq = mysplit(caption, ' ')
--             local len = #seq
--             if len2captions[len] == nil then
--                 len2captions[len] = {}
--             end
--             -- override caption as a table
--             caption = {}
--             for _, word in ipairs(seq) do
--                 if word2index[word] ~= nil then
--                     table.insert(caption, word2index[word])
--                 else
--                     table.insert(caption, UNK_NUM)
--                 end
--             end
-- ]]

            local ann_id, caption = unpack(caption_)
            local len = tablex.size(caption)
            
            if len2captions[len] == nil then
                len2captions[len] = {}
            end
            
            table.insert(len2captions[len], {id, ann_id, caption})
            
        end
    end
    return len2captions
end

return utils


