
local data = {}

data.CODE_TRUNCATE = 300 -- 300
data.COMMENT_TRUNCATE = 300 -- 300
data.BATCH_SIZE = 5
data.EOS_NUM = 1
data.RES_NUM = 1

function data.string_split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function data.indexing(filenames)
    local code_set, comment_set = {}, {}
    for _, filename in ipairs(filenames) do
        for line in io.open(filename):lines() do
            local inputs = data.string_split(line, '\t')
            local code_seq, comment_seq = data.string_split(inputs[1]), data.string_split(inputs[2])
            for _, code_token in ipairs(code_seq) do code_set[code_token] = true end
            for _, comment_token in ipairs(comment_seq) do comment_set[comment_token] = true end
        end
    end

    -- the first word_cnt tokens are comment tokens; additional code tokens are indexed from word_cnt + 1 to token_cnt
    -- data.RES_NUM is the number of reserved tokens (e.g. EOS), which take the indices starting from 1
    local token2index, index2token, token_cnt, word_cnt = {}, {}, data.RES_NUM, 0
    token2index['EOS'] = data.EOS_NUM
    for token, _ in pairs(comment_set) do
        token_cnt = token_cnt + 1
        token2index[token] = token_cnt
    end
    word_cnt = token_cnt
    for token, _ in pairs(code_set) do
        if token2index[token] == nil then
            token_cnt = token_cnt + 1
            token2index[token] = token_cnt
        end
    end
    for k, v in pairs(token2index) do
        index2token[v] = k
    end

    return token2index, index2token, token_cnt, word_cnt
end

function data.prepare_data(filename, token2index, word_cnt)
    local seq_data = {}
    for line in io.open(filename):lines() do
        local inputs = data.string_split(line, '\t')
        local code_seq, comment_seq = data.string_split(inputs[1]), data.string_split(inputs[2])
        table.insert(seq_data, {code_seq = code_seq, comment_seq = comment_seq})
    end
    table.sort(seq_data, function(a, b) return #a.comment_seq < #b.comment_seq end)
    local batches = {}
    local i, j = 1, 1
    while i <= #seq_data do
        j = math.min(i + data.BATCH_SIZE - 1, #seq_data)
        local max_code_len, max_comment_len = 0, 0
        for k = i, j do
            -- seq length + 1 for eos
            max_code_len = math.max(max_code_len, #seq_data[k].code_seq + 1)
            max_comment_len = math.max(max_comment_len, #seq_data[k].comment_seq + 1)
        end
        max_code_len = math.min(max_code_len, data.CODE_TRUNCATE)
        max_comment_len = math.min(max_comment_len, data.COMMENT_TRUNCATE)
        local t_size = j - i + 1
        -- code_matrix: code seq
        -- comment matrix: EOS + code seq
        -- comment next: code seq + EOS
        -- mask indices correspond to "matrix"
        local code_matrix, comment_matrix = torch.Tensor(t_size, max_code_len):fill(data.EOS_NUM), torch.Tensor(t_size, max_comment_len):fill(data.EOS_NUM)
        local code_mask, comment_mask = torch.Tensor(t_size, max_code_len):zero(), torch.Tensor(t_size, max_comment_len):zero()
        local comment_next = torch.Tensor(t_size, max_comment_len):fill(data.EOS_NUM)
        local word_label = torch.Tensor(t_size, word_cnt):zero()
        for k = i, j do
            local word_dict = {}
            local t_code_len = math.min(#seq_data[k].code_seq, max_code_len)
            for l = 1, t_code_len do
                code_matrix[k - i + 1][l] = token2index[seq_data[k].code_seq[l]]
                code_mask[k - i + 1][l] = 1
            end
            local t_comment_len = math.min(#seq_data[k].comment_seq, max_comment_len - 1)
            for l = 1, t_comment_len do
                comment_matrix[k - i + 1][l + 1] = token2index[seq_data[k].comment_seq[l]]
                comment_next[k - i + 1][l] = token2index[seq_data[k].comment_seq[l]]
                comment_mask[k - i + 1][l + 1] = 1
                word_dict[comment_next[k - i + 1][l]] = true
            end
            comment_mask[k - i + 1][1] = 1
            local ind = 1
            for key, _ in pairs(word_dict) do
                word_label[k - i + 1][ind] = key
                ind = ind + 1
            end
        end
        table.insert(batches, {code_matrix, code_mask, comment_matrix, comment_mask, comment_next, word_label})
        i = j + 1
    end
    return batches
end

return data
