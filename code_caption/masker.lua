
local Masker, parent = torch.class('nn.Masker', 'nn.Module')

function Masker:__init()
    parent.__init(self)
    self.gradInput = {}
end

function Masker:updateOutput(input)
    local x, mask = unpack(input)
    self.index_cpu = torch.range(1, mask:size()[1])
    self.index = self.index or x.new()
    self.index:resize(mask:size()):copy(self.index_cpu)
    self.masked_index = x:index(1, self.index[mask])
    self.output:resizeAs(self.masked_index):copy(self.masked_index)
    return self.output
end

function Masker:updateGradInput(input, gradOutput)
    local x, mask = unpack(input)
    for i = 1, 2 do self.gradInput[i] = self.gradInput[i] or x.new() end
    self.gradInput[1]:resizeAs(x):zero()
    local pt = 1
    for i = 1, x:size()[1] do
        if mask[i] == 1 then
            self.gradInput[1][i]:copy(gradOutput[pt])
            pt = pt + 1
        end
    end
    self.gradInput[2]:resizeAs(mask):zero()
    return self.gradInput
end

function Masker:clearState()
    return nn.utils.clear(self, {'index_cpu', 'index', 'masked_index'})
end
