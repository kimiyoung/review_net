local MaskTable, parent = torch.class('nn.MaskTable', 'nn.Module')

function MaskTable:__init()
    parent.__init(self)
    self.gradInput = {}
end

function MaskTable:updateOutput(input)
    local x, mask = unpack(input)
    self.tout = self.tout or x.new()
    self.tout:resizeAs(mask):copy(mask)
    if x:dim() == 2 then
        self.output:resizeAs(x):copy(x):cmul(self.tout:view(x:size()[1], 1):expandAs(x))
    else
        self.output:resizeAs(x):copy(x):cmul(self.tout)
    end
    return self.output
end

function MaskTable:updateGradInput(input, gradOutput)
    local x, mask = unpack(input)
    for i = 1, 2 do self.gradInput[i] = self.gradInput[i] or x.new() end
    self.gradInput[2]:resizeAs(mask):zero()
    if x:dim() == 2 then
        self.gradInput[1]:resizeAs(x):copy(gradOutput):cmul(self.tout:view(x:size()[1], 1):expandAs(x))
    else
        self.gradInput[1]:resizeAs(x):copy(gradOutput):cmul(self.tout)
    end
    return self.gradInput
end

function MaskTable:clearState()
    if self.tout then self.tout:set() end
    return parent.clearState(self)
end