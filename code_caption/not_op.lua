
local NotOp, parent = torch.class('nn.NotOp', 'nn.Module')

function NotOp:__init()
    parent.__init(self)
end

function NotOp:updateOutput(input)
    self.output:resizeAs(input):copy(input):mul(-1):add(1)
    return self.output
end

function NotOp:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput):mul(-1)
    return self.gradInput
end
