import torch

model = torch.load("./trained_models/flip_sugar_gru_no_vision.pth")
model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print "loaded..."
print model.pos_x
print ""
#print model.ref_descriptor_vec
print ""
print count_parameters(model), "is num params"